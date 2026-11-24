//! Zero-dependency JSON-RPC 2.0 Stdio MCP Server for `loraspectra` (`loraspectra mcp`).
//!
//! Supports both newline-delimited JSON-RPC and LSP-style `Content-Length:` framed messages
//! so AI coding and ML agents (Claude Code, Cursor, Windsurf, Codex, Antigravity) can inspect
//! `.safetensors` LoRA checkpoints, run QR-Core SVD spectral audits, analyze LoRA+ factor
//! imbalance, compute AdaLoRA water-filling rank schedules, and run multi-checkpoint batch
//! leaderboards directly from chat with zero Python/PyTorch overhead.

use std::collections::BTreeMap;
use std::io::{self, BufRead, BufReader, Read, Write};
use std::path::PathBuf;

use crate::adalora::{optimize_rank_schedule, render_rank_schedule_json};
use crate::audit::{audit_adapter, audit_adapter_integrity, render_audit_json};
use crate::batch::{discover_checkpoints, render_batch_json, run_batch_audit};
use crate::loraplus::{analyze_loraplus, render_loraplus_json};
use crate::remote::resolve_source_path;
use crate::safetensors::{
    extract_dora_scales, extract_lora_alphas, extract_lora_layers, inspect_adapter_architecture,
    read_safetensors,
};
use crate::sarif::LORASPECTRA_SARIF_RULES;

const PROTOCOL_VERSION: &str = "2025-06-18";

#[derive(Debug, Clone, PartialEq)]
enum JsonVal {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    Arr(Vec<JsonVal>),
    Obj(BTreeMap<String, JsonVal>),
}

impl JsonVal {
    fn as_str(&self) -> Option<&str> {
        match self {
            JsonVal::Str(s) => Some(s.as_str()),
            _ => None,
        }
    }

    fn as_f64(&self) -> Option<f64> {
        match self {
            JsonVal::Num(n) => Some(*n),
            _ => None,
        }
    }

    fn as_obj(&self) -> Option<&BTreeMap<String, JsonVal>> {
        match self {
            JsonVal::Obj(m) => Some(m),
            _ => None,
        }
    }

    fn as_arr(&self) -> Option<&[JsonVal]> {
        match self {
            JsonVal::Arr(a) => Some(a.as_slice()),
            _ => None,
        }
    }

    fn to_json_literal(&self) -> String {
        match self {
            JsonVal::Null => "null".to_string(),
            JsonVal::Bool(b) => if *b { "true" } else { "false" }.to_string(),
            JsonVal::Num(n) => {
                if n.fract() == 0.0 && n.abs() < 1e15 {
                    format!("{}", *n as i64)
                } else {
                    format!("{}", n)
                }
            }
            JsonVal::Str(s) => format!("\"{}\"", json_escape(s)),
            JsonVal::Arr(items) => {
                let parts: Vec<String> = items.iter().map(|v| v.to_json_literal()).collect();
                format!("[{}]", parts.join(","))
            }
            JsonVal::Obj(map) => {
                let parts: Vec<String> = map
                    .iter()
                    .map(|(k, v)| format!("\"{}\":{}", json_escape(k), v.to_json_literal()))
                    .collect();
                format!("{{{}}}", parts.join(","))
            }
        }
    }
}

struct JsonParser<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> JsonParser<'a> {
    fn new(input: &'a str) -> Self {
        Self { bytes: input.as_bytes(), pos: 0 }
    }

    fn skip_ws(&mut self) {
        while self.pos < self.bytes.len() && matches!(self.bytes[self.pos], b' ' | b'\t' | b'\r' | b'\n') {
            self.pos += 1;
        }
    }

    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }

    fn parse_value(&mut self) -> Result<JsonVal, String> {
        self.skip_ws();
        match self.peek() {
            Some(b'"') => self.parse_string().map(JsonVal::Str),
            Some(b'{') => self.parse_object(),
            Some(b'[') => self.parse_array(),
            Some(b't') => self.expect_lit(b"true", JsonVal::Bool(true)),
            Some(b'f') => self.expect_lit(b"false", JsonVal::Bool(false)),
            Some(b'n') => self.expect_lit(b"null", JsonVal::Null),
            Some(c) if c == b'-' || c.is_ascii_digit() => self.parse_number(),
            Some(c) => Err(format!("Unexpected character '{}' at byte {}", c as char, self.pos)),
            None => Err("Unexpected end of JSON input".to_string()),
        }
    }

    fn expect_lit(&mut self, lit: &[u8], val: JsonVal) -> Result<JsonVal, String> {
        if self.bytes.get(self.pos..self.pos + lit.len()) == Some(lit) {
            self.pos += lit.len();
            Ok(val)
        } else {
            Err(format!("Invalid literal at byte {}", self.pos))
        }
    }

    fn parse_string(&mut self) -> Result<String, String> {
        if self.peek() != Some(b'"') {
            return Err("Expected '\"'".to_string());
        }
        self.pos += 1;
        let mut out = String::new();
        while self.pos < self.bytes.len() {
            let b = self.bytes[self.pos];
            self.pos += 1;
            if b == b'"' {
                return Ok(out);
            }
            if b == b'\\' {
                if self.pos >= self.bytes.len() {
                    return Err("Unterminated escape sequence".to_string());
                }
                let esc = self.bytes[self.pos];
                self.pos += 1;
                match esc {
                    b'"' => out.push('"'),
                    b'\\' => out.push('\\'),
                    b'/' => out.push('/'),
                    b'b' => out.push('\u{0008}'),
                    b'f' => out.push('\u{000C}'),
                    b'n' => out.push('\n'),
                    b'r' => out.push('\r'),
                    b't' => out.push('\t'),
                    b'u' => {
                        if self.pos + 4 > self.bytes.len() {
                            return Err("Incomplete \\uXXXX escape".to_string());
                        }
                        let hex = std::str::from_utf8(&self.bytes[self.pos..self.pos + 4])
                            .map_err(|_| "Invalid UTF-8 in \\u escape".to_string())?;
                        let code = u32::from_str_radix(hex, 16).map_err(|_| "Invalid hex in \\u escape".to_string())?;
                        if let Some(ch) = char::from_u32(code) {
                            out.push(ch);
                        }
                        self.pos += 4;
                    }
                    other => out.push(other as char),
                }
            } else {
                // Handle UTF-8 slices safely
                let start = self.pos - 1;
                let mut end = self.pos;
                while end < self.bytes.len() && (self.bytes[end] & 0xC0) == 0x80 {
                    end += 1;
                }
                let s = std::str::from_utf8(&self.bytes[start..end])
                    .map_err(|_| "Invalid UTF-8 sequence in string".to_string())?;
                out.push_str(s);
                self.pos = end;
            }
        }
        Err("Unterminated JSON string".to_string())
    }

    fn parse_number(&mut self) -> Result<JsonVal, String> {
        let start = self.pos;
        if self.peek() == Some(b'-') {
            self.pos += 1;
        }
        while self.pos < self.bytes.len() && self.bytes[self.pos].is_ascii_digit() {
            self.pos += 1;
        }
        if self.peek() == Some(b'.') {
            self.pos += 1;
            while self.pos < self.bytes.len() && self.bytes[self.pos].is_ascii_digit() {
                self.pos += 1;
            }
        }
        if matches!(self.peek(), Some(b'e' | b'E')) {
            self.pos += 1;
            if matches!(self.peek(), Some(b'+' | b'-')) {
                self.pos += 1;
            }
            while self.pos < self.bytes.len() && self.bytes[self.pos].is_ascii_digit() {
                self.pos += 1;
            }
        }
        let s = std::str::from_utf8(&self.bytes[start..self.pos]).map_err(|_| "Bad number UTF-8".to_string())?;
        let n = s.parse::<f64>().map_err(|e| format!("Invalid number '{}': {}", s, e))?;
        Ok(JsonVal::Num(n))
    }

    fn parse_array(&mut self) -> Result<JsonVal, String> {
        self.pos += 1; // '['
        let mut items = Vec::new();
        self.skip_ws();
        if self.peek() == Some(b']') {
            self.pos += 1;
            return Ok(JsonVal::Arr(items));
        }
        loop {
            items.push(self.parse_value()?);
            self.skip_ws();
            match self.peek() {
                Some(b',') => self.pos += 1,
                Some(b']') => {
                    self.pos += 1;
                    return Ok(JsonVal::Arr(items));
                }
                _ => return Err(format!("Expected ',' or ']' at byte {}", self.pos)),
            }
        }
    }

    fn parse_object(&mut self) -> Result<JsonVal, String> {
        self.pos += 1; // '{'
        let mut map = BTreeMap::new();
        self.skip_ws();
        if self.peek() == Some(b'}') {
            self.pos += 1;
            return Ok(JsonVal::Obj(map));
        }
        loop {
            self.skip_ws();
            let key = self.parse_string()?;
            self.skip_ws();
            if self.peek() != Some(b':') {
                return Err(format!("Expected ':' after key '{}' at byte {}", key, self.pos));
            }
            self.pos += 1;
            let val = self.parse_value()?;
            map.insert(key, val);
            self.skip_ws();
            match self.peek() {
                Some(b',') => self.pos += 1,
                Some(b'}') => {
                    self.pos += 1;
                    return Ok(JsonVal::Obj(map));
                }
                _ => return Err(format!("Expected ',' or '}}' at byte {}", self.pos)),
            }
        }
    }
}

fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

fn rpc_result(id_lit: &str, result_json: &str) -> String {
    format!(r#"{{"jsonrpc":"2.0","id":{},"result":{}}}"#, id_lit, result_json)
}

fn rpc_error(id_lit: &str, code: i32, msg: &str) -> String {
    format!(
        r#"{{"jsonrpc":"2.0","id":{},"error":{{"code":{},"message":"{}"}}}}"#,
        id_lit,
        code,
        json_escape(msg)
    )
}

fn tool_text_response(id_lit: &str, text: &str, is_error: bool) -> String {
    let res = format!(
        r#"{{"content":[{{"type":"text","text":"{}"}}],"isError":{}}}"#,
        json_escape(text),
        if is_error { "true" } else { "false" }
    );
    rpc_result(id_lit, &res)
}

fn tools_list_json() -> &'static str {
    r#"{"tools":[
{"name":"loraspectra_info","description":"Inspect a .safetensors LoRA checkpoint (local path or hf://org/repo URI) for architecture, naming format, tensor count, rank, and dtype.","inputSchema":{"type":"object","properties":{"source":{"type":"string","description":"Local .safetensors path or hf://org/repo URI"}},"required":["source"],"additionalProperties":false}},
{"name":"loraspectra_audit","description":"Run exact QR-Core SVD spectral audit across all LoRA layers (effective rank, stable rank, Marchenko-Pastur noise floor, rank collapse & dead layer detection).","inputSchema":{"type":"object","properties":{"source":{"type":"string","description":"Local .safetensors path or hf://org/repo URI"}},"required":["source"],"additionalProperties":false}},
{"name":"loraspectra_loraplus","description":"Audit LoRA+ variational nuclear-norm factor imbalance (LS047-LS048) and compute per-block recommended learning-rate ratio lambda* = eta_B / eta_A.","inputSchema":{"type":"object","properties":{"source":{"type":"string"},"imbalance_warn":{"type":"number"},"cond_warn":{"type":"number"}},"required":["source"],"additionalProperties":false}},
{"name":"loraspectra_rank_schedule","description":"Compute AdaLoRA spectral-importance water-filling heterogeneous rank_pattern & alpha_pattern schedule (LS049-LS050).","inputSchema":{"type":"object","properties":{"source":{"type":"string"},"target_avg_rank":{"type":"number"},"min_rank":{"type":"integer"},"max_rank":{"type":"integer"}},"required":["source"],"additionalProperties":false}},
{"name":"loraspectra_batch_audit","description":"Recursively scan directories or checkpoint files and rank all LoRA adapters on a 0-100 Spectral Health Leaderboard.","inputSchema":{"type":"object","properties":{"paths":{"type":"array","items":{"type":"string"}},"max_depth":{"type":"integer"}},"required":["paths"],"additionalProperties":false}},
{"name":"loraspectra_list_rules","description":"List all 50 OASIS SARIF v2.1.0 LoRASpectra diagnostic rules (LS001-LS050).","inputSchema":{"type":"object","properties":{},"additionalProperties":false}}
]}"#
}

/// Handle a single JSON-RPC 2.0 request string. Returns `None` for notifications.
pub fn handle_mcp_message(raw: &str) -> Option<String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return None;
    }
    let mut parser = JsonParser::new(trimmed);
    let root = match parser.parse_value() {
        Ok(v) => v,
        Err(e) => return Some(rpc_error("null", -32700, &format!("Malformed JSON-RPC: {}", e))),
    };
    let Some(obj) = root.as_obj() else {
        return Some(rpc_error("null", -32600, "JSON-RPC payload must be an object."));
    };
    let Some(id_val) = obj.get("id") else {
        return None; // notification
    };
    let id_lit = id_val.to_json_literal();
    let Some(method) = obj.get("method").and_then(|v| v.as_str()) else {
        return Some(rpc_error(&id_lit, -32600, "Missing JSON-RPC method."));
    };

    let empty_map = BTreeMap::new();
    let params = obj.get("params").and_then(|v| v.as_obj()).unwrap_or(&empty_map);

    match method {
        "initialize" => {
            let res = format!(
                r#"{{"protocolVersion":"{}","serverInfo":{{"name":"loraspectra","version":"{}"}},"capabilities":{{"tools":{{"listChanged":false}}}}}}"#,
                PROTOCOL_VERSION,
                env!("CARGO_PKG_VERSION")
            );
            Some(rpc_result(&id_lit, &res))
        }
        "ping" => Some(rpc_result(&id_lit, "{}")),
        "tools/list" => Some(rpc_result(&id_lit, tools_list_json())),
        "tools/call" => Some(handle_tool_call(&id_lit, params)),
        other => Some(rpc_error(&id_lit, -32601, &format!("Method not found: {}", other))),
    }
}

fn handle_tool_call(id_lit: &str, params: &BTreeMap<String, JsonVal>) -> String {
    let Some(name) = params.get("name").and_then(|v| v.as_str()) else {
        return rpc_error(id_lit, -32602, "Missing string 'name' in tools/call.");
    };
    let empty_map = BTreeMap::new();
    let args = params.get("arguments").and_then(|v| v.as_obj()).unwrap_or(&empty_map);

    let outcome: Result<String, String> = match name {
        "loraspectra_list_rules" => {
            let items: Vec<String> = LORASPECTRA_SARIF_RULES
                .iter()
                .map(|r| {
                    format!(
                        r#"{{"id":"{}","name":"{}","level":"{}","shortDescription":"{}"}}"#,
                        json_escape(r.id),
                        json_escape(r.name),
                        json_escape(r.default_level),
                        json_escape(r.short_description)
                    )
                })
                .collect();
            Ok(format!(
                r#"{{"rule_count":{},"rules":[{}]}}"#,
                LORASPECTRA_SARIF_RULES.len(),
                items.join(",")
            ))
        }
        "loraspectra_info" => {
            let src = args.get("source").and_then(|v| v.as_str()).ok_or_else(|| "Missing 'source' string".to_string());
            src.and_then(|s| {
                let path = resolve_source_path(s)?;
                let tensors = read_safetensors(&path)?;
                let layers = extract_lora_layers(&tensors)?;
                let alphas = extract_lora_alphas(&tensors);
                let dora = extract_dora_scales(&tensors);
                let arch = inspect_adapter_architecture(&layers, &alphas);
                Ok(format!(
                    r#"{{"source":"{}","resolved_path":"{}","tensor_count":{},"lora_layer_count":{},"naming_format":"{}","inferred_architecture":"{}","min_rank":{},"max_rank":{},"primary_dtype":"{}","total_parameters":{},"total_bytes":{},"has_dora":{}}}"#,
                    json_escape(s),
                    json_escape(&path.to_string_lossy()),
                    tensors.len(),
                    layers.len(),
                    json_escape(arch.naming_format),
                    json_escape(arch.inferred_architecture),
                    arch.min_rank,
                    arch.max_rank,
                    arch.primary_dtype.as_str(),
                    arch.total_parameters,
                    arch.total_bytes,
                    if !dora.is_empty() { "true" } else { "false" }
                ))
            })
        }
        "loraspectra_audit" => {
            let src = args.get("source").and_then(|v| v.as_str()).ok_or_else(|| "Missing 'source' string".to_string());
            src.and_then(|s| {
                let path = resolve_source_path(s)?;
                let tensors = read_safetensors(&path)?;
                let layers = extract_lora_layers(&tensors)?;
                let alphas = extract_lora_alphas(&tensors);
                let dora = extract_dora_scales(&tensors);
                let summary = audit_adapter(&layers);
                let integrity = audit_adapter_integrity(&layers, &tensors, &alphas, &dora, None);
                Ok(render_audit_json(s, &summary, &integrity))
            })
        }
        "loraspectra_loraplus" => {
            let src = args.get("source").and_then(|v| v.as_str()).ok_or_else(|| "Missing 'source' string".to_string());
            src.and_then(|s| {
                let path = resolve_source_path(s)?;
                let tensors = read_safetensors(&path)?;
                let layers = extract_lora_layers(&tensors)?;
                let imb = args.get("imbalance_warn").and_then(|v| v.as_f64()).unwrap_or(1.5);
                let cond = args.get("cond_warn").and_then(|v| v.as_f64()).unwrap_or(50.0);
                let summary = analyze_loraplus(&layers, s, imb, cond);
                Ok(render_loraplus_json(&summary))
            })
        }
        "loraspectra_rank_schedule" => {
            let src = args.get("source").and_then(|v| v.as_str()).ok_or_else(|| "Missing 'source' string".to_string());
            src.and_then(|s| {
                let path = resolve_source_path(s)?;
                let tensors = read_safetensors(&path)?;
                let layers = extract_lora_layers(&tensors)?;
                let target = args.get("target_avg_rank").and_then(|v| v.as_f64()).unwrap_or(16.0);
                let min_r = args.get("min_rank").and_then(|v| v.as_f64()).unwrap_or(4.0).max(1.0) as usize;
                let max_r = args.get("max_rank").and_then(|v| v.as_f64()).unwrap_or(64.0).max(min_r as f64) as usize;
                let summary = optimize_rank_schedule(&layers, s, target, min_r, max_r);
                Ok(render_rank_schedule_json(&summary))
            })
        }
        "loraspectra_batch_audit" => {
            let raw_paths = args.get("paths").and_then(|v| v.as_arr()).ok_or_else(|| "Missing 'paths' array".to_string());
            raw_paths.and_then(|arr| {
                let inputs: Vec<PathBuf> = arr
                    .iter()
                    .filter_map(|v| v.as_str().map(PathBuf::from))
                    .collect();
                let depth = args.get("max_depth").and_then(|v| v.as_f64()).unwrap_or(6.0).max(1.0) as usize;
                let discovered = discover_checkpoints(&inputs, depth);
                let report = run_batch_audit(&discovered);
                Ok(render_batch_json(&report))
            })
        }
        unknown => return rpc_error(id_lit, -32602, &format!("Unknown tool: {}", unknown)),
    };

    match outcome {
        Ok(text) => tool_text_response(id_lit, &text, false),
        Err(err) => tool_text_response(id_lit, &err, true),
    }
}

/// Serve JSON-RPC 2.0 MCP traffic over stdin/stdout (supporting both newline & Content-Length framing).
pub fn serve_stdio() -> io::Result<()> {
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut reader = BufReader::new(stdin.lock());
    let mut writer = stdout.lock();

    loop {
        let mut line = String::new();
        let n = reader.read_line(&mut line)?;
        if n == 0 {
            break;
        }
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let (body, framed) = if trimmed.to_ascii_lowercase().starts_with("content-length:") {
            let len: usize = trimmed
                .split_once(':')
                .and_then(|(_, v)| v.trim().parse().ok())
                .unwrap_or(0);
            // Consume remaining headers until empty line
            loop {
                let mut hdr = String::new();
                if reader.read_line(&mut hdr)? == 0 || hdr.trim().is_empty() {
                    break;
                }
            }
            let mut buf = vec![0u8; len];
            reader.read_exact(&mut buf)?;
            (String::from_utf8_lossy(&buf).into_owned(), true)
        } else {
            (trimmed.to_string(), false)
        };

        if let Some(resp) = handle_mcp_message(&body) {
            if framed {
                write!(writer, "Content-Length: {}\r\n\r\n{}", resp.len(), resp)?;
            } else {
                writeln!(writer, "{}", resp)?;
            }
            writer.flush()?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::safetensors::{layers_to_tensor_map, write_safetensors, DType, LoraLayer, Tensor2D};
    use crate::Rng64;

    #[test]
    fn mcp_server_handles_initialize_tools_list_rules_and_live_audit() {
        let init = handle_mcp_message(r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}"#).unwrap();
        assert!(init.contains(r#""name":"loraspectra""#));
        assert!(init.contains(PROTOCOL_VERSION));

        let tools = handle_mcp_message(r#"{"jsonrpc":"2.0","id":2,"method":"tools/list"}"#).unwrap();
        assert!(tools.contains("loraspectra_audit"));
        assert!(tools.contains("loraspectra_loraplus"));
        assert!(tools.contains("loraspectra_rank_schedule"));
        assert!(tools.contains("loraspectra_batch_audit"));

        let rules = handle_mcp_message(
            r#"{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"loraspectra_list_rules","arguments":{}}}"#,
        )
        .unwrap();
        assert!(rules.contains(r#"rule_count\":50"#));
        assert!(rules.contains("LS050"));

        // Write a tiny synthetic safetensors adapter and audit via MCP!
        let path = std::env::temp_dir().join("loraspectra_mcp_test_v111.safetensors");
        let mut rng = Rng64::new(99);
        let layer = LoraLayer {
            base_name: "model.layers.0.self_attn.q_proj".to_string(),
            a_key: "model.layers.0.self_attn.q_proj.lora_A.weight".to_string(),
            b_key: "model.layers.0.self_attn.q_proj.lora_B.weight".to_string(),
            a: Tensor2D { rows: 4, cols: 16, dtype: DType::F32, data: (0..64).map(|_| rng.next_gaussian() * 0.2).collect() },
            b: Tensor2D { rows: 16, cols: 4, dtype: DType::F32, data: (0..64).map(|_| rng.next_gaussian() * 0.2).collect() },
        };
        write_safetensors(&path, &layers_to_tensor_map(&[layer])).unwrap();
        let req = format!(
            r#"{{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{{"name":"loraspectra_loraplus","arguments":{{"source":"{}"}}}}}}"#,
            json_escape(&path.to_string_lossy())
        );
        let lp_resp = handle_mcp_message(&req).unwrap();
        assert!(lp_resp.contains(r#""isError":false"#));
        assert!(lp_resp.contains("mean_imbalance_index"));
        let _ = std::fs::remove_file(&path);
    }
}

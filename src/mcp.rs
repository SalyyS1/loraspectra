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
use crate::card::generate_model_card;
use crate::compat::{check_serving_compatibility, render_compat_json};
use crate::diff::{diff_adapters_with_dora, render_diff_json};
use crate::doctor::{render_doctor_json, run_doctor};
use crate::loraplus::{analyze_loraplus, render_loraplus_json};
use crate::merge_check::{analyze_multi_adapter_merge, render_merge_check_json, AdapterMergeInput};
use crate::quant_plan::{build_quant_plan, render_quant_plan_json};
use crate::remote::resolve_source_path;
use crate::safetensors::{
    extract_dora_scales, extract_lora_alphas, extract_lora_layers, inspect_adapter_architecture,
    read_safetensors,
};
use crate::sarif::LORASPECTRA_SARIF_RULES;
use crate::sensitivity::{analyze_sensitivity, render_sensitivity_json};
use crate::trajectory::{analyze_trajectory, render_trajectory_json};

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
{"name":"loraspectra_sensitivity","description":"Audit layer-wise spectral sensitivity across Attention vs MLP and Early/Mid/Deep transformer blocks for brittle rank-1 dominance (LS051) and energy spikes (LS052).","inputSchema":{"type":"object","properties":{"source":{"type":"string"},"dominance_warn_pct":{"type":"number"},"spike_multiplier":{"type":"number"}},"required":["source"],"additionalProperties":false}},
{"name":"loraspectra_doctor","description":"Run an all-in-one executive adapter health diagnosis combining SVD utilization, integrity, LoRA+ imbalance, sensitivity/forgetting risk, and AdaLoRA budget into an A-F grade.","inputSchema":{"type":"object","properties":{"source":{"type":"string"}},"required":["source"],"additionalProperties":false}},
{"name":"loraspectra_diff","description":"Compute exact O(r^2*(d_in+d_out)) Frobenius drift, Grassmannian subspace overlap, and DoRA magnitude drift between two .safetensors LoRA checkpoints.","inputSchema":{"type":"object","properties":{"source_a":{"type":"string"},"source_b":{"type":"string"}},"required":["source_a","source_b"],"additionalProperties":false}},
{"name":"loraspectra_merge_check","description":"Evaluate N >= 2 LoRA adapters jointly prior to TIES/DARE/linear/ortho-svd merging for subspace collision, spectral norm inflation, and incompatible shapes.","inputSchema":{"type":"object","properties":{"sources":{"type":"array","items":{"type":"string"}}},"required":["sources"],"additionalProperties":false}},
{"name":"loraspectra_model_card","description":"Generate a ready-to-paste HuggingFace README.md Spectral Audit Model Card with shields.io badges and bottleneck tables from a local .safetensors or hf://org/repo URI.","inputSchema":{"type":"object","properties":{"source":{"type":"string"}},"required":["source"],"additionalProperties":false}},
{"name":"loraspectra_trajectory","description":"Analyze an ordered sequence of >=2 training checkpoints for Frobenius velocity, effective-rank evolution, Grassmannian subspace convergence, rank collapse, and early-stopping recommendation.","inputSchema":{"type":"object","properties":{"sources":{"type":"array","items":{"type":"string"}}},"required":["sources"],"additionalProperties":false}},
{"name":"loraspectra_quant_plan","description":"Generate a quantization-aware mixed-precision LoRA rank compression plan (nf4/fp4/int8/fp8/bf16/fp16) projecting byte savings and flagging brittle rank-1 outlier layers.","inputSchema":{"type":"object","properties":{"source":{"type":"string"},"target_quant":{"type":"string"},"target_energy":{"type":"number"}},"required":["source"],"additionalProperties":false}},
{"name":"loraspectra_check_compatibility","description":"Evaluate a .safetensors LoRA adapter's serve-readiness across vLLM, SGLang, HuggingFace PEFT, llama.cpp/Ollama, and ComfyUI/Diffusers with exact conversion commands.","inputSchema":{"type":"object","properties":{"source":{"type":"string"}},"required":["source"],"additionalProperties":false}},
{"name":"loraspectra_svd_compress_preview","description":"Run a read-only Eckart-Young SVD compression dry-run comparing nominal vs 90%, 95%, 99% energy, and Marchenko-Pastur optimal ranks with exact Frobenius/spectral error bounds and byte savings.","inputSchema":{"type":"object","properties":{"source":{"type":"string"}},"required":["source"],"additionalProperties":false}},
{"name":"loraspectra_batch_sarif","description":"Recursively audit a directory or list of .safetensors LoRA checkpoints and emit an OASIS SARIF v2.1.0 Code Scanning report flagging low-health adapters below min_score.","inputSchema":{"type":"object","properties":{"paths":{"type":"array","items":{"type":"string"}},"max_depth":{"type":"integer"},"min_score":{"type":"number"}},"required":["paths"],"additionalProperties":false}},
{"name":"loraspectra_sensitivity_sarif","description":"Audit layer-wise catastrophic forgetting sensitivity and export OASIS SARIF v2.1.0 findings (LS051 brittle rank-1 dominance, LS052 energy spikes) for GitHub Code Scanning.","inputSchema":{"type":"object","properties":{"source":{"type":"string"},"dominance_warn_pct":{"type":"number"},"spike_multiplier":{"type":"number"}},"required":["source"],"additionalProperties":false}},
{"name":"loraspectra_doctor_sarif","description":"Run Executive Adapter Doctor (A-F) and emit OASIS SARIF v2.1.0 Code Scanning JSON for GitHub CI gates.","inputSchema":{"type":"object","properties":{"source":{"type":"string"}},"required":["source"],"additionalProperties":false}},
{"name":"loraspectra_compat_html","description":"Evaluate 5-runtime LoRA serving compatibility (vLLM, SGLang, PEFT, llama.cpp, ComfyUI) and render a self-contained dark-mode HTML Scorecard.","inputSchema":{"type":"object","properties":{"source":{"type":"string"}},"required":["source"],"additionalProperties":false}},
{"name":"loraspectra_sensitivity_html","description":"Audit layer-wise catastrophic forgetting sensitivity (LS051-LS052) and render a self-contained dark-mode HTML Report.","inputSchema":{"type":"object","properties":{"source":{"type":"string"},"dominance_warn_pct":{"type":"number"},"spike_multiplier":{"type":"number"}},"required":["source"],"additionalProperties":false}},
{"name":"loraspectra_loraplus_html","description":"Audit LoRA+ B/A factor imbalance (LS047-LS048) and render a self-contained dark-mode HTML Balance & Lambda* Report.","inputSchema":{"type":"object","properties":{"source":{"type":"string"},"imbalance_warn":{"type":"number"},"cond_warn":{"type":"number"}},"required":["source"],"additionalProperties":false}},
{"name":"loraspectra_rank_schedule_html","description":"Compute AdaLoRA water-filling heterogeneous rank allocation (LS049-LS050) and render a self-contained dark-mode HTML Report.","inputSchema":{"type":"object","properties":{"source":{"type":"string"},"target_avg_rank":{"type":"number"},"min_rank":{"type":"integer"},"max_rank":{"type":"integer"}},"required":["source"],"additionalProperties":false}},
{"name":"loraspectra_quant_plan_html","description":"Generate a quantization-aware mixed-precision LoRA rank compression plan (nf4/fp4/int8/fp8/bf16/fp16) and render a self-contained dark-mode HTML Report.","inputSchema":{"type":"object","properties":{"source":{"type":"string"},"target_quant":{"type":"string"},"target_energy":{"type":"number"}},"required":["source"],"additionalProperties":false}},
{"name":"loraspectra_diff_html","description":"Compute Frobenius drift, Grassmannian subspace overlap, and DoRA magnitude drift between two .safetensors LoRA checkpoints and render a self-contained dark-mode HTML Report.","inputSchema":{"type":"object","properties":{"source_a":{"type":"string"},"source_b":{"type":"string"}},"required":["source_a","source_b"],"additionalProperties":false}},
{"name":"loraspectra_merge_check_html","description":"Run multi-adapter pre-merge subspace collision & norm-inflation analysis across >= 2 .safetensors LoRAs and render a self-contained dark-mode HTML Report.","inputSchema":{"type":"object","properties":{"sources":{"type":"array","items":{"type":"string"}}},"required":["sources"],"additionalProperties":false}},
{"name":"loraspectra_align_html","description":"Compute cross-layer Grassmannian subspace alignment (LS029/LS030) for a .safetensors LoRA adapter and render a self-contained dark-mode HTML Report.","inputSchema":{"type":"object","properties":{"source":{"type":"string"}},"required":["source"],"additionalProperties":false}},
{"name":"loraspectra_fp8_validate_html","description":"Validate FP8 E4M3/E5M2 scale factor tensors (LS031-LS034) in a .safetensors checkpoint and render a self-contained dark-mode HTML Report.","inputSchema":{"type":"object","properties":{"source":{"type":"string"},"format":{"type":"string"}},"required":["source"],"additionalProperties":false}},
{"name":"loraspectra_vera_html","description":"Audit VeRA adapter scalar gating vectors (lambda_b/lambda_d, LS025-LS026) and render a self-contained dark-mode HTML Spectral & Gating Report.","inputSchema":{"type":"object","properties":{"source":{"type":"string"}},"required":["source"],"additionalProperties":false}},
{"name":"loraspectra_batch_badge","description":"Recursively audit a directory or list of .safetensors LoRA checkpoints and emit a shields.io Endpoint Badge JSON summarizing mean health score and audited adapter count.","inputSchema":{"type":"object","properties":{"paths":{"type":"array","items":{"type":"string"}},"max_depth":{"type":"integer"}},"required":["paths"],"additionalProperties":false}},
{"name":"loraspectra_subspace_drift_html","description":"Compute principal-angle subspace drift (LS027-LS028) between two .safetensors LoRA checkpoints and render a self-contained dark-mode HTML Report.","inputSchema":{"type":"object","properties":{"source_a":{"type":"string"},"source_b":{"type":"string"}},"required":["source_a","source_b"],"additionalProperties":false}},
{"name":"loraspectra_compat_badge","description":"Evaluate 5-runtime LoRA serving compatibility (vLLM, SGLang, PEFT, llama.cpp, ComfyUI) and emit a shields.io Endpoint Badge JSON readiness payload.","inputSchema":{"type":"object","properties":{"source":{"type":"string"}},"required":["source"],"additionalProperties":false}},
{"name":"loraspectra_loraplus_badge","description":"Audit LoRA+ B/A factor imbalance (LS047-LS048) and emit a shields.io Endpoint Badge JSON summarizing mean imbalance index and recommended lambda*.","inputSchema":{"type":"object","properties":{"source":{"type":"string"},"imbalance_warn":{"type":"number"},"cond_warn":{"type":"number"}},"required":["source"],"additionalProperties":false}},
{"name":"loraspectra_sensitivity_badge","description":"Audit layer-wise catastrophic forgetting sensitivity (LS051-LS052) and emit a shields.io Endpoint Badge JSON summarizing forgetting risk level.","inputSchema":{"type":"object","properties":{"source":{"type":"string"},"dominance_warn_pct":{"type":"number"},"spike_multiplier":{"type":"number"}},"required":["source"],"additionalProperties":false}},
{"name":"loraspectra_quant_plan_badge","description":"Generate a mixed-precision LoRA quantization plan (nf4/fp4/int8/fp8/bf16/fp16) and emit a shields.io Endpoint Badge JSON summarizing projected savings and protected layers.","inputSchema":{"type":"object","properties":{"source":{"type":"string"},"target_quant":{"type":"string"},"target_energy":{"type":"number"}},"required":["source"],"additionalProperties":false}},
{"name":"loraspectra_rank_schedule_badge","description":"Compute AdaLoRA water-filling heterogeneous rank allocation (LS049-LS050) and emit a shields.io Endpoint Badge JSON summarizing rank reallocation and bottlenecks.","inputSchema":{"type":"object","properties":{"source":{"type":"string"},"target_avg_rank":{"type":"number"},"min_rank":{"type":"integer"},"max_rank":{"type":"integer"}},"required":["source"],"additionalProperties":false}},
{"name":"loraspectra_list_rules","description":"List all 52 OASIS SARIF v2.1.0 LoRASpectra diagnostic rules (LS001-LS052).","inputSchema":{"type":"object","properties":{},"additionalProperties":false}}
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
        "loraspectra_sensitivity" => {
            let src = args.get("source").and_then(|v| v.as_str()).ok_or_else(|| "Missing 'source' string".to_string());
            src.and_then(|s| {
                let path = resolve_source_path(s)?;
                let tensors = read_safetensors(&path)?;
                let layers = extract_lora_layers(&tensors)?;
                let dom = args.get("dominance_warn_pct").and_then(|v| v.as_f64()).unwrap_or(85.0);
                let spike = args.get("spike_multiplier").and_then(|v| v.as_f64()).unwrap_or(2.5);
                let rep = analyze_sensitivity(&layers, s, dom, spike);
                Ok(render_sensitivity_json(&rep))
            })
        }
        "loraspectra_doctor" => {
            let src = args.get("source").and_then(|v| v.as_str()).ok_or_else(|| "Missing 'source' string".to_string());
            src.and_then(|s| {
                let path = resolve_source_path(s)?;
                let tensors = read_safetensors(&path)?;
                let layers = extract_lora_layers(&tensors)?;
                let rep = run_doctor(s, &layers, &tensors, None);
                Ok(render_doctor_json(&rep))
            })
        }
        "loraspectra_diff" => {
            let src_a = args.get("source_a").and_then(|v| v.as_str()).ok_or_else(|| "Missing 'source_a' string".to_string());
            let src_b = args.get("source_b").and_then(|v| v.as_str()).ok_or_else(|| "Missing 'source_b' string".to_string());
            src_a.and_then(|a| {
                src_b.and_then(|b| {
                    let pa = resolve_source_path(a)?;
                    let pb = resolve_source_path(b)?;
                    let ta = read_safetensors(&pa)?;
                    let tb = read_safetensors(&pb)?;
                    let la = extract_lora_layers(&ta)?;
                    let lb = extract_lora_layers(&tb)?;
                    let aa = extract_lora_alphas(&ta);
                    let ab = extract_lora_alphas(&tb);
                    let da = extract_dora_scales(&ta);
                    let db = extract_dora_scales(&tb);
                    let summary = diff_adapters_with_dora(&la, &aa, &da, &lb, &ab, &db);
                    Ok(render_diff_json(&summary))
                })
            })
        }
        "loraspectra_merge_check" => {
            let raw_sources = args
                .get("sources")
                .and_then(|v| v.as_arr())
                .ok_or_else(|| "Missing 'sources' array (>= 2 adapter paths/URIs)".to_string());
            raw_sources.and_then(|arr| {
                let mut inputs = Vec::new();
                for item in arr {
                    if let Some(s) = item.as_str() {
                        let p = resolve_source_path(s)?;
                        let t = read_safetensors(&p)?;
                        let layers = extract_lora_layers(&t)?;
                        let alphas = extract_lora_alphas(&t);
                        inputs.push(AdapterMergeInput {
                            name: s.to_string(),
                            layers,
                            alphas,
                            peft_config: None,
                            weight: 1.0,
                        });
                    }
                }
                let summary = analyze_multi_adapter_merge(&inputs)?;
                Ok(render_merge_check_json(&summary))
            })
        }
        "loraspectra_model_card" => {
            let src = args.get("source").and_then(|v| v.as_str()).ok_or_else(|| "Missing 'source' string".to_string());
            src.and_then(|s| {
                let path = resolve_source_path(s)?;
                let tensors = read_safetensors(&path)?;
                let layers = extract_lora_layers(&tensors)?;
                let alphas = extract_lora_alphas(&tensors);
                let dora = extract_dora_scales(&tensors);
                let arch = inspect_adapter_architecture(&layers, &alphas);
                let summary = audit_adapter(&layers);
                let integrity = audit_adapter_integrity(&layers, &tensors, &alphas, &dora, None);
                let md = generate_model_card(s, &arch, &summary, &integrity);
                Ok(format!(
                    r#"{{"source":"{}","markdown":"{}"}}"#,
                    json_escape(s),
                    json_escape(&md)
                ))
            })
        }
        "loraspectra_trajectory" => {
            let raw_sources = args
                .get("sources")
                .and_then(|v| v.as_arr())
                .ok_or_else(|| "Missing 'sources' array (>= 2 ordered checkpoints)".to_string());
            raw_sources.and_then(|arr| {
                let srcs: Vec<String> = arr
                    .iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect();
                let rep = analyze_trajectory(&srcs)?;
                Ok(render_trajectory_json(&rep))
            })
        }
        "loraspectra_quant_plan" => {
            let src = args.get("source").and_then(|v| v.as_str()).ok_or_else(|| "Missing 'source' string".to_string());
            src.and_then(|s| {
                let path = resolve_source_path(s)?;
                let tensors = read_safetensors(&path)?;
                let layers = extract_lora_layers(&tensors)?;
                let q = args.get("target_quant").and_then(|v| v.as_str()).unwrap_or("nf4");
                let energy = args.get("target_energy").and_then(|v| v.as_f64()).unwrap_or(0.99);
                let rep = build_quant_plan(s, &layers, q, energy);
                Ok(render_quant_plan_json(&rep))
            })
        }
        "loraspectra_check_compatibility" => {
            let src = args.get("source").and_then(|v| v.as_str()).ok_or_else(|| "Missing 'source' string".to_string());
            src.and_then(|s| {
                let path = resolve_source_path(s)?;
                let tensors = read_safetensors(&path)?;
                let rep = check_serving_compatibility(s, &tensors)?;
                Ok(render_compat_json(&rep))
            })
        }
        "loraspectra_svd_compress_preview" => {
            let src = args.get("source").and_then(|v| v.as_str()).ok_or_else(|| "Missing 'source' string".to_string());
            src.and_then(|s| {
                let path = resolve_source_path(s)?;
                let tensors = read_safetensors(&path)?;
                let layers = extract_lora_layers(&tensors)?;
                let summary = audit_adapter(&layers);
                let saved_bytes = summary.total_nominal_bytes.saturating_sub(summary.total_optimal_99_bytes);
                let saved_pct = if summary.total_nominal_bytes > 0 {
                    (saved_bytes as f64 / summary.total_nominal_bytes as f64) * 100.0
                } else {
                    0.0
                };
                Ok(format!(
                    r#"{{"source":"{}","layer_count":{},"total_nominal_bytes":{},"total_optimal_99_bytes":{},"saved_bytes_at_99":{},"saved_pct_at_99":{:.2},"mean_nominal_rank":{:.2},"mean_effective_rank":{:.2},"mean_rank_at_99_energy":{:.2},"mean_rank_mp":{:.2},"max_mp_relative_spectral_error":{:.6},"mean_99_relative_nuclear_error":{:.6},"dry_run":true}}"#,
                    json_escape(s),
                    summary.layers.len(),
                    summary.total_nominal_bytes,
                    summary.total_optimal_99_bytes,
                    saved_bytes,
                    saved_pct,
                    summary.mean_nominal_rank,
                    summary.mean_effective_rank,
                    summary.mean_rank_at_99_energy,
                    summary.mean_rank_mp,
                    summary.max_mp_relative_spectral_error,
                    summary.mean_99_relative_nuclear_error
                ))
            })
        }
        "loraspectra_batch_sarif" => {
            let raw_paths = args.get("paths").and_then(|v| v.as_arr()).ok_or_else(|| "Missing 'paths' array".to_string());
            raw_paths.and_then(|arr| {
                let inputs: Vec<PathBuf> = arr.iter().filter_map(|v| v.as_str().map(PathBuf::from)).collect();
                let depth = args.get("max_depth").and_then(|v| v.as_f64()).unwrap_or(6.0).max(1.0) as usize;
                let min_score = args.get("min_score").and_then(|v| v.as_f64()).unwrap_or(70.0);
                let discovered = discover_checkpoints(&inputs, depth);
                let report = run_batch_audit(&discovered);
                Ok(crate::batch::render_batch_sarif(&report, min_score))
            })
        }
        "loraspectra_sensitivity_sarif" => {
            let src = args.get("source").and_then(|v| v.as_str()).ok_or_else(|| "Missing 'source' string".to_string());
            src.and_then(|s| {
                let path = resolve_source_path(s)?;
                let tensors = read_safetensors(&path)?;
                let layers = extract_lora_layers(&tensors)?;
                let dom = args.get("dominance_warn_pct").and_then(|v| v.as_f64()).unwrap_or(85.0);
                let spike = args.get("spike_multiplier").and_then(|v| v.as_f64()).unwrap_or(2.5);
                let rep = analyze_sensitivity(&layers, s, dom, spike);
                Ok(crate::sarif::build_sensitivity_sarif(&rep, env!("CARGO_PKG_VERSION")))
            })
        }
        "loraspectra_doctor_sarif" => {
            let src = args.get("source").and_then(|v| v.as_str()).ok_or_else(|| "Missing 'source' string".to_string());
            src.and_then(|s| {
                let path = resolve_source_path(s)?;
                let tensors = read_safetensors(&path)?;
                let layers = extract_lora_layers(&tensors)?;
                let rep = run_doctor(s, &layers, &tensors, None);
                Ok(crate::doctor::render_doctor_sarif(&rep))
            })
        }
        "loraspectra_compat_html" => {
            let src = args.get("source").and_then(|v| v.as_str()).ok_or_else(|| "Missing 'source' string".to_string());
            src.and_then(|s| {
                let path = resolve_source_path(s)?;
                let tensors = read_safetensors(&path)?;
                let rep = check_serving_compatibility(s, &tensors)?;
                let html = crate::compat::render_compat_html(&rep);
                Ok(format!(
                    r#"{{"source":"{}","html_bytes":{},"html":"{}"}}"#,
                    json_escape(s),
                    html.len(),
                    json_escape(&html)
                ))
            })
        }
        "loraspectra_compat_badge" => {
            let src = args.get("source").and_then(|v| v.as_str()).ok_or_else(|| "Missing 'source' string".to_string());
            src.and_then(|s| {
                let path = resolve_source_path(s)?;
                let tensors = read_safetensors(&path)?;
                let rep = check_serving_compatibility(s, &tensors)?;
                let badge = crate::compat::render_compat_badge_json(&rep);
                Ok(format!(
                    r#"{{"source":"{}","ready_count":{},"total_runtimes":{},"badge":{}}}"#,
                    json_escape(s),
                    rep.ready_count,
                    rep.runtimes.len(),
                    badge
                ))
            })
        }
        "loraspectra_sensitivity_html" => {
            let src = args.get("source").and_then(|v| v.as_str()).ok_or_else(|| "Missing 'source' string".to_string());
            src.and_then(|s| {
                let path = resolve_source_path(s)?;
                let tensors = read_safetensors(&path)?;
                let layers = extract_lora_layers(&tensors)?;
                let dom = args.get("dominance_warn_pct").and_then(|v| v.as_f64()).unwrap_or(85.0);
                let spike = args.get("spike_multiplier").and_then(|v| v.as_f64()).unwrap_or(2.5);
                let rep = analyze_sensitivity(&layers, s, dom, spike);
                let html = crate::sensitivity::render_sensitivity_html(&rep);
                Ok(format!(
                    r#"{{"source":"{}","forgetting_risk_level":"{}","html_bytes":{},"html":"{}"}}"#,
                    json_escape(s),
                    rep.forgetting_risk_level,
                    html.len(),
                    json_escape(&html)
                ))
            })
        }
        "loraspectra_loraplus_html" => {
            let src = args.get("source").and_then(|v| v.as_str()).ok_or_else(|| "Missing 'source' string".to_string());
            src.and_then(|s| {
                let path = resolve_source_path(s)?;
                let tensors = read_safetensors(&path)?;
                let layers = extract_lora_layers(&tensors)?;
                let imb = args.get("imbalance_warn").and_then(|v| v.as_f64()).unwrap_or(1.5);
                let cond = args.get("cond_warn").and_then(|v| v.as_f64()).unwrap_or(50.0);
                let summary = analyze_loraplus(&layers, s, imb, cond);
                let html = crate::loraplus::render_loraplus_html(&summary);
                Ok(format!(
                    r#"{{"source":"{}","mean_imbalance_index":{:.4},"html_bytes":{},"html":"{}"}}"#,
                    json_escape(s),
                    summary.mean_imbalance_index,
                    html.len(),
                    json_escape(&html)
                ))
            })
        }
        "loraspectra_loraplus_badge" => {
            let src = args.get("source").and_then(|v| v.as_str()).ok_or_else(|| "Missing 'source' string".to_string());
            src.and_then(|s| {
                let path = resolve_source_path(s)?;
                let tensors = read_safetensors(&path)?;
                let layers = extract_lora_layers(&tensors)?;
                let imb = args.get("imbalance_warn").and_then(|v| v.as_f64()).unwrap_or(1.5);
                let cond = args.get("cond_warn").and_then(|v| v.as_f64()).unwrap_or(50.0);
                let summary = analyze_loraplus(&layers, s, imb, cond);
                let badge = crate::loraplus::render_loraplus_badge_json(&summary);
                Ok(format!(
                    r#"{{"source":"{}","mean_imbalance_index":{:.4},"badge":{}}}"#,
                    json_escape(s),
                    summary.mean_imbalance_index,
                    badge
                ))
            })
        }
        "loraspectra_sensitivity_badge" => {
            let src = args.get("source").and_then(|v| v.as_str()).ok_or_else(|| "Missing 'source' string".to_string());
            src.and_then(|s| {
                let path = resolve_source_path(s)?;
                let tensors = read_safetensors(&path)?;
                let layers = extract_lora_layers(&tensors)?;
                let dom = args.get("dominance_warn_pct").and_then(|v| v.as_f64()).unwrap_or(85.0);
                let spike = args.get("spike_multiplier").and_then(|v| v.as_f64()).unwrap_or(2.5);
                let rep = analyze_sensitivity(&layers, s, dom, spike);
                let badge = crate::sensitivity::render_sensitivity_badge_json(&rep);
                Ok(format!(
                    r#"{{"source":"{}","forgetting_risk_level":"{}","badge":{}}}"#,
                    json_escape(s),
                    rep.forgetting_risk_level,
                    badge
                ))
            })
        }
        "loraspectra_quant_plan_badge" => {
            let src = args.get("source").and_then(|v| v.as_str()).ok_or_else(|| "Missing 'source' string".to_string());
            src.and_then(|s| {
                let path = resolve_source_path(s)?;
                let tensors = read_safetensors(&path)?;
                let layers = extract_lora_layers(&tensors)?;
                let quant = args.get("target_quant").and_then(|v| v.as_str()).unwrap_or("nf4");
                let energy = args.get("target_energy").and_then(|v| v.as_f64()).unwrap_or(0.99);
                let rep = crate::quant_plan::build_quant_plan(s, &layers, quant, energy);
                let badge = crate::quant_plan::render_quant_plan_badge_json(&rep);
                Ok(format!(
                    r#"{{"source":"{}","target_quant":"{}","total_savings_pct":{:.2},"badge":{}}}"#,
                    json_escape(s),
                    json_escape(&rep.target_quant),
                    rep.total_savings_pct,
                    badge
                ))
            })
        }
        "loraspectra_rank_schedule_badge" => {
            let src = args.get("source").and_then(|v| v.as_str()).ok_or_else(|| "Missing 'source' string".to_string());
            src.and_then(|s| {
                let path = resolve_source_path(s)?;
                let tensors = read_safetensors(&path)?;
                let layers = extract_lora_layers(&tensors)?;
                let target = args.get("target_avg_rank").and_then(|v| v.as_f64()).unwrap_or(16.0);
                let min_r = args.get("min_rank").and_then(|v| v.as_f64()).unwrap_or(4.0).max(1.0) as usize;
                let max_r = args.get("max_rank").and_then(|v| v.as_f64()).unwrap_or(64.0).max(min_r as f64) as usize;
                let summary = optimize_rank_schedule(&layers, s, target, min_r, max_r);
                let badge = crate::adalora::render_rank_schedule_badge_json(&summary);
                Ok(format!(
                    r#"{{"source":"{}","bottleneck_count":{},"badge":{}}}"#,
                    json_escape(s),
                    summary.bottleneck_count,
                    badge
                ))
            })
        }
        "loraspectra_rank_schedule_html" => {
            let src = args.get("source").and_then(|v| v.as_str()).ok_or_else(|| "Missing 'source' string".to_string());
            src.and_then(|s| {
                let path = resolve_source_path(s)?;
                let tensors = read_safetensors(&path)?;
                let layers = extract_lora_layers(&tensors)?;
                let target = args.get("target_avg_rank").and_then(|v| v.as_f64()).unwrap_or(16.0);
                let min_r = args.get("min_rank").and_then(|v| v.as_f64()).unwrap_or(4.0).max(1.0) as usize;
                let max_r = args.get("max_rank").and_then(|v| v.as_f64()).unwrap_or(64.0).max(min_r as f64) as usize;
                let summary = optimize_rank_schedule(&layers, s, target, min_r, max_r);
                let html = crate::adalora::render_rank_schedule_html(&summary);
                Ok(format!(
                    r#"{{"source":"{}","bottleneck_count":{},"over_allocated_count":{},"html_bytes":{},"html":"{}"}}"#,
                    json_escape(s),
                    summary.bottleneck_count,
                    summary.over_allocated_count,
                    html.len(),
                    json_escape(&html)
                ))
            })
        }
        "loraspectra_quant_plan_html" => {
            let src = args.get("source").and_then(|v| v.as_str()).ok_or_else(|| "Missing 'source' string".to_string());
            src.and_then(|s| {
                let path = resolve_source_path(s)?;
                let tensors = read_safetensors(&path)?;
                let layers = extract_lora_layers(&tensors)?;
                let q = args.get("target_quant").and_then(|v| v.as_str()).unwrap_or("nf4");
                let energy = args.get("target_energy").and_then(|v| v.as_f64()).unwrap_or(0.99);
                let rep = build_quant_plan(s, &layers, q, energy);
                let html = crate::quant_plan::render_quant_plan_html(&rep);
                Ok(format!(
                    r#"{{"source":"{}","target_quant":"{}","html_bytes":{},"html":"{}"}}"#,
                    json_escape(s),
                    json_escape(q),
                    html.len(),
                    json_escape(&html)
                ))
            })
        }
        "loraspectra_diff_html" => {
            let sa = args.get("source_a").and_then(|v| v.as_str()).ok_or_else(|| "Missing 'source_a' string".to_string());
            let sb = args.get("source_b").and_then(|v| v.as_str()).ok_or_else(|| "Missing 'source_b' string".to_string());
            sa.and_then(|a| {
                sb.and_then(|b| {
                    let pa = resolve_source_path(a)?;
                    let pb = resolve_source_path(b)?;
                    let ta = read_safetensors(&pa)?;
                    let tb = read_safetensors(&pb)?;
                    let la = extract_lora_layers(&ta)?;
                    let lb = extract_lora_layers(&tb)?;
                    let aa = extract_lora_alphas(&ta);
                    let ab = extract_lora_alphas(&tb);
                    let da = extract_dora_scales(&ta);
                    let db = extract_dora_scales(&tb);
                    let summary = diff_adapters_with_dora(&la, &aa, &da, &lb, &ab, &db);
                    let html = crate::diff::render_diff_html(&summary);
                    Ok(format!(
                        r#"{{"shared_layers":{},"mean_relative_drift":{:.4},"mean_subspace_similarity":{:.4},"html_bytes":{},"html":"{}"}}"#,
                        summary.shared_layers.len(),
                        summary.mean_relative_drift,
                        summary.mean_subspace_similarity,
                        html.len(),
                        json_escape(&html)
                    ))
                })
            })
        }
        "loraspectra_merge_check_html" => {
            let raw_sources = args
                .get("sources")
                .and_then(|v| v.as_arr())
                .ok_or_else(|| "Missing 'sources' array (>= 2 adapter paths/URIs)".to_string());
            raw_sources.and_then(|arr| {
                let mut inputs = Vec::new();
                for item in arr {
                    if let Some(s) = item.as_str() {
                        let p = resolve_source_path(s)?;
                        let t = read_safetensors(&p)?;
                        let layers = extract_lora_layers(&t)?;
                        let alphas = extract_lora_alphas(&t);
                        inputs.push(AdapterMergeInput {
                            name: s.to_string(),
                            layers,
                            alphas,
                            peft_config: None,
                            weight: 1.0,
                        });
                    }
                }
                let summary = analyze_multi_adapter_merge(&inputs)?;
                let html = crate::merge_check::render_merge_check_html(&summary);
                Ok(format!(
                    r#"{{"adapter_count":{},"risk_level":"{}","recommended_mode":"{}","html_bytes":{},"html":"{}"}}"#,
                    summary.adapter_names.len(),
                    json_escape(summary.advisor.risk_level),
                    json_escape(summary.advisor.recommended_mode),
                    html.len(),
                    json_escape(&html)
                ))
            })
        }
        "loraspectra_align_html" => {
            let src = args.get("source").and_then(|v| v.as_str()).ok_or_else(|| "Missing 'source' string".to_string());
            src.and_then(|s| {
                let path = resolve_source_path(s)?;
                let tensors = read_safetensors(&path)?;
                let layers = extract_lora_layers(&tensors)?;
                let matrix = crate::align::analyze_cross_layer_alignment(&layers);
                let html = crate::align::render_alignment_html(&matrix);
                Ok(format!(
                    r#"{{"source":"{}","pair_count":{},"redundant_pairs":{},"disjoint_pairs":{},"mean_grassmannian_distance":{:.4},"html_bytes":{},"html":"{}"}}"#,
                    json_escape(s),
                    matrix.pairs.len(),
                    matrix.redundant_pairs,
                    matrix.disjoint_pairs,
                    matrix.mean_grassmannian_distance,
                    html.len(),
                    json_escape(&html)
                ))
            })
        }
        "loraspectra_fp8_validate_html" => {
            let src = args.get("source").and_then(|v| v.as_str()).ok_or_else(|| "Missing 'source' string".to_string());
            src.and_then(|s| {
                let path = resolve_source_path(s)?;
                let tensors = read_safetensors(&path)?;
                let fmt_str = args.get("format").and_then(|v| v.as_str()).unwrap_or("e4m3");
                let fp8_fmt = crate::quant_fp8::Fp8Format::from_name(fmt_str)?;
                let rep = crate::quant_fp8::validate_fp8_scale_factors(&tensors, fp8_fmt);
                let html = crate::quant_fp8::render_fp8_scale_html(&rep);
                Ok(format!(
                    r#"{{"source":"{}","format":"{}","total_scale_tensors":{},"critical_count":{},"error_count":{},"html_bytes":{},"html":"{}"}}"#,
                    json_escape(s),
                    rep.format.as_str(),
                    rep.total_scale_tensors,
                    rep.critical_count,
                    rep.error_count,
                    html.len(),
                    json_escape(&html)
                ))
            })
        }
        "loraspectra_vera_html" => {
            let src = args.get("source").and_then(|v| v.as_str()).ok_or_else(|| "Missing 'source' string".to_string());
            src.and_then(|s| {
                let path = resolve_source_path(s)?;
                let tensors = read_safetensors(&path)?;
                let rep = crate::vera::audit_vera_adapter(&tensors);
                let html = crate::vera::render_vera_html(&rep);
                Ok(format!(
                    r#"{{"source":"{}","total_layers":{},"healthy_layers":{},"dead_rank_layers":{},"html_bytes":{},"html":"{}"}}"#,
                    json_escape(s),
                    rep.total_layers,
                    rep.healthy_layers,
                    rep.dead_rank_layers,
                    html.len(),
                    json_escape(&html)
                ))
            })
        }
        "loraspectra_batch_badge" => {
            let raw_paths = args
                .get("paths")
                .and_then(|v| v.as_arr())
                .ok_or_else(|| "Missing 'paths' array of directories or files".to_string());
            raw_paths.and_then(|arr| {
                let paths: Vec<PathBuf> = arr.iter().filter_map(|v| v.as_str().map(PathBuf::from)).collect();
                let max_depth = args.get("max_depth").and_then(|v| v.as_f64()).map(|n| n.max(1.0) as usize).unwrap_or(6);
                let discovered = discover_checkpoints(&paths, max_depth);
                let report = run_batch_audit(&discovered);
                let badge = crate::batch::render_batch_badge_json(&report);
                Ok(format!(
                    r#"{{"audited_count":{},"mean_health_score":{:.2},"badge":{}}}"#,
                    report.audited_count, report.mean_health_score, badge
                ))
            })
        }
        "loraspectra_subspace_drift_html" => {
            let sa = args.get("source_a").and_then(|v| v.as_str()).ok_or_else(|| "Missing 'source_a' string".to_string());
            let sb = args.get("source_b").and_then(|v| v.as_str()).ok_or_else(|| "Missing 'source_b' string".to_string());
            sa.and_then(|a| {
                sb.and_then(|b| {
                    let pa = resolve_source_path(a)?;
                    let pb = resolve_source_path(b)?;
                    let ta = read_safetensors(&pa)?;
                    let tb = read_safetensors(&pb)?;
                    let la = extract_lora_layers(&ta)?;
                    let lb = extract_lora_layers(&tb)?;
                    let aa = extract_lora_alphas(&ta);
                    let ab = extract_lora_alphas(&tb);
                    let summary = crate::diff::analyze_subspace_drift(&la, &aa, &lb, &ab);
                    let html = crate::diff::render_subspace_drift_html(&summary);
                    Ok(format!(
                        r#"{{"total_shared_layers":{},"healthy_layers":{},"catastrophic_drift_layers":{},"mean_angle_deg":{:.4},"html_bytes":{},"html":"{}"}}"#,
                        summary.total_shared_layers,
                        summary.healthy_layers,
                        summary.catastrophic_drift_layers,
                        summary.mean_angle_deg,
                        html.len(),
                        json_escape(&html)
                    ))
                })
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
        assert!(tools.contains("loraspectra_doctor"));
        assert!(tools.contains("loraspectra_diff"));
        assert!(tools.contains("loraspectra_merge_check"));
        assert!(tools.contains("loraspectra_model_card"));
        assert!(tools.contains("loraspectra_trajectory"));
        assert!(tools.contains("loraspectra_quant_plan"));
        assert!(tools.contains("loraspectra_check_compatibility"));
        assert!(tools.contains("loraspectra_svd_compress_preview"));
        assert!(tools.contains("loraspectra_batch_sarif"));
        assert!(tools.contains("loraspectra_sensitivity_sarif"));
        assert!(tools.contains("loraspectra_doctor_sarif"));
        assert!(tools.contains("loraspectra_compat_html"));
        assert!(tools.contains("loraspectra_sensitivity_html"));
        assert!(tools.contains("loraspectra_loraplus_html"));
        assert!(tools.contains("loraspectra_rank_schedule_html"));
        assert!(tools.contains("loraspectra_quant_plan_html"));
        assert!(tools.contains("loraspectra_diff_html"));
        assert!(tools.contains("loraspectra_merge_check_html"));
        assert!(tools.contains("loraspectra_align_html"));
        assert!(tools.contains("loraspectra_fp8_validate_html"));
        assert!(tools.contains("loraspectra_vera_html"));
        assert!(tools.contains("loraspectra_batch_badge"));
        assert!(tools.contains("loraspectra_subspace_drift_html"));
        assert!(tools.contains("loraspectra_compat_badge"));
        assert!(tools.contains("loraspectra_loraplus_badge"));
        assert!(tools.contains("loraspectra_sensitivity_badge"));
        assert!(tools.contains("loraspectra_quant_plan_badge"));
        assert!(tools.contains("loraspectra_rank_schedule_badge"));

        let rules = handle_mcp_message(
            r#"{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"loraspectra_list_rules","arguments":{}}}"#,
        )
        .unwrap();
        assert!(rules.contains(r#"rule_count\":52"#));
        assert!(rules.contains("LS052"));

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

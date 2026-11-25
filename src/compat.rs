//! Cross-Serving Runtime Compatibility Matrix Checker (`loraspectra_check_compatibility`).
//!
//! Evaluates whether a `.safetensors` LoRA adapter can be loaded cleanly across 6 major serving
//! runtimes (`vLLM`, `SGLang`, `HuggingFace PEFT`, `llama.cpp`, `Ollama`, `ComfyUI`) based on
//! key naming format, rank heterogeneity, DoRA magnitude vectors, `.alpha` tensors, and dtype.

use crate::safetensors::{
    extract_dora_scales, extract_lora_alphas, extract_lora_layers, inspect_adapter_architecture,
    DType, Tensor2D,
};
use std::collections::BTreeMap;

#[derive(Debug, Clone)]
pub struct RuntimeCompatEntry {
    pub runtime: String,
    pub status: String, // "ready" | "needs-conversion" | "unsupported"
    pub notes: Vec<String>,
    pub remediation: Option<String>,
}

#[derive(Debug, Clone)]
pub struct CompatibilityReport {
    pub source: String,
    pub naming_format: String,
    pub inferred_architecture: String,
    pub primary_dtype: String,
    pub heterogeneous_ranks: bool,
    pub has_dora: bool,
    pub has_alpha_tensors: bool,
    pub ready_count: usize,
    pub runtimes: Vec<RuntimeCompatEntry>,
}

fn json_esc(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c => out.push(c),
        }
    }
    out
}

pub fn check_serving_compatibility(
    source: &str,
    tensors: &BTreeMap<String, Tensor2D>,
) -> Result<CompatibilityReport, String> {
    let layers = extract_lora_layers(tensors)?;
    let alphas = extract_lora_alphas(tensors);
    let dora = extract_dora_scales(tensors);
    let arch = inspect_adapter_architecture(&layers, &alphas);

    let hetero = arch.min_rank != arch.max_rank;
    let has_dora = !dora.is_empty();
    let has_alpha = !alphas.is_empty();
    let fmt = arch.naming_format;
    let is_peft = fmt.eq_ignore_ascii_case("peft") || fmt.to_ascii_lowercase().contains("peft");
    let is_kohya = fmt.to_ascii_lowercase().contains("kohya") || fmt.to_ascii_lowercase().contains("diffusers");
    let is_quant_packed = matches!(arch.primary_dtype, DType::U8 | DType::I8);

    let mut runtimes = Vec::new();

    // 1. HuggingFace PEFT
    {
        let mut notes = Vec::new();
        let status = if is_peft {
            notes.push("Native PEFT key convention detected.".into());
            if hetero {
                notes.push("Requires rank_pattern & alpha_pattern in adapter_config.json.".into());
            }
            "ready"
        } else {
            notes.push(format!("Format '{}' should be converted to PEFT keys.", fmt));
            "needs-conversion"
        };
        let rem = if status != "ready" {
            Some(format!("loraspectra merge {} -o peft_adapter.safetensors --format peft", source))
        } else {
            None
        };
        runtimes.push(RuntimeCompatEntry {
            runtime: "HuggingFace PEFT".into(),
            status: status.into(),
            notes,
            remediation: rem,
        });
    }

    // 2. vLLM
    {
        let mut notes = Vec::new();
        let mut needs_conv = !is_peft || has_alpha || is_quant_packed;
        if !is_peft {
            notes.push("vLLM expects PEFT-formatted lora_A/lora_B keys.".into());
        }
        if has_alpha {
            notes.push("Inline .alpha tensors should be baked into (B, A) or declared in adapter_config.json.".into());
        }
        if hetero {
            notes.push("Heterogeneous ranks require --max-lora-rank >= max_rank and rank_pattern support (or baking).".into());
            needs_conv = true;
        }
        if has_dora {
            notes.push("DoRA magnitude vectors require vLLM DoRA support or base-weight merging.".into());
        }
        if is_quant_packed {
            notes.push("Packed 4-bit/8-bit adapter tensors should be dequantized to BF16/FP16 for vLLM LoRA serving.".into());
        }
        if notes.is_empty() {
            notes.push("Directly compatible with vLLM --enable-lora.".into());
        }
        let status = if needs_conv { "needs-conversion" } else { "ready" };
        let rem = if needs_conv {
            Some(format!("loraspectra compress {} -o vllm_ready.safetensors --format peft --bake-alpha --dtype bf16", source))
        } else {
            None
        };
        runtimes.push(RuntimeCompatEntry {
            runtime: "vLLM".into(),
            status: status.into(),
            notes,
            remediation: rem,
        });
    }

    // 3. SGLang
    {
        let mut notes = Vec::new();
        let needs_conv = !is_peft || has_dora || is_quant_packed;
        if !is_peft {
            notes.push("Convert keys to PEFT format for SGLang multi-LoRA serving.".into());
        }
        if has_dora {
            notes.push("DoRA adapters are safest merged into dense base weights for SGLang.".into());
        }
        if notes.is_empty() {
            notes.push("Ready for SGLang --lora-paths serving.".into());
        }
        runtimes.push(RuntimeCompatEntry {
            runtime: "SGLang".into(),
            status: if needs_conv { "needs-conversion" } else { "ready" }.into(),
            notes,
            remediation: if needs_conv {
                Some(format!("loraspectra merge {} -o sglang_ready.safetensors --format peft --bake-alpha", source))
            } else {
                None
            },
        });
    }

    // 4. llama.cpp / Ollama (GGUF LoRA)
    {
        let mut notes = Vec::new();
        let status = if has_dora {
            notes.push("DoRA magnitude vectors are not universally supported by GGUF LoRA hot-swapping; merge into base first.".into());
            "unsupported"
        } else if !is_peft || has_alpha {
            notes.push("Convert to PEFT + bake alpha before running convert_lora_to_gguf.py.".into());
            "needs-conversion"
        } else {
            notes.push("Compatible with llama.cpp convert_lora_to_gguf.py & Ollama ADAPTER directive.".into());
            "ready"
        };
        runtimes.push(RuntimeCompatEntry {
            runtime: "llama.cpp / Ollama".into(),
            status: status.into(),
            notes,
            remediation: if status != "ready" {
                Some(format!("loraspectra compress {} -o gguf_prep.safetensors --format peft --bake-alpha --dtype f16", source))
            } else {
                None
            },
        });
    }

    // 5. ComfyUI / Diffusers
    {
        let mut notes = Vec::new();
        let status = if is_kohya {
            notes.push("Kohya/Diffusers key format detected; directly loadable in ComfyUI LoadLoRA.".into());
            "ready"
        } else {
            notes.push("PEFT LLM naming detected; convert with --format kohya if targeting diffusion/FLUX workflows.".into());
            "needs-conversion"
        };
        runtimes.push(RuntimeCompatEntry {
            runtime: "ComfyUI / Diffusers".into(),
            status: status.into(),
            notes,
            remediation: if status != "ready" {
                Some(format!("loraspectra merge {} -o comfy_lora.safetensors --format kohya", source))
            } else {
                None
            },
        });
    }

    let ready_count = runtimes.iter().filter(|r| r.status == "ready").count();

    Ok(CompatibilityReport {
        source: source.to_string(),
        naming_format: fmt.to_string(),
        inferred_architecture: arch.inferred_architecture.to_string(),
        primary_dtype: arch.primary_dtype.as_str().to_string(),
        heterogeneous_ranks: hetero,
        has_dora,
        has_alpha_tensors: has_alpha,
        ready_count,
        runtimes,
    })
}

pub fn render_compat_table(rep: &CompatibilityReport) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "=== LoRASpectra Serving Runtime Compatibility ({}) ===\n",
        rep.source
    ));
    out.push_str(&format!(
        "Format: {} | Arch: {} | DType: {} | Hetero-Rank: {} | DoRA: {} | Ready: {}/{}\n\n",
        rep.naming_format,
        rep.inferred_architecture,
        rep.primary_dtype,
        rep.heterogeneous_ranks,
        rep.has_dora,
        rep.ready_count,
        rep.runtimes.len()
    ));
    for r in &rep.runtimes {
        out.push_str(&format!("- {:<20} [{}]\n", r.runtime, r.status));
        for n in &r.notes {
            out.push_str(&format!("    note: {}\n", n));
        }
        if let Some(ref rem) = r.remediation {
            out.push_str(&format!("    fix:  {}\n", rem));
        }
    }
    out
}

pub fn render_compat_json(rep: &CompatibilityReport) -> String {
    let entries: Vec<String> = rep
        .runtimes
        .iter()
        .map(|r| {
            let notes: Vec<String> = r.notes.iter().map(|n| format!("\"{}\"", json_esc(n))).collect();
            let rem = match &r.remediation {
                Some(c) => format!("\"{}\"", json_esc(c)),
                None => "null".to_string(),
            };
            format!(
                r#"{{"runtime":"{}","status":"{}","notes":[{}],"remediation":{}}}"#,
                json_esc(&r.runtime),
                json_esc(&r.status),
                notes.join(","),
                rem
            )
        })
        .collect();
    format!(
        r#"{{"source":"{}","naming_format":"{}","inferred_architecture":"{}","primary_dtype":"{}","heterogeneous_ranks":{},"has_dora":{},"has_alpha_tensors":{},"ready_count":{},"total_runtimes":{},"runtimes":[{}]}}"#,
        json_esc(&rep.source),
        json_esc(&rep.naming_format),
        json_esc(&rep.inferred_architecture),
        json_esc(&rep.primary_dtype),
        if rep.heterogeneous_ranks { "true" } else { "false" },
        if rep.has_dora { "true" } else { "false" },
        if rep.has_alpha_tensors { "true" } else { "false" },
        rep.ready_count,
        rep.runtimes.len(),
        entries.join(",")
    )
}

/// Checks whether a CompatibilityReport marks the requested serving runtime as `"ready"`.
/// Accepts aliases such as `vllm`, `sglang`, `peft`, `llama.cpp`, `ollama`, `comfyui`, `diffusers`.
pub fn is_runtime_ready(rep: &CompatibilityReport, target: &str) -> Result<bool, String> {
    let needle = target.trim().to_ascii_lowercase();
    let matched = rep.runtimes.iter().find(|r| {
        let rt = r.runtime.to_ascii_lowercase();
        rt == needle
            || (needle == "vllm" && rt.contains("vllm"))
            || (needle == "sglang" && rt.contains("sglang"))
            || ((needle == "peft" || needle == "hf") && rt.contains("peft"))
            || ((needle == "llama.cpp" || needle == "llamacpp" || needle == "ollama" || needle == "gguf")
                && rt.contains("llama.cpp"))
            || ((needle == "comfyui" || needle == "comfy" || needle == "diffusers") && rt.contains("comfyui"))
    });
    match matched {
        Some(entry) => Ok(entry.status == "ready"),
        None => Err(format!(
            "Unknown runtime '{}'. Expected one of: vllm, sglang, peft, llama.cpp, comfyui.",
            target
        )),
    }
}

fn html_esc(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

/// Renders a shields.io Endpoint Badge JSON summarizing serving runtime readiness.
pub fn render_compat_badge_json(rep: &CompatibilityReport) -> String {
    let total = rep.runtimes.len();
    let color = if rep.ready_count >= 4 {
        "brightgreen"
    } else if rep.ready_count >= 2 {
        "yellow"
    } else {
        "red"
    };
    format!(
        r#"{{"schemaVersion":1,"label":"loraspectra compat","message":"{}/{} ready ({})","color":"{}"}}"#,
        rep.ready_count,
        total,
        json_esc(&rep.naming_format),
        color
    )
}

/// Renders a self-contained dark-mode 5-runtime serving compatibility HTML scorecard.
pub fn render_compat_html(rep: &CompatibilityReport) -> String {
    let mut rows = String::new();
    for r in &rep.runtimes {
        let col = match r.status.as_str() {
            "ready" => "#3fb950",
            "needs-conversion" => "#d29922",
            _ => "#f85149",
        };
        let notes = r.notes.iter().map(|n| html_esc(n)).collect::<Vec<_>>().join("<br>");
        let fix = r.remediation.as_deref().map(|c| format!("<code>{}</code>", html_esc(c))).unwrap_or_else(|| "—".into());
        rows.push_str(&format!(
            "<tr><td><b>{}</b></td><td style=\"color:{}\"><b>{}</b></td><td>{}</td><td>{}</td></tr>\n",
            html_esc(&r.runtime), col, html_esc(&r.status), notes, fix
        ));
    }
    format!(
        r##"<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>LoRASpectra Serving Compatibility Scorecard</title>
<style>
body{{margin:0;padding:28px;background:#0b0f17;color:#e6edf3;font-family:system-ui,sans-serif}}
.wrap{{max-width:1000px;margin:0 auto}}.card{{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:18px;margin-bottom:16px}}
table{{width:100%;border-collapse:collapse;font-size:13px}}th,td{{padding:9px 10px;border-bottom:1px solid #21262d;text-align:left;vertical-align:top}}
th{{color:#8b949e}}code{{font-family:ui-monospace,monospace;font-size:12px;color:#58a6ff}}
</style></head><body><div class="wrap">
<div class="card"><h1>LoRASpectra 5-Runtime Serving Compatibility Scorecard</h1>
<p>Adapter: <b>{}</b> &middot; Format: <b>{}</b> &middot; Arch: <b>{}</b> &middot; DType: <b>{}</b> &middot; Ready: <b>{}/{}</b></p></div>
<div class="card"><table><thead><tr><th>Serving Runtime</th><th>Status</th><th>Notes</th><th>Remediation</th></tr></thead>
<tbody>{}</tbody></table></div></div></body></html>"##,
        html_esc(&rep.source),
        html_esc(&rep.naming_format),
        html_esc(&rep.inferred_architecture),
        html_esc(&rep.primary_dtype),
        rep.ready_count,
        rep.runtimes.len(),
        rows
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::safetensors::{layers_to_tensor_map, LoraLayer};
    use crate::Rng64;

    #[test]
    fn peft_uniform_bf16_adapter_is_ready_for_vllm_and_peft() {
        let mut rng = Rng64::new(11);
        let layer = LoraLayer {
            base_name: "model.layers.0.self_attn.q_proj".into(),
            a_key: "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight".into(),
            b_key: "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight".into(),
            a: Tensor2D { rows: 8, cols: 16, dtype: DType::BF16, data: (0..128).map(|_| rng.next_gaussian() * 0.1).collect() },
            b: Tensor2D { rows: 16, cols: 8, dtype: DType::BF16, data: (0..128).map(|_| rng.next_gaussian() * 0.1).collect() },
        };
        let map = layers_to_tensor_map(&[layer]);
        let rep = check_serving_compatibility("adapter.safetensors", &map).unwrap();
        assert!(rep.ready_count >= 3);
        assert!(rep.runtimes.iter().any(|r| r.runtime == "vLLM" && r.status == "ready"));
        assert!(render_compat_json(&rep).contains("\"vLLM\""));
        assert_eq!(is_runtime_ready(&rep, "vllm").unwrap(), true);
        assert_eq!(is_runtime_ready(&rep, "comfyui").unwrap(), false);
        assert!(is_runtime_ready(&rep, "unknown_engine").is_err());
        let html = render_compat_html(&rep);
        assert!(html.contains("<!doctype html>"));
        assert!(html.contains("vLLM"));
        let badge = render_compat_badge_json(&rep);
        assert!(badge.contains("\"schemaVersion\":1"));
        assert!(badge.contains("ready"));
    }
}

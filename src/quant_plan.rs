//! Quantization-Aware LoRA Compression Planner (`loraspectra_quant_plan`).
//!
//! Combines exact QR-Core SVD energy truncation (`audit_adapter`) with precision-bitwidth
//! modeling (`fp32`, `bf16`, `fp16`, `fp8`, `int8`, `nf4`, `int4`) to plan optimal
//! storage + VRAM savings and flag layers whose sharp singular spectrum makes 4-bit
//! quantization risky (recommending mixed-precision or higher kept rank).

use crate::audit::audit_adapter;
use crate::safetensors::LoraLayer;

#[derive(Debug, Clone)]
pub struct LayerQuantPlan {
    pub base_name: String,
    pub nominal_rank: usize,
    pub recommended_rank: usize,
    pub nominal_bytes: usize,
    pub projected_bytes: usize,
    pub savings_pct: f64,
    pub recommended_dtype: String,
    pub quant_risk: String,
}

#[derive(Debug, Clone)]
pub struct QuantPlanReport {
    pub source: String,
    pub target_quant: String,
    pub target_energy: f64,
    pub layer_count: usize,
    pub total_nominal_bytes: usize,
    pub total_projected_bytes: usize,
    pub total_savings_pct: f64,
    pub high_risk_layers: usize,
    pub recommended_cli_command: String,
    pub layers: Vec<LayerQuantPlan>,
}

fn bits_per_param(quant: &str) -> f64 {
    match quant.to_ascii_lowercase().as_str() {
        "fp32" | "f32" => 32.0,
        "fp16" | "f16" | "bf16" => 16.0,
        "fp8" | "int8" | "i8" => 8.5, // 8-bit + block scale overhead
        "nf4" | "fp4" | "int4" => 4.5, // 4-bit + block absmax overhead
        _ => 16.0,
    }
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

/// Build a quantization-aware rank compression plan for all LoRA layers.
pub fn build_quant_plan(
    source: &str,
    layers: &[LoraLayer],
    target_quant: &str,
    target_energy: f64,
) -> QuantPlanReport {
    let audit = audit_adapter(layers);
    let q_norm = target_quant.trim().to_ascii_lowercase();
    let q_bits = bits_per_param(&q_norm);
    let energy = target_energy.clamp(0.85, 0.999);
    let is_low_bit = matches!(q_norm.as_str(), "nf4" | "fp4" | "int4" | "int8" | "fp8");

    let mut plan_layers = Vec::with_capacity(audit.layers.len());
    let mut total_nom = 0usize;
    let mut total_proj = 0usize;
    let mut high_risk = 0usize;

    for rep in &audit.layers {
        let base_k = if energy <= 0.91 {
            rep.rank_at_90_energy
        } else if energy <= 0.96 {
            rep.rank_at_95_energy
        } else {
            rep.rank_at_99_energy
        }
        .max(1)
        .min(rep.nominal_rank.max(1));

        // High spectral concentration (stable_rank < 1.5) + large spectral norm is sensitive to 4-bit clipping
        let cond_proxy = if rep.effective_rank > 1e-6 {
            (rep.nominal_rank as f64) / rep.effective_rank
        } else {
            1.0
        };
        let (risk, rec_dtype, rec_rank) = if is_low_bit && rep.stable_rank < 1.45 && rep.spectral_norm > 1.0 {
            high_risk += 1;
            ("high".to_string(), "bf16".to_string(), base_k)
        } else if is_low_bit && cond_proxy > 4.0 {
            ("medium".to_string(), q_norm.clone(), (base_k + 1).min(rep.nominal_rank.max(1)))
        } else {
            ("low".to_string(), q_norm.clone(), base_k)
        };

        let effective_bits = bits_per_param(&rec_dtype);
        let params = rec_rank * (rep.d_in + rep.d_out);
        let proj_bytes = (((params as f64) * effective_bits) / 8.0).ceil() as usize;
        let nom_bytes = rep.raw_bytes.max(1);
        let savings = ((1.0 - (proj_bytes as f64 / nom_bytes as f64)) * 100.0).clamp(-100.0, 99.9);

        total_nom += rep.raw_bytes;
        total_proj += proj_bytes;

        plan_layers.push(LayerQuantPlan {
            base_name: rep.base_name.clone(),
            nominal_rank: rep.nominal_rank,
            recommended_rank: rec_rank,
            nominal_bytes: rep.raw_bytes,
            projected_bytes: proj_bytes,
            savings_pct: savings,
            recommended_dtype: rec_dtype,
            quant_risk: risk,
        });
    }

    let total_savings = if total_nom > 0 {
        ((1.0 - (total_proj as f64 / total_nom as f64)) * 100.0).clamp(-100.0, 99.9)
    } else {
        0.0
    };
    let dtype_flag = if q_bits <= 16.0 { "--dtype bf16" } else { "--dtype f32" };
    let cli = format!(
        "loraspectra compress {} -o compressed.safetensors --energy {:.3} {} --peft-config adapter_config.json",
        source, energy, dtype_flag
    );

    QuantPlanReport {
        source: source.to_string(),
        target_quant: q_norm,
        target_energy: energy,
        layer_count: plan_layers.len(),
        total_nominal_bytes: total_nom,
        total_projected_bytes: total_proj,
        total_savings_pct: total_savings,
        high_risk_layers: high_risk,
        recommended_cli_command: cli,
        layers: plan_layers,
    }
}

pub fn render_quant_plan_table(rep: &QuantPlanReport) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "=== LoRASpectra Quantization-Aware Compression Plan ({}) ===\n",
        rep.source
    ));
    out.push_str(&format!(
        "Target Quant: {} | Energy: {:.1}% | Savings: {:.1}% ({} B -> {} B) | High-Risk Outliers (BF16 kept): {}\n",
        rep.target_quant,
        rep.target_energy * 100.0,
        rep.total_savings_pct,
        rep.total_nominal_bytes,
        rep.total_projected_bytes,
        rep.high_risk_layers
    ));
    out.push_str(&format!("Recommended CLI: {}\n\n", rep.recommended_cli_command));
    out.push_str("Layer | NomR -> RecR | Dtype | Risk | Savings\n");
    out.push_str("------+--------------+-------+------+--------\n");
    for l in &rep.layers {
        out.push_str(&format!(
            "{} | {:>4} -> {:<4} | {:<5} | {:<4} | {:>5.1}%\n",
            l.base_name, l.nominal_rank, l.recommended_rank, l.recommended_dtype, l.quant_risk, l.savings_pct
        ));
    }
    out
}

pub fn render_quant_plan_json(rep: &QuantPlanReport) -> String {
    let items: Vec<String> = rep
        .layers
        .iter()
        .map(|l| {
            format!(
                r#"{{"base_name":"{}","nominal_rank":{},"recommended_rank":{},"nominal_bytes":{},"projected_bytes":{},"savings_pct":{:.2},"recommended_dtype":"{}","quant_risk":"{}"}}"#,
                json_esc(&l.base_name),
                l.nominal_rank,
                l.recommended_rank,
                l.nominal_bytes,
                l.projected_bytes,
                l.savings_pct,
                json_esc(&l.recommended_dtype),
                json_esc(&l.quant_risk)
            )
        })
        .collect();
    format!(
        r#"{{"source":"{}","target_quant":"{}","target_energy":{:.3},"layer_count":{},"total_nominal_bytes":{},"total_projected_bytes":{},"total_savings_pct":{:.2},"high_risk_layers":{},"recommended_cli_command":"{}","layers":[{}]}}"#,
        json_esc(&rep.source),
        json_esc(&rep.target_quant),
        rep.target_energy,
        rep.layer_count,
        rep.total_nominal_bytes,
        rep.total_projected_bytes,
        rep.total_savings_pct,
        rep.high_risk_layers,
        json_esc(&rep.recommended_cli_command),
        items.join(",")
    )
}

fn html_esc(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

/// Renders a self-contained dark-mode HTML quantization trade-off plan report.
pub fn render_quant_plan_html(rep: &QuantPlanReport) -> String {
    let mut rows = String::new();
    for l in &rep.layers {
        let risk_col = match l.quant_risk.as_str() {
            "high" => "#f85149",
            "medium" => "#d29922",
            _ => "#3fb950",
        };
        rows.push_str(&format!(
            "<tr><td>{}</td><td>{} &rarr; {}</td><td><b>{}</b></td><td style=\"color:{}\"><b>{}</b></td><td>{:.1}%</td></tr>\n",
            html_esc(&l.base_name),
            l.nominal_rank,
            l.recommended_rank,
            html_esc(&l.recommended_dtype),
            risk_col,
            html_esc(&l.quant_risk),
            l.savings_pct
        ));
    }
    format!(
        r##"<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>LoRASpectra Quantization Plan</title>
<style>
body{{margin:0;padding:28px;background:#0b0f17;color:#e6edf3;font-family:system-ui,sans-serif}}
.wrap{{max-width:980px;margin:0 auto}}.card{{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:18px;margin-bottom:16px}}
table{{width:100%;border-collapse:collapse;font-family:ui-monospace,monospace;font-size:12.5px}}
th,td{{padding:7px 10px;border-bottom:1px solid #21262d;text-align:left}}th{{color:#8b949e}}
pre{{background:#0d1117;border:1px solid #30363d;border-radius:6px;padding:10px;overflow:auto;color:#58a6ff}}
</style></head><body><div class="wrap">
<div class="card"><h1>LoRASpectra Mixed-Precision Quantization Plan</h1>
<p>Source: <b>{}</b> &middot; Target: <b>{}</b> &middot; Energy: <b>{:.1}%</b> &middot; Projected Savings: <b>{:.1}%</b> &middot; High-Risk Protected Layers: <b>{}</b></p>
<pre>{}</pre></div>
<div class="card"><table><thead><tr><th>Layer</th><th>Rank</th><th>DType</th><th>Risk</th><th>Savings</th></tr></thead>
<tbody>{}</tbody></table></div></div></body></html>"##,
        html_esc(&rep.source),
        html_esc(&rep.target_quant),
        rep.target_energy * 100.0,
        rep.total_savings_pct,
        rep.high_risk_layers,
        html_esc(&rep.recommended_cli_command),
        rows
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::safetensors::{DType, Tensor2D};
    use crate::Rng64;

    #[test]
    fn quant_plan_projects_significant_savings_and_flags_rank1_spike() {
        let mut rng = Rng64::new(42);
        let healthy = LoraLayer {
            base_name: "model.layers.0.self_attn.q_proj".into(),
            a_key: "a".into(),
            b_key: "b".into(),
            a: Tensor2D { rows: 8, cols: 32, dtype: DType::F32, data: (0..256).map(|_| rng.next_gaussian() * 0.1).collect() },
            b: Tensor2D { rows: 32, cols: 8, dtype: DType::F32, data: (0..256).map(|_| rng.next_gaussian() * 0.1).collect() },
        };
        let mut spike_a = vec![0.0; 256];
        let mut spike_b = vec![0.0; 256];
        for i in 0..32 {
            spike_a[i] = 2.0;
            spike_b[i * 8] = 2.0;
        }
        let brittle = LoraLayer {
            base_name: "model.layers.1.mlp.down_proj".into(),
            a_key: "a2".into(),
            b_key: "b2".into(),
            a: Tensor2D { rows: 8, cols: 32, dtype: DType::F32, data: spike_a },
            b: Tensor2D { rows: 32, cols: 8, dtype: DType::F32, data: spike_b },
        };

        let rep = build_quant_plan("adapter.safetensors", &[healthy, brittle], "nf4", 0.99);
        assert_eq!(rep.layer_count, 2);
        assert!(rep.total_savings_pct > 50.0);
        assert_eq!(rep.high_risk_layers, 1);
        assert_eq!(rep.layers[1].recommended_dtype, "bf16");
        assert!(render_quant_plan_json(&rep).contains("\"target_quant\":\"nf4\""));
        let html = render_quant_plan_html(&rep);
        assert!(html.contains("<!doctype html>"));
        assert!(html.contains("bf16"));
    }
}

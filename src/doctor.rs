//! Unified Executive Adapter Health Diagnosis (`loraspectra doctor`).
//!
//! Combines `inspect_adapter_architecture`, QR-Core SVD `audit_adapter`, `audit_adapter_integrity`,
//! `analyze_loraplus`, `analyze_sensitivity`, and `optimize_rank_schedule` into a single A–F
//! executive scorecard with actionable CLI remediation commands.

use std::collections::BTreeMap;

use crate::adalora::optimize_rank_schedule;
use crate::audit::{audit_adapter, audit_adapter_integrity, LayerHealth};
use crate::loraplus::analyze_loraplus;
use crate::safetensors::{
    extract_dora_scales, extract_lora_alphas, inspect_adapter_architecture, LoraLayer,
    PeftAdapterConfig, Tensor2D,
};
use crate::sensitivity::analyze_sensitivity;

#[derive(Debug, Clone)]
pub struct DoctorReport {
    pub source_uri: String,
    pub naming_format: &'static str,
    pub inferred_architecture: &'static str,
    pub layer_count: usize,
    pub min_rank: usize,
    pub max_rank: usize,
    pub total_bytes: usize,
    pub avg_utilization_pct: f64,
    pub dead_layers: usize,
    pub collapsed_layers: usize,
    pub over_param_layers: usize,
    pub integrity_issue_count: usize,
    pub loraplus_mean_imbalance: f64,
    pub loraplus_imbalanced_layers: usize,
    pub forgetting_risk_level: &'static str,
    pub brittle_rank1_layers: usize,
    pub energy_spike_layers: usize,
    pub recommended_avg_rank: f64,
    pub score: f64,
    pub grade: &'static str,
    pub recommendations: Vec<String>,
}

pub fn run_doctor(
    source_uri: &str,
    layers: &[LoraLayer],
    all_tensors: &BTreeMap<String, Tensor2D>,
    peft_config: Option<&PeftAdapterConfig>,
) -> DoctorReport {
    let alphas = extract_lora_alphas(all_tensors);
    let dora = extract_dora_scales(all_tensors);
    let arch = inspect_adapter_architecture(layers, &alphas);
    let audit = audit_adapter(layers);
    let integrity = audit_adapter_integrity(layers, all_tensors, &alphas, &dora, peft_config);
    let lp = analyze_loraplus(layers, source_uri, 1.5, 50.0);
    let sens = analyze_sensitivity(layers, source_uri, 85.0, 2.5);
    let target_r = ((arch.max_rank as f64) * 0.75).max(4.0);
    let sched = optimize_rank_schedule(layers, source_uri, target_r, 4, arch.max_rank.max(4));
    let rec_avg_rank = (sched.recommended_total_rank as f64) / (sched.layers.len().max(1) as f64);

    let n = audit.layers.len().max(1) as f64;
    let avg_util = audit.layers.iter().map(|l| l.utilization_ratio).sum::<f64>() / n;
    let dead = audit.layers.iter().filter(|l| l.health == LayerHealth::DeadLayer).count();
    let collapsed = audit.layers.iter().filter(|l| l.health == LayerHealth::RankCollapsed).count();
    let over_param = audit.layers.iter().filter(|l| l.health == LayerHealth::OverParameterized).count();

    // Composite 0..100 score
    let mut score = 100.0_f64;
    score -= (dead as f64) * 20.0;
    score -= (collapsed as f64) * 8.0;
    score -= (integrity.issues.len() as f64) * 5.0;
    score -= ((lp.mean_imbalance_index - 1.0).max(0.0) * 12.0).min(20.0);
    score -= (sens.brittle_rank1_layers as f64) * 4.0;
    score -= (sens.energy_spike_layers as f64) * 4.0;
    if avg_util < 0.55 {
        score -= (0.55 - avg_util) * 30.0;
    }
    let score = score.clamp(0.0, 100.0);

    let grade = if score >= 90.0 {
        "A"
    } else if score >= 80.0 {
        "B"
    } else if score >= 68.0 {
        "C"
    } else if score >= 52.0 {
        "D"
    } else {
        "F"
    };

    let mut recs = Vec::new();
    if dead > 0 {
        recs.push(format!(
            "CRITICAL: {} dead LoRA layer(s) detected (LS001). Verify optimizer target_modules and freeze masks.",
            dead
        ));
    }
    if lp.imbalanced_layers > 0 {
        recs.push(format!(
            "Run `loraspectra loraplus {} --rebalance balanced.safetensors` to fix {:.2}x LoRA+ gauge imbalance at zero drift.",
            source_uri, lp.mean_imbalance_index
        ));
    }
    if over_param > 0 {
        recs.push(format!(
            "Run `loraspectra compress {} compressed.safetensors --energy 0.99` ({} over-parameterized layers; suggested avg rank {:.1}).",
            source_uri, over_param, rec_avg_rank
        ));
    }
    if sens.brittle_rank1_layers > 0 || sens.energy_spike_layers > 0 {
        recs.push(format!(
            "Forgetting risk {}: {} rank-1 brittle layers (LS051), {} energy spikes (LS052). Consider lower MLP lr or orthogonal projection (`loraspectra project`).",
            sens.forgetting_risk_level, sens.brittle_rank1_layers, sens.energy_spike_layers
        ));
    }
    if recs.is_empty() {
        recs.push("Adapter spectral profile is healthy across SVD utilization, LoRA+ balance, and depth sensitivity.".to_string());
    }

    DoctorReport {
        source_uri: source_uri.to_string(),
        naming_format: arch.naming_format,
        inferred_architecture: arch.inferred_architecture,
        layer_count: audit.layers.len(),
        min_rank: arch.min_rank,
        max_rank: arch.max_rank,
        total_bytes: arch.total_bytes,
        avg_utilization_pct: avg_util * 100.0,
        dead_layers: dead,
        collapsed_layers: collapsed,
        over_param_layers: over_param,
        integrity_issue_count: integrity.issues.len(),
        loraplus_mean_imbalance: lp.mean_imbalance_index,
        loraplus_imbalanced_layers: lp.imbalanced_layers,
        forgetting_risk_level: sens.forgetting_risk_level,
        brittle_rank1_layers: sens.brittle_rank1_layers,
        energy_spike_layers: sens.energy_spike_layers,
        recommended_avg_rank: rec_avg_rank,
        score,
        grade,
        recommendations: recs,
    }
}

pub fn render_doctor_markdown(r: &DoctorReport) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "### LoRASpectra Doctor Report — Grade **{}** ({:.1}/100)\n\n",
        r.grade, r.score
    ));
    out.push_str("| Metric | Value |\n| --- | --- |\n");
    out.push_str(&format!("| Source | `{}` |\n", r.source_uri));
    out.push_str(&format!("| Format / Arch | {} · {} |\n", r.naming_format, r.inferred_architecture));
    out.push_str(&format!("| Layers / Rank | {} layers (r={}..{}, {} bytes) |\n", r.layer_count, r.min_rank, r.max_rank, r.total_bytes));
    out.push_str(&format!("| Mean Utilization | {:.1}% (dead={}, collapsed={}, over-param={}) |\n", r.avg_utilization_pct, r.dead_layers, r.collapsed_layers, r.over_param_layers));
    out.push_str(&format!("| LoRA+ Imbalance | {:.2}x ({} imbalanced layers) |\n", r.loraplus_mean_imbalance, r.loraplus_imbalanced_layers));
    out.push_str(&format!("| Forgetting Risk | **{}** (LS051={}, LS052={}) |\n", r.forgetting_risk_level, r.brittle_rank1_layers, r.energy_spike_layers));
    out.push_str(&format!("| AdaLoRA Target Avg Rank | {:.1} |\n\n", r.recommended_avg_rank));
    out.push_str("#### Recommendations\n");
    for rec in &r.recommendations {
        out.push_str(&format!("- {}\n", rec));
    }
    out
}

fn html_esc(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#39;")
}

pub fn render_doctor_html(r: &DoctorReport) -> String {
    let grade_color = match r.grade {
        "A" | "B" => "#3fb950",
        "C" => "#58a6ff",
        _ => "#f85149",
    };
    let recs_html: String = r
        .recommendations
        .iter()
        .map(|rec| format!("<li>{}</li>", html_esc(rec)))
        .collect::<Vec<_>>()
        .join("\n        ");
    format!(
        r#"<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>LoRASpectra Doctor Report — {src}</title>
  <style>
    :root {{ color-scheme: dark; }}
    body {{ margin:0; background:#0b0f17; color:#e6edf3; font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif; line-height:1.55; }}
    .wrap {{ max-width:940px; margin:0 auto; padding:28px 20px 56px; }}
    .hero {{ background:#161b22; border:1px solid #30363d; border-radius:12px; padding:20px 22px; display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:14px; }}
    .hero h1 {{ margin:0 0 4px; font-size:1.45rem; }}
    .muted {{ color:#8b949e; font-size:0.88rem; }}
    .grade {{ font-size:1.4rem; font-weight:800; padding:8px 18px; border-radius:999px; border:1px solid {gcol}; color:{gcol}; }}
    .kpis {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(175px,1fr)); gap:12px; margin:18px 0 22px; }}
    .kpi {{ background:#161b22; border:1px solid #30363d; border-radius:10px; padding:14px; }}
    .kpi span {{ display:block; color:#8b949e; font-size:0.78rem; text-transform:uppercase; }}
    .kpi strong {{ font-size:1.25rem; color:#58a6ff; }}
    .card {{ background:#161b22; border:1px solid #30363d; border-radius:12px; padding:18px 20px; margin-bottom:16px; }}
    ul {{ margin:8px 0 0 18px; padding:0; }}
    li {{ margin-bottom:6px; }}
    footer {{ text-align:center; color:#8b949e; font-size:0.8rem; margin-top:24px; }}
  </style>
</head>
<body>
  <div class="wrap">
    <header class="hero">
      <div>
        <h1>LoRASpectra Executive Doctor Report</h1>
        <div class="muted"><code>{src}</code> · {fmt} · {arch} · {layers} layers (r={minr}..{maxr}, {bytes} bytes)</div>
      </div>
      <div class="grade">Grade {grade} ({score:.1}/100)</div>
    </header>
    <section class="kpis">
      <div class="kpi"><span>Mean Utilization</span><strong>{util:.1}%</strong></div>
      <div class="kpi"><span>LoRA+ Imbalance</span><strong>{imb:.2}x</strong></div>
      <div class="kpi"><span>Forgetting Risk</span><strong>{risk}</strong></div>
      <div class="kpi"><span>AdaLoRA Target Avg r</span><strong>{recr:.1}</strong></div>
    </section>
    <section class="card">
      <h2>Diagnostics Breakdown</h2>
      <p class="muted">Dead layers: <strong>{dead}</strong> · Collapsed: <strong>{col}</strong> · Over-parameterized: <strong>{over}</strong> · Integrity issues: <strong>{integ}</strong> · LS051 Rank-1 Brittle: <strong>{brittle}</strong> · LS052 Spikes: <strong>{spikes}</strong></p>
    </section>
    <section class="card">
      <h2>Actionable Remediation</h2>
      <ul>
        {recs}
      </ul>
    </section>
    <footer>Generated locally by loraspectra v{ver} · Zero Python/PyTorch required.</footer>
  </div>
</body>
</html>
"#,
        src = html_esc(&r.source_uri),
        fmt = html_esc(r.naming_format),
        arch = html_esc(r.inferred_architecture),
        layers = r.layer_count,
        minr = r.min_rank,
        maxr = r.max_rank,
        bytes = r.total_bytes,
        gcol = grade_color,
        grade = r.grade,
        score = r.score,
        util = r.avg_utilization_pct,
        imb = r.loraplus_mean_imbalance,
        risk = r.forgetting_risk_level,
        recr = r.recommended_avg_rank,
        dead = r.dead_layers,
        col = r.collapsed_layers,
        over = r.over_param_layers,
        integ = r.integrity_issue_count,
        brittle = r.brittle_rank1_layers,
        spikes = r.energy_spike_layers,
        recs = recs_html,
        ver = env!("CARGO_PKG_VERSION"),
    )
}

fn json_esc(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

pub fn render_doctor_json(r: &DoctorReport) -> String {
    let recs: Vec<String> = r
        .recommendations
        .iter()
        .map(|s| format!("\"{}\"", json_esc(s)))
        .collect();
    format!(
        r#"{{
  "source_uri": "{}",
  "grade": "{}",
  "score": {:.2},
  "naming_format": "{}",
  "inferred_architecture": "{}",
  "layer_count": {},
  "min_rank": {},
  "max_rank": {},
  "total_bytes": {},
  "avg_utilization_pct": {:.2},
  "dead_layers": {},
  "collapsed_layers": {},
  "over_param_layers": {},
  "integrity_issue_count": {},
  "loraplus_mean_imbalance": {:.4},
  "loraplus_imbalanced_layers": {},
  "forgetting_risk_level": "{}",
  "brittle_rank1_layers": {},
  "energy_spike_layers": {},
  "recommended_avg_rank": {:.2},
  "recommendations": [{}]
}}"#,
        json_esc(&r.source_uri),
        r.grade,
        r.score,
        json_esc(r.naming_format),
        json_esc(r.inferred_architecture),
        r.layer_count,
        r.min_rank,
        r.max_rank,
        r.total_bytes,
        r.avg_utilization_pct,
        r.dead_layers,
        r.collapsed_layers,
        r.over_param_layers,
        r.integrity_issue_count,
        r.loraplus_mean_imbalance,
        r.loraplus_imbalanced_layers,
        r.forgetting_risk_level,
        r.brittle_rank1_layers,
        r.energy_spike_layers,
        r.recommended_avg_rank,
        recs.join(", ")
    )
}

/// Render an OASIS SARIF v2.1.0 report summarizing executive doctor findings (`LS001`, `LS047`, `LS051`, `LS052`).
pub fn render_doctor_sarif(r: &DoctorReport) -> String {
    let mut results = Vec::new();
    let uri = json_esc(&r.source_uri);
    if r.dead_layers > 0 {
        results.push(format!(
            r#"        {{ "ruleId": "LS001", "level": "error", "message": {{ "text": "{} dead LoRA layer(s) detected in {} (Grade {}, score {:.1}/100)." }}, "locations": [{{ "physicalLocation": {{ "artifactLocation": {{ "uri": "{}" }} }} }}] }}"#,
            r.dead_layers, uri, r.grade, r.score, uri
        ));
    }
    if r.loraplus_imbalanced_layers > 0 {
        results.push(format!(
            r#"        {{ "ruleId": "LS047", "level": "warning", "message": {{ "text": "{} layer(s) exhibit LoRA+ gauge imbalance (mean {:.2}x) in {}." }}, "locations": [{{ "physicalLocation": {{ "artifactLocation": {{ "uri": "{}" }} }} }}] }}"#,
            r.loraplus_imbalanced_layers, r.loraplus_mean_imbalance, uri, uri
        ));
    }
    if r.brittle_rank1_layers > 0 {
        results.push(format!(
            r#"        {{ "ruleId": "LS051", "level": "warning", "message": {{ "text": "{} brittle rank-1 layer(s) detected (forgetting risk: {})." }}, "locations": [{{ "physicalLocation": {{ "artifactLocation": {{ "uri": "{}" }} }} }}] }}"#,
            r.brittle_rank1_layers, r.forgetting_risk_level, uri
        ));
    }
    if r.energy_spike_layers > 0 {
        results.push(format!(
            r#"        {{ "ruleId": "LS052", "level": "warning", "message": {{ "text": "{} depth energy spike layer(s) detected (forgetting risk: {})." }}, "locations": [{{ "physicalLocation": {{ "artifactLocation": {{ "uri": "{}" }} }} }}] }}"#,
            r.energy_spike_layers, r.forgetting_risk_level, uri
        ));
    }
    format!(
        r#"{{
  "version": "2.1.0",
  "$schema": "https://json.schemastore.org/sarif-2.1.0.json",
  "runs": [
    {{
      "tool": {{
        "driver": {{
          "name": "loraspectra-doctor",
          "semanticVersion": "{}",
          "informationUri": "https://github.com/SalyyS1/loraspectra"
        }}
      }},
      "properties": {{
        "grade": "{}",
        "score": {:.2},
        "forgettingRisk": "{}"
      }},
      "results": [
{}
      ]
    }}
  ]
}}
"#,
        env!("CARGO_PKG_VERSION"),
        r.grade,
        r.score,
        r.forgetting_risk_level,
        results.join(",\n")
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::safetensors::{layers_to_tensor_map, DType};
    use crate::Rng64;

    #[test]
    fn doctor_grades_healthy_adapter_higher_than_dead_adapter() {
        let mut rng = Rng64::new(42);
        let good = LoraLayer {
            base_name: "model.layers.0.self_attn.q_proj".to_string(),
            a_key: "model.layers.0.self_attn.q_proj.lora_A.weight".to_string(),
            b_key: "model.layers.0.self_attn.q_proj.lora_B.weight".to_string(),
            a: Tensor2D { rows: 4, cols: 16, dtype: DType::F32, data: (0..64).map(|_| rng.next_gaussian() * 0.2).collect() },
            b: Tensor2D { rows: 16, cols: 4, dtype: DType::F32, data: (0..64).map(|_| rng.next_gaussian() * 0.2).collect() },
        };
        let dead = LoraLayer {
            base_name: "model.layers.0.self_attn.q_proj".to_string(),
            a_key: "model.layers.0.self_attn.q_proj.lora_A.weight".to_string(),
            b_key: "model.layers.0.self_attn.q_proj.lora_B.weight".to_string(),
            a: Tensor2D { rows: 4, cols: 16, dtype: DType::F32, data: vec![0.0; 64] },
            b: Tensor2D { rows: 16, cols: 4, dtype: DType::F32, data: vec![0.0; 64] },
        };
        let g_rep = run_doctor("good.safetensors", &[good.clone()], &layers_to_tensor_map(&[good]), None);
        let d_rep = run_doctor("dead.safetensors", &[dead.clone()], &layers_to_tensor_map(&[dead]), None);
        assert!(g_rep.score > d_rep.score);
        assert!(render_doctor_markdown(&g_rep).contains("Grade **"));
        assert!(render_doctor_json(&d_rep).contains("\"dead_layers\": 1"));
        let html = render_doctor_html(&g_rep);
        assert!(html.contains("<!doctype html>"));
        assert!(html.contains("LoRASpectra Executive Doctor Report"));
        let sarif = render_doctor_sarif(&d_rep);
        assert!(sarif.contains("\"version\": \"2.1.0\""));
        assert!(sarif.contains("LS001"));
    }
}

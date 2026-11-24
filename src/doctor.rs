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
    }
}

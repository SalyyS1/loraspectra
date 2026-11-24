//! LoRA+ Spectral Learning-Rate Asymmetry & Variational Factor-Balance Auditor (`LS047`-`LS048`).
//!
//! Grounded in Hayou et al. (ICML 2024, "LoRA+: Efficient Low Rank Adaptation of Large Models")
//! and the nuclear-norm variational characterization:
//!   `min_{B A = ΔW} 0.5 * (||B||_F^2 + ||A||_F^2) == ||ΔW||_* = sum_i sigma_i(ΔW)`,
//! attained iff `sigma_i(B) = sigma_i(A) = sqrt(sigma_i(ΔW))`.
//!
//! Computes exact `O(r^2 (d_in + d_out))` factor balance indices, spectral condition numbers,
//! recommended per-block LoRA+ learning-rate multiplier `lambda* = eta_B / eta_A`, and
//! zero-drift gauge equilibration (`rebalance_adapter_factors`) that preserves `ΔW = B A`
//! to IEEE 754 `f64` machine precision while minimizing quantization dynamic-range waste.

use crate::linalg::{
    jacobi_svd, qr_core_singular_values, qr_core_svd_truncated, svd_to_lora_factors,
    thin_qr_columns, transpose,
};
use crate::safetensors::{LoraLayer, Tensor2D};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoraPlusSeverity {
    Ok,
    Warning,
    Error,
}

impl LoraPlusSeverity {
    pub fn as_str(&self) -> &'static str {
        match self {
            LoraPlusSeverity::Ok => "note",
            LoraPlusSeverity::Warning => "warning",
            LoraPlusSeverity::Error => "error",
        }
    }
}

#[derive(Debug, Clone)]
pub struct LoraPlusDiagnostic {
    pub rule_id: &'static str,
    pub severity: LoraPlusSeverity,
    pub layer_name: String,
    pub message: String,
    pub imbalance_index: f64,
    pub norm_ratio_b_over_a: f64,
    pub cond_max: f64,
}

#[derive(Debug, Clone)]
pub struct LoraPlusLayerReport {
    pub base_name: String,
    pub block_type: &'static str,
    pub rank: usize,
    pub norm_f_b: f64,
    pub norm_f_a: f64,
    pub norm_ratio_b_over_a: f64,
    pub nuclear_norm_dw: f64,
    pub variational_energy: f64,
    /// `0.5 * (||B||_F^2 + ||A||_F^2) / ||ΔW||_* >= 1.0`. Equals `1.0` iff spectrally balanced.
    pub imbalance_index: f64,
    pub cond_b: f64,
    pub cond_a: f64,
    pub recommended_lambda: f64,
    pub severity: LoraPlusSeverity,
}

#[derive(Debug, Clone)]
pub struct LoraPlusSummary {
    pub source_uri: String,
    pub layer_count: usize,
    pub mean_imbalance_index: f64,
    pub max_imbalance_index: f64,
    pub mean_norm_ratio: f64,
    pub recommended_attn_lambda: f64,
    pub recommended_mlp_lambda: f64,
    pub imbalanced_layers: usize,
    pub ill_conditioned_layers: usize,
    pub layers: Vec<LoraPlusLayerReport>,
    pub diagnostics: Vec<LoraPlusDiagnostic>,
}

fn classify_block(name: &str) -> &'static str {
    let lower = name.to_ascii_lowercase();
    if lower.contains("attn")
        || lower.contains("q_proj")
        || lower.contains("k_proj")
        || lower.contains("v_proj")
        || lower.contains("o_proj")
        || lower.contains("to_q")
        || lower.contains("to_k")
        || lower.contains("to_v")
    {
        "attn"
    } else if lower.contains("mlp")
        || lower.contains("ffn")
        || lower.contains("gate_proj")
        || lower.contains("up_proj")
        || lower.contains("down_proj")
        || lower.contains("fc1")
        || lower.contains("fc2")
    {
        "mlp"
    } else {
        "other"
    }
}

fn factor_condition_number(mat: &Tensor2D) -> f64 {
    let (_, r) = thin_qr_columns(mat);
    let svd = jacobi_svd(&r);
    if svd.sigma.is_empty() {
        return 1.0;
    }
    let s_max = svd.sigma[0];
    let s_min = *svd.sigma.last().unwrap_or(&0.0);
    if s_max <= 1e-12 {
        return 1.0;
    }
    if s_min <= 1e-12 {
        return 1e6;
    }
    (s_max / s_min).min(1e6)
}

/// Audit LoRA+ factor scale imbalance, variational gap, and conditioning across all layers.
pub fn analyze_loraplus(
    layers: &[LoraLayer],
    source_uri: &str,
    imbalance_warn: f64,
    cond_warn: f64,
) -> LoraPlusSummary {
    let mut reports = Vec::with_capacity(layers.len());
    let mut diagnostics = Vec::new();
    let mut attn_lambdas = Vec::new();
    let mut mlp_lambdas = Vec::new();

    for layer in layers {
        let b_sq = layer.b.frobenius_norm_sq();
        let a_sq = layer.a.frobenius_norm_sq();
        let norm_f_b = b_sq.sqrt();
        let norm_f_a = a_sq.sqrt();
        let norm_ratio = if norm_f_a > 1e-12 {
            norm_f_b / norm_f_a
        } else if norm_f_b > 1e-12 {
            1e6
        } else {
            1.0
        };

        let svals = qr_core_singular_values(&layer.b, &layer.a);
        let nuclear_norm: f64 = svals.iter().sum();
        let variational_energy = 0.5 * (b_sq + a_sq);
        let imbalance_index = if nuclear_norm > 1e-12 {
            (variational_energy / nuclear_norm).max(1.0)
        } else {
            1.0
        };

        let cond_b = factor_condition_number(&layer.b);
        let a_t = transpose(&layer.a);
        let cond_a = factor_condition_number(&a_t);
        let cond_max = cond_b.max(cond_a);

        // Estimate recommended LoRA+ lr multiplier lambda* = eta_B / eta_A clamped to [2.0, 16.0].
        // When ||B||_F << ||A||_F, B is under-trained relative to A so lambda* increases toward 16.
        let rec_lambda = if norm_ratio < 1e-6 {
            16.0
        } else {
            (4.0 / norm_ratio.clamp(0.125, 4.0)).clamp(2.0, 16.0)
        };

        let block_type = classify_block(&layer.base_name);
        match block_type {
            "attn" => attn_lambdas.push(rec_lambda),
            "mlp" => mlp_lambdas.push(rec_lambda),
            _ => {}
        }

        let mut severity = LoraPlusSeverity::Ok;
        if nuclear_norm > 1e-10 && imbalance_index >= imbalance_warn {
            severity = if imbalance_index >= imbalance_warn * 2.5 {
                LoraPlusSeverity::Error
            } else {
                LoraPlusSeverity::Warning
            };
            diagnostics.push(LoraPlusDiagnostic {
                rule_id: "LS047",
                severity,
                layer_name: layer.base_name.clone(),
                message: format!(
                    "Layer '{}' exhibits LoRA factor scale imbalance I={:.3} (||B||_F/||A||_F={:.3}, variational energy {:.4} vs nuclear norm {:.4}); run --rebalance or train with LoRA+ lambda*={:.1}",
                    layer.base_name, imbalance_index, norm_ratio, variational_energy, nuclear_norm, rec_lambda
                ),
                imbalance_index,
                norm_ratio_b_over_a: norm_ratio,
                cond_max,
            });
        }
        if nuclear_norm > 1e-10 && cond_max >= cond_warn {
            let c_sev = if cond_max >= cond_warn * 5.0 {
                LoraPlusSeverity::Error
            } else {
                LoraPlusSeverity::Warning
            };
            if c_sev == LoraPlusSeverity::Error {
                severity = LoraPlusSeverity::Error;
            } else if severity == LoraPlusSeverity::Ok {
                severity = LoraPlusSeverity::Warning;
            }
            diagnostics.push(LoraPlusDiagnostic {
                rule_id: "LS048",
                severity: c_sev,
                layer_name: layer.base_name.clone(),
                message: format!(
                    "Layer '{}' has ill-conditioned LoRA factors (kappa(B)={:.1}, kappa(A)={:.1}); quantization/FP8 error will amplify unless SVD-rebalanced",
                    layer.base_name, cond_b, cond_a
                ),
                imbalance_index,
                norm_ratio_b_over_a: norm_ratio,
                cond_max,
            });
        }

        reports.push(LoraPlusLayerReport {
            base_name: layer.base_name.clone(),
            block_type,
            rank: layer.rank(),
            norm_f_b,
            norm_f_a,
            norm_ratio_b_over_a: norm_ratio,
            nuclear_norm_dw: nuclear_norm,
            variational_energy,
            imbalance_index,
            cond_b,
            cond_a,
            recommended_lambda: rec_lambda,
            severity,
        });
    }

    let n = reports.len().max(1) as f64;
    let mean_imbalance = reports.iter().map(|r| r.imbalance_index).sum::<f64>() / n;
    let max_imbalance = reports
        .iter()
        .map(|r| r.imbalance_index)
        .fold(1.0_f64, f64::max);
    let mean_ratio = reports.iter().map(|r| r.norm_ratio_b_over_a).sum::<f64>() / n;
    let rec_attn = if attn_lambdas.is_empty() {
        8.0
    } else {
        attn_lambdas.iter().sum::<f64>() / attn_lambdas.len() as f64
    };
    let rec_mlp = if mlp_lambdas.is_empty() {
        8.0
    } else {
        mlp_lambdas.iter().sum::<f64>() / mlp_lambdas.len() as f64
    };
    let imbalanced_layers = diagnostics.iter().filter(|d| d.rule_id == "LS047").count();
    let ill_conditioned_layers = diagnostics.iter().filter(|d| d.rule_id == "LS048").count();

    LoraPlusSummary {
        source_uri: source_uri.to_string(),
        layer_count: reports.len(),
        mean_imbalance_index: mean_imbalance,
        max_imbalance_index: max_imbalance,
        mean_norm_ratio: mean_ratio,
        recommended_attn_lambda: rec_attn,
        recommended_mlp_lambda: rec_mlp,
        imbalanced_layers,
        ill_conditioned_layers,
        layers: reports,
        diagnostics,
    }
}

/// Gauge-equilibrate every LoRA layer `(B, A) -> (U Sigma^{1/2}, Sigma^{1/2} V^T)` with zero
/// mathematical change to `ΔW = B A` (drives variational imbalance index `I -> 1.000000`).
pub fn rebalance_adapter_factors(layers: &[LoraLayer]) -> Vec<LoraLayer> {
    layers
        .iter()
        .map(|layer| {
            let r = layer.rank();
            let svd = qr_core_svd_truncated(&layer.b, &layer.a, r);
            let keep = svd.sigma.len().min(r).max(1);
            let (new_b, new_a) = svd_to_lora_factors(&svd, keep, layer.b.dtype);
            LoraLayer {
                base_name: layer.base_name.clone(),
                a_key: layer.a_key.clone(),
                b_key: layer.b_key.clone(),
                a: new_a,
                b: new_b,
            }
        })
        .collect()
}

pub fn render_loraplus_table(summary: &LoraPlusSummary) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "{:<44} {:>4} {:>8} {:>8} {:>9} {:>9} {:>7}\n",
        "LAYER", "RANK", "||B||/||A||", "IMBAL_I", "KAPPA_B", "KAPPA_A", "LAMBDA*"
    ));
    out.push_str(&"-".repeat(97));
    out.push('\n');
    for r in &summary.layers {
        out.push_str(&format!(
            "{:<44} {:>4} {:>8.3} {:>8.3} {:>9.1} {:>9.1} {:>6.1}x\n",
            r.base_name,
            r.rank,
            r.norm_ratio_b_over_a,
            r.imbalance_index,
            r.cond_b,
            r.cond_a,
            r.recommended_lambda
        ));
    }
    out.push_str(&"-".repeat(97));
    out.push('\n');
    out.push_str(&format!(
        "Summary: {} layers | Mean imbalance I={:.3} (max {:.3}) | Imbalanced (LS047): {} | Ill-conditioned (LS048): {} | Recommended LoRA+ lambda*: attn={:.1}x, mlp={:.1}x\n",
        summary.layer_count,
        summary.mean_imbalance_index,
        summary.max_imbalance_index,
        summary.imbalanced_layers,
        summary.ill_conditioned_layers,
        summary.recommended_attn_lambda,
        summary.recommended_mlp_lambda
    ));
    out
}

pub fn render_loraplus_json(summary: &LoraPlusSummary) -> String {
    let mut layers_json = Vec::with_capacity(summary.layers.len());
    for r in &summary.layers {
        layers_json.push(format!(
            r#"    {{"layer":"{}","block":"{}","rank":{},"norm_f_b":{:.6},"norm_f_a":{:.6},"norm_ratio_b_over_a":{:.6},"nuclear_norm":{:.6},"imbalance_index":{:.6},"cond_b":{:.4},"cond_a":{:.4},"recommended_lambda":{:.2}}}"#,
            r.base_name.replace('"', "\\\""),
            r.block_type,
            r.rank,
            r.norm_f_b,
            r.norm_f_a,
            r.norm_ratio_b_over_a,
            r.nuclear_norm_dw,
            r.imbalance_index,
            r.cond_b,
            r.cond_a,
            r.recommended_lambda
        ));
    }
    format!(
        "{{\n  \"source\": \"{}\",\n  \"layer_count\": {},\n  \"mean_imbalance_index\": {:.6},\n  \"max_imbalance_index\": {:.6},\n  \"imbalanced_layers\": {},\n  \"ill_conditioned_layers\": {},\n  \"recommended_attn_lambda\": {:.2},\n  \"recommended_mlp_lambda\": {:.2},\n  \"layers\": [\n{}\n  ]\n}}\n",
        summary.source_uri.replace('"', "\\\""),
        summary.layer_count,
        summary.mean_imbalance_index,
        summary.max_imbalance_index,
        summary.imbalanced_layers,
        summary.ill_conditioned_layers,
        summary.recommended_attn_lambda,
        summary.recommended_mlp_lambda,
        layers_json.join(",\n")
    )
}

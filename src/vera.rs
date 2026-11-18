/// VeRA (Vector-based Random Matrix Adaptation) spectral fingerprinting.
///
/// VeRA (Kopiczko et al., NeurIPS 2024, arXiv 2310.11454) replaces per-layer trainable A/B
/// matrices with *shared* frozen random matrices and trains only two per-layer scalar vectors
/// `lambda_b` (length d_out) and `lambda_d` (length r).  The effective rank-r update is:
///
///   DeltaW = diag(lambda_b) * B_frozen * diag(lambda_d) * A_frozen
///
/// where B_frozen in R^{d_out x r} and A_frozen in R^{r x d_in} are drawn once and reused across
/// all layers.  This module detects VeRA adapters, audits their scalar scaling health, and reports
/// the effective spectral fingerprint of each layer's induced update.

use std::collections::BTreeMap;

use crate::linalg::dot_slice;
use crate::safetensors::Tensor2D;

/// Health classification for a single VeRA scaling vector pair.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VeraLayerHealth {
    /// All scaling factors are finite and non-negative.
    Healthy,
    /// lambda_b or lambda_d contains negative entries (sign flip induces cancellation).
    NegativeScale,
    /// lambda_b or lambda_d contains non-finite (NaN or Inf) entries.
    NonFiniteScale,
    /// lambda_d is effectively zero across all rank positions (dead layer).
    DeadRankGating,
    /// lambda_b is effectively zero across all output channels (dead output).
    DeadOutputGating,
}

impl VeraLayerHealth {
    pub fn as_str(&self) -> &'static str {
        match self {
            VeraLayerHealth::Healthy => "HEALTHY",
            VeraLayerHealth::NegativeScale => "NEGATIVE_SCALE",
            VeraLayerHealth::NonFiniteScale => "NON_FINITE_SCALE",
            VeraLayerHealth::DeadRankGating => "DEAD_RANK_GATING",
            VeraLayerHealth::DeadOutputGating => "DEAD_OUTPUT_GATING",
        }
    }
}

/// Per-layer VeRA audit result.
#[derive(Debug, Clone)]
pub struct VeraLayerReport {
    /// Module base name (e.g. `base_model.model.layers.0.self_attn.q_proj`).
    pub base_name: String,
    /// Rank `r` inferred from lambda_d length.
    pub rank: usize,
    /// Output dimension inferred from lambda_b length.
    pub d_out: usize,
    /// L2 norm of lambda_b.
    pub lambda_b_norm: f64,
    /// L2 norm of lambda_d.
    pub lambda_d_norm: f64,
    /// Mean absolute value of lambda_b.
    pub lambda_b_mean_abs: f64,
    /// Mean absolute value of lambda_d.
    pub lambda_d_mean_abs: f64,
    /// Coefficient of variation of |lambda_b| -- measures output channel scale dispersion.
    pub lambda_b_cv: f64,
    /// Coefficient of variation of |lambda_d| -- measures rank dimension scale dispersion.
    pub lambda_d_cv: f64,
    /// Fraction of lambda_d entries with absolute value < 1e-6 (gated-off rank slots).
    pub dead_rank_fraction: f64,
    /// Fraction of lambda_b entries with absolute value < 1e-6 (dead output channels).
    pub dead_output_fraction: f64,
    /// Effective spectral norm lower bound: max(|lambda_b|) * max(|lambda_d|).
    /// Tight lower bound on ||DeltaW||_2 when B_frozen and A_frozen are isotropic.
    pub spectral_norm_lower_bound: f64,
    /// Effective nuclear norm proxy: ||lambda_b||_2 * ||lambda_d||_2.
    /// Proportional to ||DeltaW||_* when random bases are orthonormal.
    pub nuclear_norm_proxy: f64,
    /// Health classification.
    pub health: VeraLayerHealth,
    /// SARIF rule IDs triggered.
    pub issues: Vec<&'static str>,
}

impl VeraLayerReport {
    /// True when any critical issue is present.
    pub fn is_critical(&self) -> bool {
        self.issues.iter().any(|&id| id == "LS025")
    }
}

/// Summary report for an entire VeRA adapter.
#[derive(Debug, Clone)]
pub struct VeraAdapterReport {
    pub total_layers: usize,
    pub healthy_layers: usize,
    pub dead_rank_layers: usize,
    pub dead_output_layers: usize,
    pub negative_scale_layers: usize,
    pub non_finite_layers: usize,
    pub mean_lambda_b_cv: f64,
    pub mean_lambda_d_cv: f64,
    pub mean_dead_rank_fraction: f64,
    pub mean_dead_output_fraction: f64,
    pub layers: Vec<VeraLayerReport>,
}

/// Compute coefficient of variation of absolute values.
fn cv_abs(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let abs_vals: Vec<f64> = values.iter().map(|v| v.abs()).collect();
    let mean = abs_vals.iter().sum::<f64>() / abs_vals.len() as f64;
    if mean < 1e-18 {
        return 0.0;
    }
    let var = abs_vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / abs_vals.len() as f64;
    var.sqrt() / mean
}

/// Detect VeRA scaling vectors from a raw safetensors tensor map.
///
/// Returns a map from base_name to `(lambda_b_tensor, lambda_d_tensor)`.
/// VeRA tensors are identified by the suffixes:
///   - `.vera_lambda_b` / `.vera_lambda_d`  (PEFT format)
///   - `.lora_lambda_b` / `.lora_lambda_d`  (some community checkpoints)
///   - `.scaling_b` / `.scaling_d`           (alternative naming)
pub fn detect_vera_tensors(
    tensor_map: &BTreeMap<String, Tensor2D>,
) -> BTreeMap<String, (Tensor2D, Tensor2D)> {
    // Collect lambda_b candidates
    let mut lambda_b_map: BTreeMap<String, Tensor2D> = BTreeMap::new();
    let mut lambda_d_map: BTreeMap<String, Tensor2D> = BTreeMap::new();

    for (key, tensor) in tensor_map {
        let key = key.as_str();
        if let Some(base) = key.strip_suffix(".vera_lambda_b")
            .or_else(|| key.strip_suffix(".lora_lambda_b"))
            .or_else(|| key.strip_suffix(".scaling_b"))
        {
            lambda_b_map.insert(base.to_string(), tensor.clone());
        } else if let Some(base) = key.strip_suffix(".vera_lambda_d")
            .or_else(|| key.strip_suffix(".lora_lambda_d"))
            .or_else(|| key.strip_suffix(".scaling_d"))
        {
            lambda_d_map.insert(base.to_string(), tensor.clone());
        }
    }

    // Keep only layers that have BOTH lambda_b and lambda_d
    let mut result: BTreeMap<String, (Tensor2D, Tensor2D)> = BTreeMap::new();
    for (base_name, lb) in &lambda_b_map {
        if let Some(ld) = lambda_d_map.get(base_name) {
            result.insert(base_name.clone(), (lb.clone(), ld.clone()));
        }
    }
    result
}

/// Audit a single VeRA layer's scaling vectors.
pub fn audit_vera_layer(
    base_name: &str,
    lambda_b: &Tensor2D,
    lambda_d: &Tensor2D,
) -> VeraLayerReport {
    // Flatten to 1D (VeRA stores these as [1, r] or [d_out] etc.)
    let lb: Vec<f64> = lambda_b.data.clone();
    let ld: Vec<f64> = lambda_d.data.clone();

    let d_out = lb.len().max(1);
    let rank = ld.len().max(1);

    // Compute basic stats
    let lb_norm = dot_slice(&lb, &lb).sqrt();
    let ld_norm = dot_slice(&ld, &ld).sqrt();

    let lb_mean_abs = lb.iter().map(|v| v.abs()).sum::<f64>() / d_out as f64;
    let ld_mean_abs = ld.iter().map(|v| v.abs()).sum::<f64>() / rank as f64;

    let lb_cv = cv_abs(&lb);
    let ld_cv = cv_abs(&ld);

    let lb_max_abs = lb.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
    let ld_max_abs = ld.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);

    let dead_rank_fraction = ld.iter().filter(|&&v| v.abs() < 1e-6).count() as f64 / rank as f64;
    let dead_output_fraction = lb.iter().filter(|&&v| v.abs() < 1e-6).count() as f64 / d_out as f64;

    let spectral_norm_lower_bound = lb_max_abs * ld_max_abs;
    let nuclear_norm_proxy = lb_norm * ld_norm;

    // Classify health
    let has_non_finite = lb.iter().chain(ld.iter()).any(|v| !v.is_finite());
    let has_negative = lb.iter().chain(ld.iter()).any(|&v| v < 0.0);
    let dead_rank = dead_rank_fraction > 0.95;
    let dead_output = dead_output_fraction > 0.95;

    let health = if has_non_finite {
        VeraLayerHealth::NonFiniteScale
    } else if dead_rank {
        VeraLayerHealth::DeadRankGating
    } else if dead_output {
        VeraLayerHealth::DeadOutputGating
    } else if has_negative {
        VeraLayerHealth::NegativeScale
    } else {
        VeraLayerHealth::Healthy
    };

    // Emit SARIF issues
    let mut issues: Vec<&'static str> = Vec::new();
    if has_non_finite || dead_rank || dead_output {
        issues.push("LS025");
    }
    if has_negative || lb_cv > 2.0 || ld_cv > 2.0 {
        issues.push("LS026");
    }

    VeraLayerReport {
        base_name: base_name.to_string(),
        rank,
        d_out,
        lambda_b_norm: lb_norm,
        lambda_d_norm: ld_norm,
        lambda_b_mean_abs: lb_mean_abs,
        lambda_d_mean_abs: ld_mean_abs,
        lambda_b_cv: lb_cv,
        lambda_d_cv: ld_cv,
        dead_rank_fraction,
        dead_output_fraction,
        spectral_norm_lower_bound,
        nuclear_norm_proxy,
        health,
        issues,
    }
}

/// Audit all VeRA layers detected in a tensor map.
pub fn audit_vera_adapter(tensor_map: &BTreeMap<String, Tensor2D>) -> VeraAdapterReport {
    let vera_pairs = detect_vera_tensors(tensor_map);

    let mut layer_reports: Vec<VeraLayerReport> = vera_pairs
        .iter()
        .map(|(base_name, (lb, ld))| audit_vera_layer(base_name, lb, ld))
        .collect();
    layer_reports.sort_by(|a, b| a.base_name.cmp(&b.base_name));

    let total = layer_reports.len();
    let healthy = layer_reports.iter().filter(|r| r.health == VeraLayerHealth::Healthy).count();
    let dead_rank = layer_reports.iter().filter(|r| r.health == VeraLayerHealth::DeadRankGating).count();
    let dead_output = layer_reports.iter().filter(|r| r.health == VeraLayerHealth::DeadOutputGating).count();
    let negative = layer_reports.iter().filter(|r| r.health == VeraLayerHealth::NegativeScale).count();
    let non_finite = layer_reports.iter().filter(|r| r.health == VeraLayerHealth::NonFiniteScale).count();

    let (mean_lb_cv, mean_ld_cv, mean_dr, mean_do) = if total > 0 {
        let n = total as f64;
        (
            layer_reports.iter().map(|r| r.lambda_b_cv).sum::<f64>() / n,
            layer_reports.iter().map(|r| r.lambda_d_cv).sum::<f64>() / n,
            layer_reports.iter().map(|r| r.dead_rank_fraction).sum::<f64>() / n,
            layer_reports.iter().map(|r| r.dead_output_fraction).sum::<f64>() / n,
        )
    } else {
        (0.0, 0.0, 0.0, 0.0)
    };

    VeraAdapterReport {
        total_layers: total,
        healthy_layers: healthy,
        dead_rank_layers: dead_rank,
        dead_output_layers: dead_output,
        negative_scale_layers: negative,
        non_finite_layers: non_finite,
        mean_lambda_b_cv: mean_lb_cv,
        mean_lambda_d_cv: mean_ld_cv,
        mean_dead_rank_fraction: mean_dr,
        mean_dead_output_fraction: mean_do,
        layers: layer_reports,
    }
}

/// Render a VeRA audit report as compact JSON.
pub fn render_vera_json(report: &VeraAdapterReport) -> String {
    let mut out = String::new();
    out.push_str("{\n");
    out.push_str(&format!("  \"total_layers\": {},\n", report.total_layers));
    out.push_str(&format!("  \"healthy_layers\": {},\n", report.healthy_layers));
    out.push_str(&format!("  \"dead_rank_layers\": {},\n", report.dead_rank_layers));
    out.push_str(&format!("  \"dead_output_layers\": {},\n", report.dead_output_layers));
    out.push_str(&format!("  \"negative_scale_layers\": {},\n", report.negative_scale_layers));
    out.push_str(&format!("  \"non_finite_layers\": {},\n", report.non_finite_layers));
    out.push_str(&format!("  \"mean_lambda_b_cv\": {:.6},\n", report.mean_lambda_b_cv));
    out.push_str(&format!("  \"mean_lambda_d_cv\": {:.6},\n", report.mean_lambda_d_cv));
    out.push_str(&format!("  \"mean_dead_rank_fraction\": {:.6},\n", report.mean_dead_rank_fraction));
    out.push_str(&format!("  \"mean_dead_output_fraction\": {:.6},\n", report.mean_dead_output_fraction));
    out.push_str("  \"layers\": [\n");
    for (i, layer) in report.layers.iter().enumerate() {
        out.push_str("    {\n");
        out.push_str(&format!("      \"base_name\": \"{}\",\n", layer.base_name));
        out.push_str(&format!("      \"rank\": {},\n", layer.rank));
        out.push_str(&format!("      \"d_out\": {},\n", layer.d_out));
        out.push_str(&format!("      \"lambda_b_norm\": {:.6},\n", layer.lambda_b_norm));
        out.push_str(&format!("      \"lambda_d_norm\": {:.6},\n", layer.lambda_d_norm));
        out.push_str(&format!("      \"lambda_b_cv\": {:.6},\n", layer.lambda_b_cv));
        out.push_str(&format!("      \"lambda_d_cv\": {:.6},\n", layer.lambda_d_cv));
        out.push_str(&format!("      \"dead_rank_fraction\": {:.6},\n", layer.dead_rank_fraction));
        out.push_str(&format!("      \"dead_output_fraction\": {:.6},\n", layer.dead_output_fraction));
        out.push_str(&format!("      \"spectral_norm_lower_bound\": {:.6},\n", layer.spectral_norm_lower_bound));
        out.push_str(&format!("      \"nuclear_norm_proxy\": {:.6},\n", layer.nuclear_norm_proxy));
        out.push_str(&format!("      \"health\": \"{}\",\n", layer.health.as_str()));
        out.push_str(&format!(
            "      \"issues\": [{}]\n",
            layer.issues.iter().map(|s| format!("\"{}\"", s)).collect::<Vec<_>>().join(", ")
        ));
        out.push_str("    }");
        if i + 1 < report.layers.len() {
            out.push(',');
        }
        out.push('\n');
    }
    out.push_str("  ]\n");
    out.push('}');
    out
}

/// Render a VeRA audit report as a human-readable table.
pub fn render_vera_table(report: &VeraAdapterReport) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "VeRA Adapter Audit: {}/{} healthy layers, {:.1}% mean dead rank slots\n",
        report.healthy_layers,
        report.total_layers,
        report.mean_dead_rank_fraction * 100.0
    ));
    out.push_str(&format!(
        "lambda_b CV (output channel dispersion): {:.4}  lambda_d CV (rank gating dispersion): {:.4}\n\n",
        report.mean_lambda_b_cv, report.mean_lambda_d_cv
    ));
    out.push_str(&format!(
        "{:<52} {:>5} {:>7} {:>8} {:>8} {:>12} {:>14} {:<}\n",
        "Layer", "rank", "d_out", "lb_cv", "ld_cv", "dead_rank%", "spectral_lb", "health"
    ));
    out.push_str(&"-".repeat(130));
    out.push('\n');
    for layer in &report.layers {
        let short_name = if layer.base_name.len() > 52 {
            format!("...{}", &layer.base_name[layer.base_name.len() - 49..])
        } else {
            layer.base_name.clone()
        };
        let issues_str = if layer.issues.is_empty() {
            String::new()
        } else {
            format!(" [{}]", layer.issues.join(", "))
        };
        out.push_str(&format!(
            "{:<52} {:>5} {:>7} {:>8.4} {:>8.4} {:>11.1}% {:>14.6} {}{}\n",
            short_name,
            layer.rank,
            layer.d_out,
            layer.lambda_b_cv,
            layer.lambda_d_cv,
            layer.dead_rank_fraction * 100.0,
            layer.spectral_norm_lower_bound,
            layer.health.as_str(),
            issues_str
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::safetensors::DType;

    fn make_tensor1d(data: Vec<f64>, dtype: DType) -> Tensor2D {
        let n = data.len();
        Tensor2D { rows: 1, cols: n, dtype, data }
    }

    #[test]
    fn test_vera_detect_peft_format() {
        let mut map: BTreeMap<String, Tensor2D> = BTreeMap::new();
        // Two VeRA layers
        map.insert(
            "base_model.model.layers.0.self_attn.q_proj.vera_lambda_b".to_string(),
            make_tensor1d(vec![1.0; 64], DType::F32),
        );
        map.insert(
            "base_model.model.layers.0.self_attn.q_proj.vera_lambda_d".to_string(),
            make_tensor1d(vec![0.5; 8], DType::F32),
        );
        map.insert(
            "base_model.model.layers.0.self_attn.v_proj.vera_lambda_b".to_string(),
            make_tensor1d(vec![2.0; 64], DType::F32),
        );
        map.insert(
            "base_model.model.layers.0.self_attn.v_proj.vera_lambda_d".to_string(),
            make_tensor1d(vec![0.1; 8], DType::F32),
        );
        // Non-VeRA tensor that should be ignored
        map.insert(
            "base_model.model.layers.0.self_attn.q_proj.lora_A.weight".to_string(),
            Tensor2D::zeros(8, 64, DType::F32),
        );

        let detected = detect_vera_tensors(&map);
        assert_eq!(detected.len(), 2);
        assert!(detected.contains_key("base_model.model.layers.0.self_attn.q_proj"));
        assert!(detected.contains_key("base_model.model.layers.0.self_attn.v_proj"));
    }

    #[test]
    fn test_vera_detect_alternative_naming() {
        let mut map: BTreeMap<String, Tensor2D> = BTreeMap::new();
        map.insert("model.layer0.scaling_b".to_string(), make_tensor1d(vec![1.0; 32], DType::F32));
        map.insert("model.layer0.scaling_d".to_string(), make_tensor1d(vec![0.5; 4], DType::F32));
        // lora_lambda variant
        map.insert("model.layer1.lora_lambda_b".to_string(), make_tensor1d(vec![1.0; 32], DType::F32));
        map.insert("model.layer1.lora_lambda_d".to_string(), make_tensor1d(vec![0.5; 4], DType::F32));

        let detected = detect_vera_tensors(&map);
        assert_eq!(detected.len(), 2);
    }

    #[test]
    fn test_vera_healthy_layer_audit() {
        let lb = make_tensor1d(vec![1.0; 64], DType::F32);
        let ld = make_tensor1d(vec![0.5; 8], DType::F32);
        let report = audit_vera_layer("model.layers.0.q_proj", &lb, &ld);

        assert_eq!(report.health, VeraLayerHealth::Healthy);
        assert_eq!(report.rank, 8);
        assert_eq!(report.d_out, 64);
        assert_eq!(report.dead_rank_fraction, 0.0);
        assert_eq!(report.dead_output_fraction, 0.0);
        // lambda_b CV = 0 when all entries are identical
        assert!(report.lambda_b_cv < 1e-10);
        assert!(report.lambda_d_cv < 1e-10);
        // Spectral norm lower bound: max(|lb|) * max(|ld|) = 1.0 * 0.5
        assert!((report.spectral_norm_lower_bound - 0.5).abs() < 1e-10);
        // Nuclear norm proxy: ||lb||_2 * ||ld||_2 = sqrt(64) * sqrt(8 * 0.25) = 8 * sqrt(2)
        let expected_nn = (64.0_f64).sqrt() * (8.0 * 0.25_f64).sqrt();
        assert!((report.nuclear_norm_proxy - expected_nn).abs() < 1e-6);
        assert!(report.issues.is_empty());
    }

    #[test]
    fn test_vera_dead_rank_gating_detection() {
        // lambda_d is all zeros -> dead rank gating
        let lb = make_tensor1d(vec![1.0; 64], DType::F32);
        let ld = make_tensor1d(vec![0.0; 8], DType::F32);
        let report = audit_vera_layer("model.layers.0.q_proj", &lb, &ld);

        assert_eq!(report.health, VeraLayerHealth::DeadRankGating);
        assert_eq!(report.dead_rank_fraction, 1.0);
        assert!(report.issues.contains(&"LS025"));
    }

    #[test]
    fn test_vera_dead_output_gating_detection() {
        // lambda_b is all zeros -> dead output gating
        let lb = make_tensor1d(vec![0.0; 64], DType::F32);
        let ld = make_tensor1d(vec![0.5; 8], DType::F32);
        let report = audit_vera_layer("model.layers.0.q_proj", &lb, &ld);

        assert_eq!(report.health, VeraLayerHealth::DeadOutputGating);
        assert_eq!(report.dead_output_fraction, 1.0);
        assert!(report.issues.contains(&"LS025"));
    }

    #[test]
    fn test_vera_non_finite_scale_detection() {
        let lb = make_tensor1d(vec![1.0, f64::NAN, 1.0], DType::F32);
        let ld = make_tensor1d(vec![0.5; 4], DType::F32);
        let report = audit_vera_layer("model.layers.0.v_proj", &lb, &ld);

        assert_eq!(report.health, VeraLayerHealth::NonFiniteScale);
        assert!(report.issues.contains(&"LS025"));
    }

    #[test]
    fn test_vera_negative_scale_detection() {
        // Negative entries in lambda_b/lambda_d
        let lb = make_tensor1d(vec![1.0, -0.5, 1.0, 0.8], DType::F32);
        let ld = make_tensor1d(vec![0.5; 4], DType::F32);
        let report = audit_vera_layer("model.layers.0.v_proj", &lb, &ld);

        assert_eq!(report.health, VeraLayerHealth::NegativeScale);
        assert!(report.issues.contains(&"LS026"));
    }

    #[test]
    fn test_vera_high_dispersion_triggers_ls026() {
        // Extreme imbalance in lambda_b: one huge, rest tiny -> very high CV
        let mut lb_data = vec![0.001_f64; 64];
        lb_data[0] = 100.0; // outlier
        let lb = make_tensor1d(lb_data, DType::F32);
        let ld = make_tensor1d(vec![0.5; 8], DType::F32);
        let report = audit_vera_layer("model.layers.0.q_proj", &lb, &ld);

        assert!(report.lambda_b_cv > 2.0, "Expected high CV, got {}", report.lambda_b_cv);
        assert!(report.issues.contains(&"LS026"));
    }

    #[test]
    fn test_vera_adapter_report_aggregation() {
        let mut map: BTreeMap<String, Tensor2D> = BTreeMap::new();

        // Healthy layer
        map.insert(
            "model.layers.0.q_proj.vera_lambda_b".to_string(),
            make_tensor1d(vec![1.0; 32], DType::F32),
        );
        map.insert(
            "model.layers.0.q_proj.vera_lambda_d".to_string(),
            make_tensor1d(vec![0.5; 4], DType::F32),
        );

        // Dead rank layer
        map.insert(
            "model.layers.1.q_proj.vera_lambda_b".to_string(),
            make_tensor1d(vec![1.0; 32], DType::F32),
        );
        map.insert(
            "model.layers.1.q_proj.vera_lambda_d".to_string(),
            make_tensor1d(vec![0.0; 4], DType::F32),
        );

        let report = audit_vera_adapter(&map);
        assert_eq!(report.total_layers, 2);
        assert_eq!(report.healthy_layers, 1);
        assert_eq!(report.dead_rank_layers, 1);
        assert_eq!(report.dead_output_layers, 0);
        assert!((report.mean_dead_rank_fraction - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_vera_json_render_contains_expected_fields() {
        let mut map: BTreeMap<String, Tensor2D> = BTreeMap::new();
        map.insert(
            "model.q_proj.vera_lambda_b".to_string(),
            make_tensor1d(vec![1.0; 16], DType::F32),
        );
        map.insert(
            "model.q_proj.vera_lambda_d".to_string(),
            make_tensor1d(vec![0.5; 4], DType::F32),
        );
        let report = audit_vera_adapter(&map);
        let json = render_vera_json(&report);
        assert!(json.contains("\"total_layers\""));
        assert!(json.contains("\"lambda_b_cv\""));
        assert!(json.contains("\"spectral_norm_lower_bound\""));
        assert!(json.contains("HEALTHY"));
    }

    #[test]
    fn test_vera_table_render_no_panic() {
        let mut map: BTreeMap<String, Tensor2D> = BTreeMap::new();
        map.insert(
            "model.q_proj.vera_lambda_b".to_string(),
            make_tensor1d(vec![1.0; 16], DType::F32),
        );
        map.insert(
            "model.q_proj.vera_lambda_d".to_string(),
            make_tensor1d(vec![0.5; 4], DType::F32),
        );
        let report = audit_vera_adapter(&map);
        let table = render_vera_table(&report);
        assert!(table.contains("VeRA Adapter Audit"));
        assert!(table.contains("HEALTHY"));
    }

    #[test]
    fn test_vera_mixed_format_detection() {
        // scaling_b/scaling_d + vera_lambda_b/d in same map
        let mut map: BTreeMap<String, Tensor2D> = BTreeMap::new();
        map.insert("layer_a.scaling_b".to_string(), make_tensor1d(vec![1.0; 8], DType::F32));
        map.insert("layer_a.scaling_d".to_string(), make_tensor1d(vec![0.1; 4], DType::F32));
        map.insert("layer_b.vera_lambda_b".to_string(), make_tensor1d(vec![2.0; 8], DType::F32));
        map.insert("layer_b.vera_lambda_d".to_string(), make_tensor1d(vec![0.2; 4], DType::F32));
        // Orphan (only lambda_b, no lambda_d) - should be ignored
        map.insert("layer_c.vera_lambda_b".to_string(), make_tensor1d(vec![1.0; 8], DType::F32));

        let detected = detect_vera_tensors(&map);
        assert_eq!(detected.len(), 2, "Orphan should be excluded");
        assert!(detected.contains_key("layer_a"));
        assert!(detected.contains_key("layer_b"));
    }

    #[test]
    fn test_vera_spectral_norm_bound_correctness() {
        // lambda_b = [3.0, 1.0, 2.0], lambda_d = [0.5, 1.5, 0.1]
        // spectral_norm_lower_bound = max(|lb|) * max(|ld|) = 3.0 * 1.5 = 4.5
        let lb = make_tensor1d(vec![3.0, 1.0, 2.0], DType::F32);
        let ld = make_tensor1d(vec![0.5, 1.5, 0.1], DType::F32);
        let report = audit_vera_layer("test.layer", &lb, &ld);
        assert!((report.spectral_norm_lower_bound - 4.5).abs() < 1e-10);
    }
}

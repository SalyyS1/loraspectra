
// ----- fp8 scale factor validation (E4M3 / E5M2) ----------------------------------------

/// FP8 format variant.
///
/// Two FP8 formats are standardized by the OFP8 group (arXiv 2209.05433) and used in
/// TransformerEngine / llm.int8() 8-bit weight-only quantization:
///
///   E4M3: 1 sign + 4 exponent + 3 mantissa bits.
///         Finite range: [-448, 448].  NaN encoded as S=1, E=1111, M=111.
///         No infinities (all E=1111 non-NaN patterns used for mantissa).
///
///   E5M2: 1 sign + 5 exponent + 2 mantissa bits.
///         Finite range: [-57344, 57344].  IEEE-style: infinities and NaN.
///         Recommended for gradients due to wider dynamic range.
///
/// References:
///   Micikevicius et al. (2022). "FP8 Formats for Deep Learning." arXiv 2209.05433.
///   NVIDIA TransformerEngine FP8 recipe documentation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fp8Format {
    /// 1-sign, 4-exponent, 3-mantissa (weights, activations).
    E4M3,
    /// 1-sign, 5-exponent, 2-mantissa (gradients, wider range).
    E5M2,
}

impl Fp8Format {
    /// Maximum finite positive value representable in this FP8 format.
    pub fn max_finite(&self) -> f64 {
        match self {
            Fp8Format::E4M3 => 448.0,
            Fp8Format::E5M2 => 57344.0,
        }
    }

    /// Minimum positive normal value (smallest normalized fp8 value).
    pub fn min_normal(&self) -> f64 {
        match self {
            // E4M3: min normal = 2^(1-7) * 1.0 = 2^{-6} = 0.015625  (bias=7)
            Fp8Format::E4M3 => 2.0_f64.powi(-6),
            // E5M2: min normal = 2^(1-15) * 1.0 = 2^{-14}             (bias=15)
            Fp8Format::E5M2 => 2.0_f64.powi(-14),
        }
    }

    /// Minimum positive subnormal value (epsilon for denormals).
    pub fn min_subnormal(&self) -> f64 {
        match self {
            // E4M3: smallest denormal = 2^(1-7) * 2^{-3} = 2^{-9}
            Fp8Format::E4M3 => 2.0_f64.powi(-9),
            // E5M2: smallest denormal = 2^(1-15) * 2^{-2} = 2^{-16}
            Fp8Format::E5M2 => 2.0_f64.powi(-16),
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Fp8Format::E4M3 => "E4M3",
            Fp8Format::E5M2 => "E5M2",
        }
    }

    pub fn from_name(s: &str) -> Result<Self, String> {
        match s.trim().to_ascii_lowercase().as_str() {
            "e4m3" | "fp8_e4m3" | "float8_e4m3fn" | "e4m3fn" => Ok(Fp8Format::E4M3),
            "e5m2" | "fp8_e5m2" | "float8_e5m2" | "e5m2fnuz" => Ok(Fp8Format::E5M2),
            other => Err(format!(
                "Unsupported FP8 format '{}': expected e4m3 or e5m2",
                other
            )),
        }
    }
}

/// Severity classification for a single fp8 scale factor issue.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fp8ScaleSeverity {
    /// Scale is NaN or infinite: layer cannot be used.
    Critical,
    /// Scale is zero or negative: layer will produce zero or sign-inverted output.
    Error,
    /// Scale would cause weights to overflow the fp8 max finite value.
    Warning,
    /// Scale is in the subnormal range: potential precision loss on dequantization.
    Info,
}

impl Fp8ScaleSeverity {
    pub fn as_str(&self) -> &'static str {
        match self {
            Fp8ScaleSeverity::Critical => "CRITICAL",
            Fp8ScaleSeverity::Error => "ERROR",
            Fp8ScaleSeverity::Warning => "WARNING",
            Fp8ScaleSeverity::Info => "INFO",
        }
    }
}

/// A single diagnostic finding for an FP8 scale factor tensor.
#[derive(Debug, Clone)]
pub struct Fp8ScaleIssue {
    /// Tensor key in the safetensors map (e.g. `base_model.model.layers.0.q_proj.weight_scale`).
    pub key: String,
    /// FP8 format for which the scale was validated.
    pub format: Fp8Format,
    /// The scale factor value that triggered this issue.
    pub scale_value: f64,
    /// Maximum absolute value of the weight tensor entries divided by scale (effective max code).
    pub effective_max_code: Option<f64>,
    /// Issue severity.
    pub severity: Fp8ScaleSeverity,
    /// Human-readable explanation.
    pub message: String,
    /// SARIF-style rule code.
    pub rule_code: &'static str,
}

/// SARIF rule IDs for FP8 scale factor validation.
pub const FP8_SCALE_NAN_INF: &str = "LS031";
pub const FP8_SCALE_NON_POSITIVE: &str = "LS032";
pub const FP8_SCALE_OVERFLOW: &str = "LS033";
pub const FP8_SCALE_SUBNORMAL: &str = "LS034";

/// Summary of FP8 scale factor validation for a single safetensors checkpoint.
#[derive(Debug, Clone)]
pub struct Fp8ScaleReport {
    /// FP8 format validated against.
    pub format: Fp8Format,
    /// Total scale factor tensors found and validated.
    pub total_scale_tensors: usize,
    /// Critical issues (NaN/Inf scale).
    pub critical_count: usize,
    /// Error-level issues (zero or negative scale).
    pub error_count: usize,
    /// Warning-level issues (overflow).
    pub warning_count: usize,
    /// Info-level issues (subnormal).
    pub info_count: usize,
    /// All per-tensor issues found.
    pub issues: Vec<Fp8ScaleIssue>,
}

impl Fp8ScaleReport {
    /// True when the checkpoint has at least one critical or error issue.
    pub fn has_blocking_issues(&self) -> bool {
        self.critical_count > 0 || self.error_count > 0
    }
}

/// Validate FP8 scale factor tensors in a safetensors map against the given FP8 format.
///
/// Scale factor tensors are identified by key patterns:
///   - ends with `_scale` or `_scale_inv`
///   - ends with `.weight_scale` or `.input_scale` or `.output_scale`
///   - contains `fp8_scale` or `quant_scale`
///
/// For each scale tensor, each scalar value is validated against the format's representable range.
/// When the corresponding weight tensor is present, the effective max code
/// (max(|weight|) / scale) is also checked for overflow.
///
/// # Arguments
/// * `tensor_map` - Safetensors tensor map as returned by `read_safetensors`.
/// * `format` - The FP8 format to validate against (E4M3 or E5M2).
pub fn validate_fp8_scale_factors(
    tensor_map: &std::collections::BTreeMap<String, crate::safetensors::Tensor2D>,
    format: Fp8Format,
) -> Fp8ScaleReport {
    let max_finite = format.max_finite();
    let _min_normal = format.min_normal();
    let min_subnormal = format.min_subnormal();

    let mut issues: Vec<Fp8ScaleIssue> = Vec::new();
    let mut total_scale_tensors = 0usize;

    for (key, tensor) in tensor_map {
        if !is_scale_key(key) {
            continue;
        }
        total_scale_tensors += 1;

        // For each scalar in the scale tensor
        for &val in &tensor.data {
            if !val.is_finite() {
                issues.push(Fp8ScaleIssue {
                    key: key.clone(),
                    format,
                    scale_value: val,
                    effective_max_code: None,
                    severity: Fp8ScaleSeverity::Critical,
                    message: format!(
                        "Scale factor is {} (non-finite); dequantization will produce NaN/Inf weights",
                        if val.is_nan() { "NaN" } else { "Inf" }
                    ),
                    rule_code: FP8_SCALE_NAN_INF,
                });
                continue;
            }
            if val <= 0.0 {
                issues.push(Fp8ScaleIssue {
                    key: key.clone(),
                    format,
                    scale_value: val,
                    effective_max_code: None,
                    severity: Fp8ScaleSeverity::Error,
                    message: format!(
                        "Scale factor is {:.6e} (<= 0); zero or negative scale inverts/zeros weight magnitudes",
                        val
                    ),
                    rule_code: FP8_SCALE_NON_POSITIVE,
                });
                continue;
            }
            // Check subnormal range
            if val < min_subnormal {
                issues.push(Fp8ScaleIssue {
                    key: key.clone(),
                    format,
                    scale_value: val,
                    effective_max_code: None,
                    severity: Fp8ScaleSeverity::Info,
                    message: format!(
                        "Scale factor {:.6e} is below {} minimum subnormal {:.6e}; \
                         dequantized values will underflow to 0",
                        val,
                        format.as_str(),
                        min_subnormal
                    ),
                    rule_code: FP8_SCALE_SUBNORMAL,
                });
                continue;
            }

            // Look up the corresponding weight tensor to check for overflow
            let weight_key = scale_key_to_weight_key(key);
            let effective_max_code = weight_key.as_deref().and_then(|wk| {
                tensor_map.get(wk).map(|wt| {
                    wt.data
                        .iter()
                        .cloned()
                        .fold(0.0_f64, |acc, x| acc.max(x.abs()))
                        / val
                })
            });

            // If the effective max code exceeds max_finite, the scale is too small (overflow)
            if let Some(eff_max) = effective_max_code {
                if eff_max > max_finite {
                    issues.push(Fp8ScaleIssue {
                        key: key.clone(),
                        format,
                        scale_value: val,
                        effective_max_code: Some(eff_max),
                        severity: Fp8ScaleSeverity::Warning,
                        message: format!(
                            "Scale {:.6e} maps max weight magnitude to {:.2} which exceeds \
                             {} max_finite ({:.1}); clipping will occur on quantization",
                            val,
                            eff_max,
                            format.as_str(),
                            max_finite
                        ),
                        rule_code: FP8_SCALE_OVERFLOW,
                    });
                }
            }
        }
    }

    let critical_count = issues.iter().filter(|i| i.severity == Fp8ScaleSeverity::Critical).count();
    let error_count = issues.iter().filter(|i| i.severity == Fp8ScaleSeverity::Error).count();
    let warning_count = issues.iter().filter(|i| i.severity == Fp8ScaleSeverity::Warning).count();
    let info_count = issues.iter().filter(|i| i.severity == Fp8ScaleSeverity::Info).count();

    Fp8ScaleReport {
        format,
        total_scale_tensors,
        critical_count,
        error_count,
        warning_count,
        info_count,
        issues,
    }
}

/// True when a tensor key looks like a scale factor tensor.
fn is_scale_key(key: &str) -> bool {
    let lower = key.to_ascii_lowercase();
    lower.ends_with("_scale")
        || lower.ends_with("_scale_inv")
        || lower.ends_with(".weight_scale")
        || lower.ends_with(".input_scale")
        || lower.ends_with(".output_scale")
        || lower.contains("fp8_scale")
        || lower.contains("quant_scale")
        || lower.ends_with("_amax")
        || lower.ends_with(".amax_history")
}

/// Attempt to derive the corresponding weight tensor key from a scale key.
/// E.g. `layers.0.q_proj.weight_scale` -> `layers.0.q_proj.weight`.
fn scale_key_to_weight_key(scale_key: &str) -> Option<String> {
    let suffixes = [
        ("_scale_inv", ""),
        ("_scale", ""),
        (".weight_scale", ".weight"),
        (".input_scale", ".weight"),
        (".output_scale", ".weight"),
    ];
    for (suffix, replacement) in &suffixes {
        if scale_key.ends_with(suffix) {
            let base = &scale_key[..scale_key.len() - suffix.len()];
            return Some(format!("{}{}", base, replacement));
        }
    }
    None
}

/// Render a Fp8ScaleReport as a compact JSON object.
pub fn render_fp8_scale_json(report: &Fp8ScaleReport) -> String {
    let mut out = String::new();
    out.push_str("{\n");
    out.push_str(&format!("  \"format\": \"{}\",\n", report.format.as_str()));
    out.push_str(&format!(
        "  \"total_scale_tensors\": {},\n",
        report.total_scale_tensors
    ));
    out.push_str(&format!("  \"critical\": {},\n", report.critical_count));
    out.push_str(&format!("  \"error\": {},\n", report.error_count));
    out.push_str(&format!("  \"warning\": {},\n", report.warning_count));
    out.push_str(&format!("  \"info\": {},\n", report.info_count));
    out.push_str("  \"issues\": [\n");
    for (i, issue) in report.issues.iter().enumerate() {
        let comma = if i + 1 < report.issues.len() { "," } else { "" };
        out.push_str("    {\n");
        out.push_str(&format!(
            "      \"key\": \"{}\",\n",
            fp8_json_escape(&issue.key)
        ));
        out.push_str(&format!("      \"rule\": \"{}\",\n", issue.rule_code));
        out.push_str(&format!(
            "      \"severity\": \"{}\",\n",
            issue.severity.as_str()
        ));
        out.push_str(&format!(
            "      \"scale_value\": {},\n",
            fmt_f64(issue.scale_value)
        ));
        if let Some(emc) = issue.effective_max_code {
            out.push_str(&format!("      \"effective_max_code\": {:.6},\n", emc));
        }
        out.push_str(&format!(
            "      \"message\": \"{}\"\n",
            fp8_json_escape(&issue.message)
        ));
        out.push_str(&format!("    }}{}\n", comma));
    }
    out.push_str("  ]\n");
    out.push('}');
    out
}

/// Render a Fp8ScaleReport as a human-readable text table.
pub fn render_fp8_scale_table(report: &Fp8ScaleReport) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "FP8 Scale Factor Validation  (format: {})\n",
        report.format.as_str()
    ));
    out.push_str(&format!(
        "  Scale tensors found : {}\n",
        report.total_scale_tensors
    ));
    out.push_str(&format!("  Critical            : {}\n", report.critical_count));
    out.push_str(&format!("  Error               : {}\n", report.error_count));
    out.push_str(&format!("  Warning             : {}\n", report.warning_count));
    out.push_str(&format!("  Info                : {}\n\n", report.info_count));
    if report.issues.is_empty() {
        out.push_str("  All scale factors are valid.\n");
        return out;
    }
    out.push_str(&format!(
        "{:<8}  {:<8}  {:>14}  {}\n",
        "Rule", "Severity", "Scale", "Key"
    ));
    out.push_str(&"-".repeat(80));
    out.push('\n');
    for issue in &report.issues {
        out.push_str(&format!(
            "{:<8}  {:<8}  {:>14.4e}  {}\n",
            issue.rule_code,
            issue.severity.as_str(),
            issue.scale_value,
            issue.key
        ));
    }
    out
}

fn fp8_html_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

/// Render a shields.io Endpoint Badge JSON summarizing FP8 E4M3/E5M2 scale validation (LS031-LS034).
pub fn render_fp8_badge_json(report: &Fp8ScaleReport) -> String {
    let ok = report.critical_count == 0 && report.error_count == 0;
    let color = if ok && report.warning_count == 0 {
        "brightgreen"
    } else if ok {
        "yellow"
    } else {
        "red"
    };
    let msg = if ok {
        format!("{} PASS ({} scales, {} warn)", report.format.as_str(), report.total_scale_tensors, report.warning_count)
    } else {
        format!("{} FAIL ({} crit, {} err)", report.format.as_str(), report.critical_count, report.error_count)
    };
    format!(
        r#"{{"schemaVersion":1,"label":"loraspectra fp8","message":"{}","color":"{}"}}"#,
        fp8_json_escape(&msg),
        color
    )
}

/// Render a self-contained dark-mode HTML FP8 E4M3/E5M2 Scale Factor Validation Report (`fp8-validate --html`).
pub fn render_fp8_scale_html(report: &Fp8ScaleReport) -> String {
    let mut rows = String::new();
    if report.issues.is_empty() {
        rows.push_str(
            "<tr><td colspan=\"5\" style=\"color:#3fb950\"><b>PASS &mdash; All FP8 scale factor tensors are finite, positive, and within dynamic range.</b></td></tr>\n",
        );
    } else {
        for iss in &report.issues {
            let col = match iss.severity {
                Fp8ScaleSeverity::Critical | Fp8ScaleSeverity::Error => "#f85149",
                Fp8ScaleSeverity::Warning => "#d29922",
                Fp8ScaleSeverity::Info => "#58a6ff",
            };
            rows.push_str(&format!(
                "<tr><td><code>{}</code></td><td><code>{}</code></td><td style=\"color:{}\"><b>{}</b></td><td><code>{}</code></td><td>{}</td></tr>\n",
                fp8_html_escape(&iss.key),
                fp8_html_escape(iss.rule_code),
                col,
                iss.severity.as_str(),
                fp8_html_escape(&fmt_f64(iss.scale_value)),
                fp8_html_escape(&iss.message),
            ));
        }
    }
    let ok = report.critical_count == 0 && report.error_count == 0;
    let status_col = if ok { "#3fb950" } else { "#f85149" };
    format!(
        r##"<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>loraspectra FP8 Scale Validation ({})</title>
<style>
body{{margin:0;padding:28px;background:#0b0f17;color:#e6edf3;font-family:system-ui,sans-serif}}
.wrap{{max-width:1060px;margin:0 auto}}.card{{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:18px;margin-bottom:16px}}
table{{width:100%;border-collapse:collapse;font-size:12.5px}}th,td{{padding:8px 9px;border-bottom:1px solid #21262d;text-align:left}}
th{{color:#8b949e}}code{{font-family:ui-monospace,monospace;color:#58a6ff}}
</style></head><body><div class="wrap">
<div class="card"><h1>loraspectra FP8 {} Scale Factor Validation (LS031&ndash;LS034)</h1>
<p>Status: <b style="color:{}">{}</b> &middot; Scale Tensors: <b>{}</b> &middot; Critical: <b>{}</b> &middot; Error: <b>{}</b> &middot; Warning: <b>{}</b> &middot; Info: <b>{}</b></p></div>
<div class="card"><table><thead><tr><th>Tensor Key</th><th>Rule</th><th>Severity</th><th>Scale</th><th>Detail</th></tr></thead>
<tbody>{}</tbody></table></div></div></body></html>"##,
        report.format.as_str(),
        report.format.as_str(),
        status_col,
        if ok { "PASS" } else { "FAIL" },
        report.total_scale_tensors,
        report.critical_count,
        report.error_count,
        report.warning_count,
        report.info_count,
        rows
    )
}

fn fmt_f64(v: f64) -> String {
    if v.is_nan() {
        "null".to_string()
    } else if v.is_infinite() {
        if v > 0.0 { "1e308".to_string() } else { "-1e308".to_string() }
    } else {
        format!("{:.6e}", v)
    }
}

fn fp8_json_escape(s: &str) -> String {
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

#[cfg(test)]
mod fp8_tests {
    use super::*;
    use crate::safetensors::{DType, Tensor2D};
    use std::collections::BTreeMap;

    fn scalar_tensor(value: f64) -> Tensor2D {
        Tensor2D { rows: 1, cols: 1, data: vec![value], dtype: DType::F32 }
    }

    fn weight_tensor(max_abs: f64) -> Tensor2D {
        Tensor2D {
            rows: 1,
            cols: 4,
            data: vec![max_abs, -max_abs * 0.5, 0.1, -0.1],
            dtype: DType::F32,
        }
    }

    // --- Fp8Format unit tests ---

    #[test]
    fn test_fp8_e4m3_max_finite() {
        assert_eq!(Fp8Format::E4M3.max_finite(), 448.0);
    }

    #[test]
    fn test_fp8_e5m2_max_finite() {
        assert_eq!(Fp8Format::E5M2.max_finite(), 57344.0);
    }

    #[test]
    fn test_fp8_e4m3_min_normal() {
        let expected = 2.0_f64.powi(-6);
        assert!((Fp8Format::E4M3.min_normal() - expected).abs() < 1e-15);
    }

    #[test]
    fn test_fp8_e5m2_min_normal() {
        let expected = 2.0_f64.powi(-14);
        assert!((Fp8Format::E5M2.min_normal() - expected).abs() < 1e-20);
    }

    #[test]
    fn test_fp8_e4m3_min_subnormal() {
        let expected = 2.0_f64.powi(-9);
        assert!((Fp8Format::E4M3.min_subnormal() - expected).abs() < 1e-15);
    }

    #[test]
    fn test_fp8_e5m2_min_subnormal() {
        let expected = 2.0_f64.powi(-16);
        assert!((Fp8Format::E5M2.min_subnormal() - expected).abs() < 1e-20);
    }

    #[test]
    fn test_fp8_format_from_name_e4m3() {
        assert_eq!(Fp8Format::from_name("e4m3").unwrap(), Fp8Format::E4M3);
        assert_eq!(Fp8Format::from_name("float8_e4m3fn").unwrap(), Fp8Format::E4M3);
        assert_eq!(Fp8Format::from_name("E4M3FN").unwrap(), Fp8Format::E4M3);
    }

    #[test]
    fn test_fp8_format_from_name_e5m2() {
        assert_eq!(Fp8Format::from_name("e5m2").unwrap(), Fp8Format::E5M2);
        assert_eq!(Fp8Format::from_name("fp8_e5m2").unwrap(), Fp8Format::E5M2);
    }

    #[test]
    fn test_fp8_format_from_name_invalid() {
        assert!(Fp8Format::from_name("fp16").is_err());
        assert!(Fp8Format::from_name("bf16").is_err());
    }

    // --- validate_fp8_scale_factors tests ---

    #[test]
    fn test_no_scale_tensors_returns_empty_report() {
        let mut map = BTreeMap::new();
        map.insert("model.layers.0.q_proj.weight".to_string(), weight_tensor(1.0));
        let report = validate_fp8_scale_factors(&map, Fp8Format::E4M3);
        assert_eq!(report.total_scale_tensors, 0);
        assert_eq!(report.issues.len(), 0);
        assert!(!report.has_blocking_issues());
    }

    #[test]
    fn test_valid_scale_no_issues() {
        let mut map = BTreeMap::new();
        // weight_scale = 1.0, weight max_abs = 100.0 < 448
        map.insert("model.layers.0.q_proj.weight_scale".to_string(), scalar_tensor(1.0));
        map.insert("model.layers.0.q_proj.weight".to_string(), weight_tensor(100.0));
        let report = validate_fp8_scale_factors(&map, Fp8Format::E4M3);
        assert_eq!(report.total_scale_tensors, 1);
        assert_eq!(report.issues.len(), 0);
        assert!(!report.has_blocking_issues());
    }

    #[test]
    fn test_nan_scale_critical() {
        let mut map = BTreeMap::new();
        map.insert("model.q_proj_scale".to_string(), scalar_tensor(f64::NAN));
        let report = validate_fp8_scale_factors(&map, Fp8Format::E4M3);
        assert_eq!(report.critical_count, 1);
        assert_eq!(report.issues[0].rule_code, FP8_SCALE_NAN_INF);
        assert_eq!(report.issues[0].severity, Fp8ScaleSeverity::Critical);
        assert!(report.has_blocking_issues());
    }

    #[test]
    fn test_inf_scale_critical() {
        let mut map = BTreeMap::new();
        map.insert("model.q_proj_scale".to_string(), scalar_tensor(f64::INFINITY));
        let report = validate_fp8_scale_factors(&map, Fp8Format::E4M3);
        assert_eq!(report.critical_count, 1);
        assert_eq!(report.issues[0].rule_code, FP8_SCALE_NAN_INF);
    }

    #[test]
    fn test_zero_scale_error() {
        let mut map = BTreeMap::new();
        map.insert("layer.weight_scale".to_string(), scalar_tensor(0.0));
        let report = validate_fp8_scale_factors(&map, Fp8Format::E4M3);
        assert_eq!(report.error_count, 1);
        assert_eq!(report.issues[0].rule_code, FP8_SCALE_NON_POSITIVE);
        assert!(report.has_blocking_issues());
    }

    #[test]
    fn test_negative_scale_error() {
        let mut map = BTreeMap::new();
        map.insert("layer.weight_scale".to_string(), scalar_tensor(-0.5));
        let report = validate_fp8_scale_factors(&map, Fp8Format::E4M3);
        assert_eq!(report.error_count, 1);
        assert_eq!(report.issues[0].rule_code, FP8_SCALE_NON_POSITIVE);
    }

    #[test]
    fn test_subnormal_scale_info() {
        let mut map = BTreeMap::new();
        // Below min_subnormal of E4M3 (2^-9 ~ 1.95e-3)
        map.insert("layer.weight_scale".to_string(), scalar_tensor(1e-10));
        let report = validate_fp8_scale_factors(&map, Fp8Format::E4M3);
        assert_eq!(report.info_count, 1);
        assert_eq!(report.issues[0].rule_code, FP8_SCALE_SUBNORMAL);
    }

    #[test]
    fn test_overflow_scale_warning() {
        let mut map = BTreeMap::new();
        // scale=1.0, weight max_abs=1000.0: effective_max_code = 1000 > 448 (E4M3 max)
        map.insert("model.q_proj.weight_scale".to_string(), scalar_tensor(1.0));
        map.insert("model.q_proj.weight".to_string(), weight_tensor(1000.0));
        let report = validate_fp8_scale_factors(&map, Fp8Format::E4M3);
        assert_eq!(report.warning_count, 1);
        assert_eq!(report.issues[0].rule_code, FP8_SCALE_OVERFLOW);
    }

    #[test]
    fn test_e5m2_no_overflow_within_range() {
        let mut map = BTreeMap::new();
        // scale=1.0, weight max_abs=1000.0: 1000 < 57344 (E5M2 max) -> no overflow
        map.insert("model.q_proj.weight_scale".to_string(), scalar_tensor(1.0));
        map.insert("model.q_proj.weight".to_string(), weight_tensor(1000.0));
        let report = validate_fp8_scale_factors(&map, Fp8Format::E5M2);
        assert_eq!(report.warning_count, 0);
        assert_eq!(report.issues.len(), 0);
    }

    #[test]
    fn test_render_fp8_json_valid() {
        let mut map = BTreeMap::new();
        map.insert("model.q_proj_scale".to_string(), scalar_tensor(f64::NAN));
        let report = validate_fp8_scale_factors(&map, Fp8Format::E4M3);
        let json = render_fp8_scale_json(&report);
        assert!(json.contains("\"format\""));
        assert!(json.contains("\"E4M3\""));
        assert!(json.contains("LS031"));
    }

    #[test]
    fn test_render_fp8_table_valid() {
        let mut map = BTreeMap::new();
        map.insert("model.q_proj.weight_scale".to_string(), scalar_tensor(1.0));
        map.insert("model.q_proj.weight".to_string(), weight_tensor(100.0));
        let report = validate_fp8_scale_factors(&map, Fp8Format::E4M3);
        let table = render_fp8_scale_table(&report);
        assert!(table.contains("FP8 Scale Factor Validation"));
        assert!(table.contains("E4M3"));
    }

    #[test]
    fn test_is_scale_key_patterns() {
        assert!(is_scale_key("model.q_proj.weight_scale"));
        assert!(is_scale_key("model.q_proj.input_scale"));
        assert!(is_scale_key("model.q_proj.output_scale"));
        assert!(is_scale_key("layer_scale"));
        assert!(is_scale_key("layer_scale_inv"));
        assert!(is_scale_key("fp8_scale_factor"));
        assert!(!is_scale_key("model.q_proj.weight"));
        assert!(!is_scale_key("model.lora_A"));
    }
}

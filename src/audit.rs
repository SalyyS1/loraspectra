use std::collections::{BTreeMap, BTreeSet};

use crate::linalg::qr_core_singular_values;
use crate::safetensors::{ LoraLayer, PeftAdapterConfig, Tensor2D };

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerHealth {
    Optimal,
    OverParameterized,
    RankCollapsed,
    DeadLayer,
}

impl LayerHealth {
    pub fn as_str(&self) -> &'static str {
        match self {
            LayerHealth::Optimal => "OPTIMAL",
            LayerHealth::OverParameterized => "OVER_PARAMETERIZED",
            LayerHealth::RankCollapsed => "RANK_COLLAPSED",
            LayerHealth::DeadLayer => "DEAD_LAYER",
        }
    }
}

#[derive(Debug, Clone)]
pub struct LayerSpectralReport {
    pub base_name: String,
    pub d_out: usize,
    pub d_in: usize,
    pub nominal_rank: usize,
    pub singular_values: Vec<f64>,
    pub spectral_norm: f64,
    pub frobenius_norm: f64,
    pub stable_rank: f64,
    pub effective_rank: f64,
    pub rank_at_90_energy: usize,
    pub rank_at_95_energy: usize,
    pub rank_at_99_energy: usize,
    pub rank_marchenko_pastur: usize,
    pub mp_noise_threshold: f64,
    pub utilization_ratio: f64,
    pub raw_bytes: usize,
    pub bytes_at_99_energy: usize,
    pub health: LayerHealth,
}

#[derive(Debug, Clone)]
pub struct AdapterAuditSummary {
    pub layers: Vec<LayerSpectralReport>,
    pub total_nominal_bytes: usize,
    pub total_optimal_99_bytes: usize,
    pub mean_nominal_rank: f64,
    pub mean_effective_rank: f64,
    pub mean_stable_rank: f64,
    pub mean_rank_at_99_energy: f64,
    pub mean_rank_mp: f64,
    pub compressible_byte_ratio: f64,
}

/// Gavish & Donoho (2014) Marchenko-Pastur optimal hard threshold estimator for a rank-`r`
/// LoRA singular spectrum. Returns `(optimal_signal_rank, tau_threshold)`.
pub fn estimate_marchenko_pastur_rank(
    singular_values: &[f64],
    d_out: usize,
    d_in: usize,
) -> (usize, f64) {
    let r = singular_values.len();
    if r == 0 {
        return (1, 0.0);
    }
    let sigma_1 = singular_values[0];
    if sigma_1 <= 1e-12 || r <= 2 {
        return (r.max(1), 0.0);
    }

    let min_d = d_out.min(d_in).max(1) as f64;
    let max_d = d_out.max(d_in).max(1) as f64;
    let beta = (min_d / max_d).clamp(1e-4, 1.0);

    // Gavish & Donoho (2014) polynomial approximation for omega(beta) under unknown noise level
    let omega = 0.56 * beta.powi(3) - 0.95 * beta.powi(2) + 1.82 * beta + 1.43;

    // Estimate the noise bulk median from the lower half of the rank-r spectrum
    let tail = &singular_values[r / 2..];
    let mid = tail.len() / 2;
    let tail_median = if tail.len() % 2 == 1 {
        tail[mid]
    } else {
        0.5 * (tail[mid - 1] + tail[mid])
    };

    let tau = omega * tail_median;

    // If the top singular value does not exceed tau, the spectrum is flat/isotropic (all modes active)
    if sigma_1 <= tau {
        return (r, tau);
    }

    let signal_k = singular_values
        .iter()
        .take_while(|&&s| s > tau)
        .count()
        .max(1)
        .min(r);
    (signal_k, tau)
}

/// Compute spectral metrics for a single `LoraLayer` using fast QR-Core SVD.
pub fn audit_layer(layer: &LoraLayer) -> LayerSpectralReport {
    let sigma = qr_core_singular_values(&layer.b, &layer.a);
    audit_from_singular_values(
        &layer.base_name,
        layer.d_out(),
        layer.d_in(),
        layer.rank(),
        layer.a.dtype.byte_size(),
        sigma,
    )
}

pub fn audit_from_singular_values(
    base_name: &str,
    d_out: usize,
    d_in: usize,
    nominal_rank: usize,
    elem_bytes: usize,
    singular_values: Vec<f64>,
) -> LayerSpectralReport {
    let spectral_norm = singular_values.first().copied().unwrap_or(0.0);
    let total_energy: f64 = singular_values.iter().map(|&s| s * s).sum();
    let frobenius_norm = total_energy.sqrt();

    if frobenius_norm < 1e-9 || spectral_norm < 1e-9 {
        let raw_bytes = nominal_rank * (d_out + d_in) * elem_bytes;
        let min_bytes = (d_out + d_in) * elem_bytes;
        return LayerSpectralReport {
            base_name: base_name.to_string(),
            d_out,
            d_in,
            nominal_rank,
            singular_values,
            spectral_norm,
            frobenius_norm,
            stable_rank: 0.0,
            effective_rank: 0.0,
            rank_at_90_energy: 1,
            rank_at_95_energy: 1,
            rank_at_99_energy: 1,
            rank_marchenko_pastur: 1,
            mp_noise_threshold: 0.0,
            utilization_ratio: 0.0,
            raw_bytes,
            bytes_at_99_energy: min_bytes,
            health: LayerHealth::DeadLayer,
        };
    }

    let stable_rank = total_energy / (spectral_norm * spectral_norm);

    // Roy & Vetterli (2007) von Neumann spectral entropy effective rank:
    // p_i = sigma_i / sum(sigma_j), H = -sum(p_i * ln(p_i)), r_eff = exp(H)
    let l1_sum: f64 = singular_values.iter().map(|&s| s.max(0.0)).sum();
    let mut entropy = 0.0;
    if l1_sum > 0.0 {
        for &s in &singular_values {
            if s > 1e-15 {
                let p = s / l1_sum;
                entropy -= p * p.ln();
            }
        }
    }
    let effective_rank = entropy.exp();

    let find_energy_rank = |threshold: f64| -> usize {
        let mut cum = 0.0;
        for (idx, &s) in singular_values.iter().enumerate() {
            cum += s * s;
            if cum / total_energy >= threshold {
                return idx + 1;
            }
        }
        nominal_rank.max(1)
    };

    let rank_at_90_energy = find_energy_rank(0.90);
    let rank_at_95_energy = find_energy_rank(0.95);
    let rank_at_99_energy = find_energy_rank(0.99);
    let (rank_marchenko_pastur, mp_noise_threshold) =
        estimate_marchenko_pastur_rank(&singular_values, d_out, d_in);

    let utilization_ratio = (rank_at_99_energy as f64) / (nominal_rank.max(1) as f64);
    let raw_bytes = nominal_rank * (d_out + d_in) * elem_bytes;
    let bytes_at_99_energy = rank_at_99_energy * (d_out + d_in) * elem_bytes;

    let health = if effective_rank < 2.0 || (nominal_rank >= 8 && utilization_ratio <= 0.15) {
        LayerHealth::RankCollapsed
    } else if utilization_ratio <= 0.55 {
        LayerHealth::OverParameterized
    } else {
        LayerHealth::Optimal
    };

    LayerSpectralReport {
        base_name: base_name.to_string(),
        d_out,
        d_in,
        nominal_rank,
        singular_values,
        spectral_norm,
        frobenius_norm,
        stable_rank,
        effective_rank,
        rank_at_90_energy,
        rank_at_95_energy,
        rank_at_99_energy,
        rank_marchenko_pastur,
        mp_noise_threshold,
        utilization_ratio,
        raw_bytes,
        bytes_at_99_energy,
        health,
    }
}

/// Audit all LoRA layers in an adapter and produce an aggregate summary.
pub fn audit_adapter(layers: &[LoraLayer]) -> AdapterAuditSummary {
    let reports: Vec<LayerSpectralReport> = layers.iter().map(audit_layer).collect();
    let n = reports.len().max(1) as f64;

    let total_nominal_bytes: usize = reports.iter().map(|r| r.raw_bytes).sum();
    let total_optimal_99_bytes: usize = reports.iter().map(|r| r.bytes_at_99_energy).sum();

    let mean_nominal_rank = reports.iter().map(|r| r.nominal_rank as f64).sum::<f64>() / n;
    let mean_effective_rank = reports.iter().map(|r| r.effective_rank).sum::<f64>() / n;
    let mean_stable_rank = reports.iter().map(|r| r.stable_rank).sum::<f64>() / n;
    let mean_rank_at_99_energy = reports.iter().map(|r| r.rank_at_99_energy as f64).sum::<f64>() / n;
    let mean_rank_mp = reports
        .iter()
        .map(|r| r.rank_marchenko_pastur as f64)
        .sum::<f64>()
        / n;

    let compressible_byte_ratio = if total_nominal_bytes > 0 {
        1.0 - (total_optimal_99_bytes as f64) / (total_nominal_bytes as f64)
    } else {
        0.0
    };

    AdapterAuditSummary {
        layers: reports,
        total_nominal_bytes,
        total_optimal_99_bytes,
        mean_nominal_rank,
        mean_effective_rank,
        mean_stable_rank,
        mean_rank_at_99_energy,
        mean_rank_mp,
        compressible_byte_ratio,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IssueSeverity {
    Critical,
    Warning,
    Info,
}

impl IssueSeverity {
    pub fn as_str(&self) -> &'static str {
        match self {
            IssueSeverity::Critical => "CRITICAL",
            IssueSeverity::Warning => "WARNING",
            IssueSeverity::Info => "INFO",
        }
    }
}

#[derive(Debug, Clone)]
pub struct IntegrityIssue {
    pub code: &'static str,
    pub severity: IssueSeverity,
    pub subject: String,
    pub message: String,
}

#[derive(Debug, Clone)]
pub struct EmbeddingAuditEntry {
    pub name: String,
    pub kind: &'static str,
    pub vocab_rows: usize,
    pub hidden_dim: usize,
    pub zero_norm_rows: usize,
    pub trailing_zero_rows: usize,
    pub dead_lora_b_cols: usize,
    pub dead_lora_a_rows: usize,
}

#[derive(Debug, Clone)]
pub struct AlphaScalingSummary {
    pub alpha_tensor_count: usize,
    pub config_present: bool,
    pub config_lora_alpha: Option<f64>,
    pub use_rslora: bool,
    pub min_alpha_r_ratio: Option<f64>,
    pub max_alpha_r_ratio: Option<f64>,
    pub dora_tensor_count: usize,
    pub status: &'static str,
}

#[derive(Debug, Clone)]
pub struct AdapterIntegrityReport {
    pub status: &'static str,
    pub dead_lora_layers: usize,
    pub embedding_entries: Vec<EmbeddingAuditEntry>,
    pub alpha_summary: AlphaScalingSummary,
    pub issues: Vec<IntegrityIssue>,
}

/// Classify whether a tensor key or LoRA `base_name` targets a token embedding or unembedding head.
pub fn classify_token_module(name: &str) -> Option<&'static str> {
    let lower = name.to_ascii_lowercase();
    if lower.contains("embed_tokens")
        || lower.contains("tok_embeddings")
        || lower.contains("word_embeddings")
        || lower.contains("lora_embedding")
    {
        return Some("embedding");
    }
    if lower.contains("lm_head") || lower.contains("embed_out") {
        return Some("lm_head");
    }
    for seg in lower.split(['.', '/', '_']) {
        if seg == "wte" {
            return Some("embedding");
        }
    }
    None
}

fn count_zero_rows(t: &Tensor2D, eps: f64) -> (usize, usize) {
    let eps_sq = eps * eps;
    let mut zero_count = 0usize;
    let mut row_is_zero = vec![false; t.rows];
    for r in 0..t.rows {
        let off = r * t.cols;
        let mut sq = 0.0;
        for c in 0..t.cols {
            let v = t.data[off + c];
            sq += v * v;
        }
        if !sq.is_finite() || sq <= eps_sq {
            zero_count += 1;
            row_is_zero[r] = true;
        }
    }
    let trailing = row_is_zero.iter().rev().take_while(|&&z| z).count();
    (zero_count, trailing)
}

fn count_zero_cols(t: &Tensor2D, eps: f64) -> (usize, usize) {
    let eps_sq = eps * eps;
    let mut zero_count = 0usize;
    let mut col_is_zero = vec![false; t.cols];
    for c in 0..t.cols {
        let mut sq = 0.0;
        for r in 0..t.rows {
            let v = t.data[r * t.cols + c];
            sq += v * v;
        }
        if !sq.is_finite() || sq <= eps_sq {
            zero_count += 1;
            col_is_zero[c] = true;
        }
    }
    let trailing = col_is_zero.iter().rev().take_while(|&&z| z).count();
    (zero_count, trailing)
}

fn lookup_config_alpha(cfg: &PeftAdapterConfig, base_name: &str) -> Option<f64> {
    if let Some(&v) = cfg.alpha_pattern.get(base_name) {
        return Some(v);
    }
    let stripped = base_name
        .strip_prefix("base_model.model.")
        .unwrap_or(base_name);
    if let Some(&v) = cfg.alpha_pattern.get(stripped) {
        return Some(v);
    }
    for (pat, &val) in &cfg.alpha_pattern {
        if base_name.ends_with(pat.as_str()) {
            return Some(val);
        }
    }
    cfg.lora_alpha
}

/// Run comprehensive adapter integrity diagnostics to detect silent SFT bugs:
/// - Zero-norm / dead rows in saved token embeddings (`embed_tokens`, `wte`, `lm_head`) or LoRA
///   adapters targeting them (e.g., added `<tool_call>` special tokens that never received gradients),
///   as well as dead `lora_B` columns and all-zero LoRA layers.
/// - Alpha/scaling anomalies: missing `lora_alpha` in `adapter_config.json` when `.alpha` tensors
///   are absent, partial or mismatched `.alpha` tensors across layers, and extreme `alpha/r`
///   ratios (`< 0.1` or `> 16.0`) that cause silent under-scaling or merge norm explosion.
/// - DoRA magnitude vector integrity (`use_dora` missing vectors, shape mismatches, non-positive norms).
pub fn audit_adapter_integrity(
    layers: &[LoraLayer],
    all_tensors: &BTreeMap<String, Tensor2D>,
    alphas: &BTreeMap<String, f64>,
    dora_scales: &BTreeMap<String, (String, Tensor2D)>,
    peft_config: Option<&PeftAdapterConfig>,
) -> AdapterIntegrityReport {
    let eps = 1e-7_f64;
    let mut issues: Vec<IntegrityIssue> = Vec::new();
    let mut embedding_entries: Vec<EmbeddingAuditEntry> = Vec::new();

    // Track keys already claimed by matched (lora_A, lora_B) pairs, .alpha scalars, or DoRA scales.
    let mut claimed_keys: BTreeSet<&str> = BTreeSet::new();
    for l in layers {
        claimed_keys.insert(l.a_key.as_str());
        claimed_keys.insert(l.b_key.as_str());
    }
    for (_, (k, _)) in dora_scales {
        claimed_keys.insert(k.as_str());
    }

    // 1. Check for completely dead LoRA layers (||B||_F <= 1e-8 or ||A||_F <= 1e-8)
    let mut dead_layer_names: Vec<&str> = Vec::new();
    for l in layers {
        let a_norm = l.a.frobenius_norm_sq().sqrt();
        let b_norm = l.b.frobenius_norm_sq().sqrt();
        if !a_norm.is_finite() || !b_norm.is_finite() || a_norm <= 1e-8 || b_norm <= 1e-8 {
            dead_layer_names.push(l.base_name.as_str());
        }
    }
    let dead_lora_layers = dead_layer_names.len();
    if !dead_layer_names.is_empty() {
        let sample: Vec<&str> = dead_layer_names.iter().copied().take(3).collect();
        let more = if dead_layer_names.len() > 3 {
            format!(" (+{} more)", dead_layer_names.len() - 3)
        } else {
            String::new()
        };
        let sev = if dead_lora_layers == layers.len() {
            IssueSeverity::Critical
        } else {
            IssueSeverity::Warning
        };
        issues.push(IntegrityIssue {
            code: "DEAD_LORA_LAYER",
            severity: sev,
            subject: format!("{}{}", sample.join(", "), more),
            message: format!(
                "{} of {} LoRA layer(s) have zero/near-zero Frobenius norm (untrained lora_B or frozen gradients)",
                dead_lora_layers,
                layers.len()
            ),
        });
    }

    // 2. Scan dense saved token embedding / unembedding tensors (e.g., modules_to_save)
    for (key, tensor) in all_tensors {
        if key.ends_with(".alpha") || claimed_keys.contains(key.as_str()) {
            continue;
        }
        if tensor.rows < 2 || tensor.cols < 2 {
            continue;
        }
        if classify_token_module(key).is_some() {
            let (zero_rows, trailing_zero) = count_zero_rows(tensor, eps);
            embedding_entries.push(EmbeddingAuditEntry {
                name: key.clone(),
                kind: "dense_tensor",
                vocab_rows: tensor.rows,
                hidden_dim: tensor.cols,
                zero_norm_rows: zero_rows,
                trailing_zero_rows: trailing_zero,
                dead_lora_b_cols: 0,
                dead_lora_a_rows: 0,
            });
            if zero_rows > 0 {
                let detail = if trailing_zero > 0 && trailing_zero < tensor.rows {
                    format!(
                        "{} of {} token rows are zero/near-zero ({} trailing rows {}..{}; added special tokens such as <tool_call> likely received zero SFT gradients)",
                        zero_rows,
                        tensor.rows,
                        trailing_zero,
                        tensor.rows - trailing_zero,
                        tensor.rows
                    )
                } else {
                    format!(
                        "{} of {} token rows are zero/near-zero in saved embedding/unembedding matrix",
                        zero_rows, tensor.rows
                    )
                };
                issues.push(IntegrityIssue {
                    code: "ZERO_NORM_EMBEDDING_ROWS",
                    severity: IssueSeverity::Critical,
                    subject: key.clone(),
                    message: detail,
                });
            }
        }
    }

    // 3. Scan LoRA adapters targeting token embedding / unembedding modules (embed_tokens, wte, lm_head)
    for l in layers {
        let role = classify_token_module(&l.base_name).or_else(|| {
            if l.a_key.contains("lora_embedding_") {
                Some("embedding")
            } else {
                None
            }
        });
        if let Some(module_role) = role {
            let (dead_b_cols, _) = count_zero_cols(&l.b, eps);
            let (dead_a_rows, _) = count_zero_rows(&l.a, eps);

            let vocab_on_a_cols =
                l.a_key.contains("lora_embedding_") || (module_role == "embedding" && l.d_in() > l.d_out());

            let (kind, vocab_rows, hidden_dim, zero_rows, trailing_zero) = if vocab_on_a_cols {
                let (zr, tz) = count_zero_cols(&l.a, eps);
                ("lora_embedding", l.d_in(), l.d_out(), zr, tz)
            } else {
                let (zr, tz) = count_zero_rows(&l.b, eps);
                let k = if module_role == "lm_head" {
                    "lora_lm_head"
                } else {
                    "lora_embedding"
                };
                (k, l.d_out(), l.d_in(), zr, tz)
            };

            embedding_entries.push(EmbeddingAuditEntry {
                name: l.base_name.clone(),
                kind,
                vocab_rows,
                hidden_dim,
                zero_norm_rows: zero_rows,
                trailing_zero_rows: trailing_zero,
                dead_lora_b_cols: dead_b_cols,
                dead_lora_a_rows: dead_a_rows,
            });

            if zero_rows > 0 {
                let detail = if trailing_zero > 0 && trailing_zero < vocab_rows {
                    format!(
                        "{} of {} token updates in {} are zero/near-zero ({} trailing token indices {}..{}; added special tokens such as <tool_call> never received SFT gradients)",
                        zero_rows,
                        vocab_rows,
                        kind,
                        trailing_zero,
                        vocab_rows - trailing_zero,
                        vocab_rows
                    )
                } else {
                    format!(
                        "{} of {} token updates in {} are zero/near-zero (silent untrained token rows)",
                        zero_rows, vocab_rows, kind
                    )
                };
                issues.push(IntegrityIssue {
                    code: "ZERO_NORM_EMBEDDING_ROWS",
                    severity: IssueSeverity::Critical,
                    subject: l.base_name.clone(),
                    message: detail,
                });
            }

            if dead_b_cols > 0 {
                issues.push(IntegrityIssue {
                    code: "DEAD_EMBEDDING_LORA_COLUMNS",
                    severity: IssueSeverity::Warning,
                    subject: l.base_name.clone(),
                    message: format!(
                        "{} of {} lora_B rank columns (and {} lora_A rows) have zero/near-zero norm in token module adapter",
                        dead_b_cols,
                        l.rank(),
                        dead_a_rows
                    ),
                });
            }
        }
    }

    // 4. Alpha & scaling anomaly diagnostics
    let mut alpha_anomaly = false;
    let config_present = peft_config.is_some();
    let config_lora_alpha = peft_config.and_then(|c| c.lora_alpha);
    let use_rslora = peft_config.map_or(false, |c| c.use_rslora);

    if let Some(cfg) = peft_config {
        if cfg.lora_alpha.is_none() && cfg.alpha_pattern.is_empty() && alphas.is_empty() {
            alpha_anomaly = true;
            issues.push(IntegrityIssue {
                code: "MISSING_LORA_ALPHA",
                severity: IssueSeverity::Warning,
                subject: "adapter_config.json".to_string(),
                message: "lora_alpha is missing from adapter_config.json while .alpha scalar tensors are absent; downstream inference engines may mis-scale or reject the adapter".to_string(),
            });
        }
    }

    let matched_alpha_tensors = layers
        .iter()
        .filter(|l| alphas.contains_key(&l.base_name))
        .count();
    if matched_alpha_tensors > 0
        && matched_alpha_tensors < layers.len()
        && config_lora_alpha.is_none()
    {
        alpha_anomaly = true;
        issues.push(IntegrityIssue {
            code: "MISMATCHED_ALPHA_TENSORS",
            severity: IssueSeverity::Warning,
            subject: ".alpha".to_string(),
            message: format!(
                "Partial .alpha scalar coverage across layers: {}/{} LoRA layers carry .alpha tensors ({} missing), causing inconsistent per-layer scaling",
                matched_alpha_tensors,
                layers.len(),
                layers.len() - matched_alpha_tensors
            ),
        });
    }

    let mut min_ratio: Option<f64> = None;
    let mut max_ratio: Option<f64> = None;
    let mut non_pos_layers: Vec<String> = Vec::new();
    let mut under_layers: Vec<(String, f64, f64, usize)> = Vec::new();
    let mut over_layers: Vec<(String, f64, f64, usize)> = Vec::new();
    let mut config_tensor_mismatches: Vec<String> = Vec::new();

    for l in layers {
        let tensor_alpha = alphas.get(&l.base_name).copied();
        let cfg_alpha = peft_config.and_then(|c| lookup_config_alpha(c, &l.base_name));
        if let (Some(ta), Some(ca)) = (tensor_alpha, cfg_alpha) {
            if (ta - ca).abs() > 1e-4 {
                config_tensor_mismatches.push(format!("{} (.alpha={} vs config={})", l.base_name, ta, ca));
            }
        }
        if let Some(alpha) = tensor_alpha.or(cfg_alpha) {
            let r = l.rank().max(1);
            if !alpha.is_finite() || alpha <= 0.0 {
                non_pos_layers.push(format!("{} (alpha={})", l.base_name, alpha));
            } else {
                let ratio = alpha / (r as f64);
                min_ratio = Some(min_ratio.map_or(ratio, |m| m.min(ratio)));
                max_ratio = Some(max_ratio.map_or(ratio, |m| m.max(ratio)));
                if ratio < 0.1 {
                    under_layers.push((l.base_name.clone(), ratio, alpha, r));
                } else if ratio > 16.0 {
                    over_layers.push((l.base_name.clone(), ratio, alpha, r));
                }
            }
        }
    }

    if !config_tensor_mismatches.is_empty() {
        alpha_anomaly = true;
        issues.push(IntegrityIssue {
            code: "MISMATCHED_ALPHA_CONFIG",
            severity: IssueSeverity::Warning,
            subject: "adapter_config.json vs .alpha".to_string(),
            message: format!(
                "{} layer(s) have conflicting scaling between .alpha tensors and adapter_config.json (e.g. {})",
                config_tensor_mismatches.len(),
                config_tensor_mismatches[0]
            ),
        });
    }

    if !non_pos_layers.is_empty() {
        alpha_anomaly = true;
        issues.push(IntegrityIssue {
            code: "NON_POSITIVE_ALPHA",
            severity: IssueSeverity::Critical,
            subject: non_pos_layers[0].clone(),
            message: format!(
                "{} layer(s) specify non-positive or NaN alpha scaling (zeroes out or inverts adapter updates)",
                non_pos_layers.len()
            ),
        });
    }

    if !under_layers.is_empty() {
        alpha_anomaly = true;
        let (ref sample_name, sample_ratio, sample_alpha, sample_r) = under_layers[0];
        issues.push(IntegrityIssue {
            code: "EXTREME_ALPHA_UNDERSCALE",
            severity: IssueSeverity::Warning,
            subject: sample_name.clone(),
            message: format!(
                "{} layer(s) have extreme alpha/r < 0.1 (e.g. alpha={}, r={}, alpha/r={:.4}), causing silent adapter under-scaling during merge or inference",
                under_layers.len(),
                sample_alpha,
                sample_r,
                sample_ratio
            ),
        });
    }

    if !over_layers.is_empty() {
        alpha_anomaly = true;
        let (ref sample_name, sample_ratio, sample_alpha, sample_r) = over_layers[0];
        issues.push(IntegrityIssue {
            code: "EXTREME_ALPHA_OVERSCALE",
            severity: IssueSeverity::Warning,
            subject: sample_name.clone(),
            message: format!(
                "{} layer(s) have extreme alpha/r > 16.0 (e.g. alpha={}, r={}, alpha/r={:.2}), risking weight norm explosion and logit saturation during merge",
                over_layers.len(),
                sample_alpha,
                sample_r,
                sample_ratio
            ),
        });
    }

    if let (Some(min_r), Some(max_r)) = (min_ratio, max_ratio) {
        let explicit_alpha_pattern = peft_config.map_or(false, |c| !c.alpha_pattern.is_empty());
        if !explicit_alpha_pattern && !use_rslora && min_r > 0.0 && (max_r / min_r) > 8.0 {
            alpha_anomaly = true;
            issues.push(IntegrityIssue {
                code: "MISMATCHED_ALPHA_SCALING",
                severity: IssueSeverity::Warning,
                subject: ".alpha".to_string(),
                message: format!(
                    "Wide unpatterned alpha/r spread across layers ({:.3}x to {:.3}x, {:.1}x disparity); verify .alpha tensors were not mixed from incompatible checkpoints",
                    min_r,
                    max_r,
                    max_r / min_r
                ),
            });
        }
    }

    // 5. DoRA magnitude vector health check
    if peft_config.map_or(false, |c| c.use_dora) && dora_scales.is_empty() {
        alpha_anomaly = true;
        issues.push(IntegrityIssue {
            code: "MISSING_DORA_VECTORS",
            severity: IssueSeverity::Warning,
            subject: "adapter_config.json".to_string(),
            message: "adapter_config.json declares use_dora=true, but zero lora_magnitude_vector / dora_scale tensors exist in .safetensors".to_string(),
        });
    }

    let layer_by_name: BTreeMap<&str, &LoraLayer> =
        layers.iter().map(|l| (l.base_name.as_str(), l)).collect();
    for (base_name, (_key, mag_tensor)) in dora_scales {
        if let Some(layer) = layer_by_name.get(base_name.as_str()) {
            if mag_tensor.data.len() != layer.d_out() {
                alpha_anomaly = true;
                issues.push(IntegrityIssue {
                    code: "DORA_SHAPE_MISMATCH",
                    severity: IssueSeverity::Critical,
                    subject: base_name.clone(),
                    message: format!(
                        "DoRA magnitude vector length ({}) does not match lora_B d_out ({})",
                        mag_tensor.data.len(),
                        layer.d_out()
                    ),
                });
            }
        }
        let bad_vals = mag_tensor
            .data
            .iter()
            .filter(|&&v| !v.is_finite() || v <= 1e-8)
            .count();
        if bad_vals > 0 {
            alpha_anomaly = true;
            issues.push(IntegrityIssue {
                code: "INVALID_DORA_MAGNITUDE",
                severity: IssueSeverity::Warning,
                subject: base_name.clone(),
                message: format!(
                    "DoRA magnitude vector contains {} zero, negative, or non-finite entries out of {}",
                    bad_vals,
                    mag_tensor.data.len()
                ),
            });
        }
    }

    let alpha_status = if alpha_anomaly {
        "ANOMALY_DETECTED"
    } else if alphas.is_empty() && config_lora_alpha.is_none() && peft_config.map_or(true, |c| c.alpha_pattern.is_empty()) {
        "IMPLICIT_UNIT_SCALE"
    } else {
        "HEALTHY"
    };

    let overall_status = if issues.is_empty() {
        "HEALTHY"
    } else {
        "ISSUES_DETECTED"
    };

    AdapterIntegrityReport {
        status: overall_status,
        dead_lora_layers,
        embedding_entries,
        alpha_summary: AlphaScalingSummary {
            alpha_tensor_count: alphas.len(),
            config_present,
            config_lora_alpha,
            use_rslora,
            min_alpha_r_ratio: min_ratio,
            max_alpha_r_ratio: max_ratio,
            dora_tensor_count: dora_scales.len(),
            status: alpha_status,
        },
        issues,
    }
}

fn json_escape_audit(s: &str) -> String {
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

fn opt_f64_json(val: Option<f64>) -> String {
    match val {
        Some(v) if v.is_finite() => format!("{:.6}", v),
        _ => "null".to_string(),
    }
}

/// Serialize `AdapterAuditSummary` and `AdapterIntegrityReport` into machine-readable JSON for `loraspectra audit --json`.
pub fn render_audit_json(
    source: &str,
    summary: &AdapterAuditSummary,
    integrity: &AdapterIntegrityReport,
) -> String {
    let mut lines: Vec<String> = Vec::new();
    lines.push("{".to_string());
    lines.push(format!("  \"source\": \"{}\",", json_escape_audit(source)));
    lines.push(format!("  \"layer_count\": {},", summary.layers.len()));
    lines.push(format!("  \"total_nominal_bytes\": {},", summary.total_nominal_bytes));
    lines.push(format!(
        "  \"total_optimal_99_bytes\": {},",
        summary.total_optimal_99_bytes
    ));
    lines.push(format!(
        "  \"compressible_byte_ratio\": {:.6},",
        summary.compressible_byte_ratio
    ));
    lines.push(format!("  \"mean_nominal_rank\": {:.2},", summary.mean_nominal_rank));
    lines.push(format!(
        "  \"mean_effective_rank\": {:.2},",
        summary.mean_effective_rank
    ));
    lines.push(format!("  \"mean_stable_rank\": {:.2},", summary.mean_stable_rank));
    lines.push(format!(
        "  \"mean_rank_at_99_energy\": {:.2},",
        summary.mean_rank_at_99_energy
    ));
    lines.push(format!(
        "  \"mean_rank_marchenko_pastur\": {:.2},",
        summary.mean_rank_mp
    ));
    lines.push(format!("  \"integrity_status\": \"{}\",", integrity.status));
    lines.push(format!("  \"dead_lora_layers\": {},", integrity.dead_lora_layers));
    lines.push("  \"alpha_scaling\": {".to_string());
    lines.push(format!("    \"status\": \"{}\",", integrity.alpha_summary.status));
    lines.push(format!(
        "    \"alpha_tensor_count\": {},",
        integrity.alpha_summary.alpha_tensor_count
    ));
    lines.push(format!(
        "    \"config_present\": {},",
        integrity.alpha_summary.config_present
    ));
    lines.push(format!(
        "    \"config_lora_alpha\": {},",
        opt_f64_json(integrity.alpha_summary.config_lora_alpha)
    ));
    lines.push(format!("    \"use_rslora\": {},", integrity.alpha_summary.use_rslora));
    lines.push(format!(
        "    \"min_alpha_r_ratio\": {},",
        opt_f64_json(integrity.alpha_summary.min_alpha_r_ratio)
    ));
    lines.push(format!(
        "    \"max_alpha_r_ratio\": {},",
        opt_f64_json(integrity.alpha_summary.max_alpha_r_ratio)
    ));
    lines.push(format!(
        "    \"dora_tensor_count\": {}",
        integrity.alpha_summary.dora_tensor_count
    ));
    lines.push("  },".to_string());

    lines.push("  \"embedding_diagnostics\": [".to_string());
    for (idx, e) in integrity.embedding_entries.iter().enumerate() {
        let comma = if idx + 1 < integrity.embedding_entries.len() {
            ","
        } else {
            ""
        };
        lines.push(format!(
            "    {{\"name\": \"{}\", \"kind\": \"{}\", \"vocab_rows\": {}, \"hidden_dim\": {}, \"zero_norm_rows\": {}, \"trailing_zero_rows\": {}, \"dead_lora_b_cols\": {}, \"dead_lora_a_rows\": {}}}{}",
            json_escape_audit(&e.name),
            e.kind,
            e.vocab_rows,
            e.hidden_dim,
            e.zero_norm_rows,
            e.trailing_zero_rows,
            e.dead_lora_b_cols,
            e.dead_lora_a_rows,
            comma
        ));
    }
    lines.push("  ],".to_string());

    lines.push("  \"integrity_issues\": [".to_string());
    for (idx, issue) in integrity.issues.iter().enumerate() {
        let comma = if idx + 1 < integrity.issues.len() {
            ","
        } else {
            ""
        };
        lines.push(format!(
            "    {{\"code\": \"{}\", \"severity\": \"{}\", \"subject\": \"{}\", \"message\": \"{}\"}}{}",
            issue.code,
            issue.severity.as_str(),
            json_escape_audit(&issue.subject),
            json_escape_audit(&issue.message),
            comma
        ));
    }
    lines.push("  ]".to_string());
    lines.push("}".to_string());
    lines.join("\n")
}

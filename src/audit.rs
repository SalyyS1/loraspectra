use std::collections::{BTreeMap, BTreeSet};

use crate::diff::diff_layer;
use crate::linalg::{dot_slice, qr_core_singular_values};
use crate::safetensors::{extract_pissa_init_layers, LoraLayer, PeftAdapterConfig, Tensor2D};

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

#[derive(Debug, Clone, Copy)]
pub struct SvdTruncationBounds {
    pub rank_k: usize,
    pub nominal_rank: usize,
    pub spectral_error: f64,
    pub relative_spectral_error: f64,
    pub frobenius_error: f64,
    pub relative_frobenius_error: f64,
    pub nuclear_error: f64,
    pub relative_nuclear_error: f64,
    pub wedin_spectral_gap: f64,
    pub relative_local_gap: f64,
}

/// Compute exact Eckart-Young-Mirsky spectral (`L2`), Frobenius (`S2`), Mirsky nuclear (`S1`),
/// and Wedin spectral-gap bounds for truncating `singular_values` to rank `k`.
pub fn compute_truncation_bounds(singular_values: &[f64], k: usize) -> SvdTruncationBounds {
    let r = singular_values.len();
    if r == 0 {
        return SvdTruncationBounds {
            rank_k: 0,
            nominal_rank: 0,
            spectral_error: 0.0,
            relative_spectral_error: 0.0,
            frobenius_error: 0.0,
            relative_frobenius_error: 0.0,
            nuclear_error: 0.0,
            relative_nuclear_error: 0.0,
            wedin_spectral_gap: 0.0,
            relative_local_gap: 0.0,
        };
    }

    let keep = k.max(1).min(r);
    let sigma_1 = singular_values[0].max(0.0);
    let total_energy: f64 = singular_values.iter().map(|&s| s * s).sum();
    let frobenius_norm = total_energy.sqrt();
    let nuclear_norm: f64 = singular_values.iter().map(|&s| s.max(0.0)).sum();

    let tail = if keep < r {
        &singular_values[keep..]
    } else {
        &[]
    };
    let spectral_error = tail.first().copied().unwrap_or(0.0).max(0.0);
    let relative_spectral_error = if sigma_1 > 1e-18 {
        spectral_error / sigma_1
    } else {
        0.0
    };

    let frobenius_error = tail.iter().map(|&s| s * s).sum::<f64>().sqrt();
    let relative_frobenius_error = if frobenius_norm > 1e-18 {
        frobenius_error / frobenius_norm
    } else {
        0.0
    };

    let nuclear_error: f64 = tail.iter().map(|&s| s.max(0.0)).sum();
    let relative_nuclear_error = if nuclear_norm > 1e-18 {
        nuclear_error / nuclear_norm
    } else {
        0.0
    };

    let sigma_k = singular_values[keep - 1].max(0.0);
    let gap = (sigma_k - spectral_error).max(0.0);
    let wedin_spectral_gap = if sigma_1 > 1e-18 {
        gap / sigma_1
    } else {
        0.0
    };
    let relative_local_gap = if sigma_k > 1e-18 {
        gap / sigma_k
    } else {
        0.0
    };

    SvdTruncationBounds {
        rank_k: keep,
        nominal_rank: r,
        spectral_error,
        relative_spectral_error,
        frobenius_error,
        relative_frobenius_error,
        nuclear_error,
        relative_nuclear_error,
        wedin_spectral_gap,
        relative_local_gap,
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
    pub truncation_at_99: SvdTruncationBounds,
    pub truncation_at_mp: SvdTruncationBounds,
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
    pub max_mp_relative_spectral_error: f64,
    pub mean_99_relative_nuclear_error: f64,
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
        let zero_bounds = compute_truncation_bounds(&singular_values, 1);
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
            truncation_at_99: zero_bounds,
            truncation_at_mp: zero_bounds,
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
    let truncation_at_99 = compute_truncation_bounds(&singular_values, rank_at_99_energy);
    let truncation_at_mp = compute_truncation_bounds(&singular_values, rank_marchenko_pastur);

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
        truncation_at_99,
        truncation_at_mp,
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

    let max_mp_relative_spectral_error = reports
        .iter()
        .map(|r| r.truncation_at_mp.relative_spectral_error)
        .fold(0.0_f64, f64::max);
    let mean_99_relative_nuclear_error = reports
        .iter()
        .map(|r| r.truncation_at_99.relative_nuclear_error)
        .sum::<f64>()
        / n;

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
        max_mp_relative_spectral_error,
        mean_99_relative_nuclear_error,
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
    pub min_effective_scale: Option<f64>,
    pub max_effective_scale: Option<f64>,
    pub dora_tensor_count: usize,
    pub init_lora_weights: Option<String>,
    pub pissa_init_tensor_pairs: usize,
    pub status: &'static str,
}

/// Per-layer PiSSA (Principal Singular Values and Singular Vectors Adaptation, Meng et al. NeurIPS 2024)
/// residual task-update diagnostics comparing active `(B, A)` against initial principal `(B_init, A_init)`
/// in `O(r^2 * (d_in + d_out))` without dense matrix allocations.
#[derive(Debug, Clone)]
pub struct PissaLayerDiagnostics {
    pub base_name: String,
    pub rank: usize,
    pub init_rank: usize,
    pub principal_norm: f64,
    pub active_norm: f64,
    pub task_delta_norm: f64,
    pub relative_task_update: f64,
    pub principal_subspace_retention: f64,
}

/// Per-layer DoRA (Weight-Decomposed Low-Rank Adaptation, Liu et al. ICML 2024) magnitude vs
/// directional spectral diagnostics computed in `O(r^2 * (d_in + d_out))` without dense matrices.
#[derive(Debug, Clone)]
pub struct DoraLayerDiagnostics {
    pub base_name: String,
    pub d_out: usize,
    pub mag_mean: f64,
    pub mag_std: f64,
    pub mag_min: f64,
    pub mag_max: f64,
    pub mag_cv: f64,
    pub invalid_channels: usize,
    pub directional_row_norm_mean: f64,
    pub dead_directional_rows: usize,
    pub mag_dir_correlation: f64,
}

#[derive(Debug, Clone)]
pub struct AdapterIntegrityReport {
    pub status: &'static str,
    pub dead_lora_layers: usize,
    pub embedding_entries: Vec<EmbeddingAuditEntry>,
    pub alpha_summary: AlphaScalingSummary,
    pub dora_diagnostics: Vec<DoraLayerDiagnostics>,
    pub pissa_diagnostics: Vec<PissaLayerDiagnostics>,
    pub quant_summary: Option<crate::quant::AdapterQuantSummary>,
    pub issues: Vec<IntegrityIssue>,
}

/// Evaluate SVD truncation error bounds (`LS024`: Eckart-Young spectral/Frobenius, Mirsky nuclear,
/// and Wedin singular gap stability) across audited layers at an optional target rank or tolerance.
pub fn audit_truncation_error_bounds(
    layers: &[LayerSpectralReport],
    target_rank: Option<usize>,
    max_error_tol: Option<f64>,
) -> Vec<IntegrityIssue> {
    let mut issues = Vec::new();
    let mut exceeded: Vec<(&LayerSpectralReport, SvdTruncationBounds)> = Vec::new();
    let mut degen_gap: Vec<(&LayerSpectralReport, SvdTruncationBounds)> = Vec::new();

    if target_rank.is_some() || max_error_tol.is_some() {
        let tol = max_error_tol.unwrap_or(0.15).max(0.0);
        for rep in layers {
            if rep.health == LayerHealth::DeadLayer || rep.nominal_rank <= 1 {
                continue;
            }
            let k = target_rank.unwrap_or(rep.rank_at_99_energy).max(1).min(rep.nominal_rank);
            if k >= rep.nominal_rank {
                continue;
            }
            let b = compute_truncation_bounds(&rep.singular_values, k);
            if b.relative_frobenius_error > tol || b.relative_spectral_error > tol {
                exceeded.push((rep, b));
            }
            if b.relative_local_gap < 0.02 && b.relative_spectral_error > 0.08 {
                degen_gap.push((rep, b));
            }
        }
        if !exceeded.is_empty() {
            let tol = max_error_tol.unwrap_or(0.15).max(0.0);
            let &(worst_rep, worst_b) = exceeded
                .iter()
                .max_by(|a, b| {
                    a.1.relative_frobenius_error
                        .max(a.1.relative_spectral_error)
                        .total_cmp(&b.1.relative_frobenius_error.max(b.1.relative_spectral_error))
                })
                .unwrap();
            let worst_val = worst_b
                .relative_frobenius_error
                .max(worst_b.relative_spectral_error);
            let sev = if worst_val > (2.0 * tol).max(0.35) {
                IssueSeverity::Critical
            } else {
                IssueSeverity::Warning
            };
            issues.push(IntegrityIssue {
                code: "SVD_TRUNCATION_BOUND_EXCEEDED",
                severity: sev,
                subject: worst_rep.base_name.clone(),
                message: format!(
                    "{} layer(s) exceed SVD truncation error tolerance {:.1}% at rank k={} (worst {}: rel_spec={:.1}%, rel_frob={:.1}%, rel_nuclear={:.1}%)",
                    exceeded.len(),
                    tol * 100.0,
                    worst_b.rank_k,
                    worst_rep.base_name,
                    worst_b.relative_spectral_error * 100.0,
                    worst_b.relative_frobenius_error * 100.0,
                    worst_b.relative_nuclear_error * 100.0
                ),
            });
        }
    } else {
        for rep in layers {
            if rep.health == LayerHealth::DeadLayer || rep.nominal_rank <= 1 {
                continue;
            }
            if rep.rank_at_99_energy < rep.nominal_rank {
                let b99 = rep.truncation_at_99;
                if b99.relative_local_gap < 0.01 && b99.relative_spectral_error > 0.08 {
                    degen_gap.push((rep, b99));
                }
            }
            if rep.rank_marchenko_pastur < rep.nominal_rank {
                let bmp = rep.truncation_at_mp;
                if bmp.relative_spectral_error > 0.35 {
                    exceeded.push((rep, bmp));
                }
            }
        }
        if !exceeded.is_empty() {
            let &(worst_rep, worst_b) = exceeded
                .iter()
                .max_by(|a, b| {
                    a.1.relative_spectral_error
                        .total_cmp(&b.1.relative_spectral_error)
                })
                .unwrap();
            issues.push(IntegrityIssue {
                code: "SVD_TRUNCATION_BOUND_EXCEEDED",
                severity: IssueSeverity::Warning,
                subject: worst_rep.base_name.clone(),
                message: format!(
                    "{} layer(s) incur >35% Eckart-Young spectral operator error at Marchenko-Pastur cutoff k={} (worst {}: sigma_{{k+1}}/sigma_1={:.1}%, rel_frob={:.1}%)",
                    exceeded.len(),
                    worst_b.rank_k,
                    worst_rep.base_name,
                    worst_b.relative_spectral_error * 100.0,
                    worst_b.relative_frobenius_error * 100.0
                ),
            });
        }
    }

    if !degen_gap.is_empty() {
        let &(worst_rep, worst_b) = degen_gap
            .iter()
            .min_by(|a, b| a.1.relative_local_gap.total_cmp(&b.1.relative_local_gap))
            .unwrap();
        issues.push(IntegrityIssue {
            code: "SVD_DEGENERATE_SPECTRAL_GAP",
            severity: IssueSeverity::Warning,
            subject: worst_rep.base_name.clone(),
            message: format!(
                "{} layer(s) bisect a nearly-degenerate singular doublet at cutoff k={} (e.g. {}: local gap (sigma_k - sigma_{{k+1}})/sigma_k={:.2}%, Wedin gap={:.4}), risking rotational subspace instability",
                degen_gap.len(),
                worst_b.rank_k,
                worst_rep.base_name,
                worst_b.relative_local_gap * 100.0,
                worst_b.wedin_spectral_gap
            ),
        });
    }

    issues
}

/// Compute exact output-channel directional Euclidean norms `||(B * A)_{i,:}||_2` for `i in 0..d_out`
/// via the `r x r` Gram matrix `M_A = A * A^T` in `O(r^2 * (d_in + d_out))` time and `O(r^2)` memory.
pub fn compute_layer_row_directional_norms(layer: &LoraLayer) -> Vec<f64> {
    let r = layer.rank();
    let d_in = layer.d_in();
    let d_out = layer.d_out();
    if r == 0 || d_in == 0 || d_out == 0 {
        return vec![0.0; d_out];
    }

    let mut ma = vec![0.0; r * r];
    for i in 0..r {
        let row_i = &layer.a.data[i * d_in..(i + 1) * d_in];
        for j in i..r {
            let row_j = &layer.a.data[j * d_in..(j + 1) * d_in];
            let dot = dot_slice(row_i, row_j);
            ma[i * r + j] = dot;
            ma[j * r + i] = dot;
        }
    }

    let mut out = Vec::with_capacity(d_out);
    for k in 0..d_out {
        let b_row = &layer.b.data[k * r..(k + 1) * r];
        let mut norm_sq = 0.0;
        for i in 0..r {
            let bi = b_row[i];
            if bi == 0.0 {
                continue;
            }
            let ma_row = &ma[i * r..(i + 1) * r];
            norm_sq += bi * dot_slice(ma_row, b_row);
        }
        out.push(norm_sq.max(0.0).sqrt());
    }
    out
}

pub(crate) fn pearson_correlation(xs: &[f64], ys: &[f64]) -> f64 {
    let n = xs.len().min(ys.len());
    if n < 2 {
        return 0.0;
    }
    let nf = n as f64;
    let mx: f64 = xs[..n].iter().sum::<f64>() / nf;
    let my: f64 = ys[..n].iter().sum::<f64>() / nf;
    let mut cov = 0.0;
    let mut vx = 0.0;
    let mut vy = 0.0;
    for i in 0..n {
        let dx = xs[i] - mx;
        let dy = ys[i] - my;
        cov += dx * dy;
        vx += dx * dx;
        vy += dy * dy;
    }
    let denom = (vx * vy).sqrt();
    if !denom.is_finite() || denom <= 1e-24 {
        0.0
    } else {
        (cov / denom).clamp(-1.0, 1.0)
    }
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
    audit_adapter_integrity_with_pissa(layers, all_tensors, alphas, dora_scales, peft_config, None)
}

/// Extended adapter integrity auditor accepting optional external PiSSA `pissa_init` reference layers
/// (e.g. from a sibling `pissa_init/adapter_model.safetensors` subdirectory) in addition to inline
/// `.lora_A.pissa_init.weight` tensors inside `all_tensors`.
pub fn audit_adapter_integrity_with_pissa(
    layers: &[LoraLayer],
    all_tensors: &BTreeMap<String, Tensor2D>,
    alphas: &BTreeMap<String, f64>,
    dora_scales: &BTreeMap<String, (String, Tensor2D)>,
    peft_config: Option<&PeftAdapterConfig>,
    extra_pissa_init: Option<&BTreeMap<String, LoraLayer>>,
) -> AdapterIntegrityReport {
    let eps = 1e-7_f64;
    let mut issues: Vec<IntegrityIssue> = Vec::new();
    let mut embedding_entries: Vec<EmbeddingAuditEntry> = Vec::new();

    let mut pissa_init_map: BTreeMap<String, LoraLayer> = match extract_pissa_init_layers(all_tensors) {
        Ok(m) => m,
        Err(e) => {
            issues.push(IntegrityIssue {
                code: "PISSA_SHAPE_MISMATCH",
                severity: IssueSeverity::Critical,
                subject: "pissa_init".to_string(),
                message: e,
            });
            BTreeMap::new()
        }
    };
    if let Some(ext_map) = extra_pissa_init {
        for (k, v) in ext_map {
            pissa_init_map.entry(k.clone()).or_insert_with(|| v.clone());
        }
    }

    // Track keys already claimed by matched (lora_A, lora_B) pairs, pissa_init pairs, .alpha scalars, or DoRA scales.
    let mut claimed_keys: BTreeSet<&str> = BTreeSet::new();
    for l in layers {
        claimed_keys.insert(l.a_key.as_str());
        claimed_keys.insert(l.b_key.as_str());
    }
    for pl in pissa_init_map.values() {
        claimed_keys.insert(pl.a_key.as_str());
        claimed_keys.insert(pl.b_key.as_str());
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
        if key.ends_with(".alpha")
            || key.ends_with(".absmax")
            || key.ends_with(".quant_map")
            || key.ends_with(".nested_absmax")
            || key.ends_with(".nested_quant_map")
            || key.ends_with(".nested_offset")
            || key.ends_with(".SCB")
            || key.ends_with(".weight_scale")
            || key.ends_with(".scale")
            || key.contains(".quant_state.bitsandbytes__")
            || claimed_keys.contains(key.as_str())
        {
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
    let mut min_eff_scale: Option<f64> = None;
    let mut max_eff_scale: Option<f64> = None;
    let mut non_pos_layers: Vec<String> = Vec::new();
    let mut under_layers: Vec<(String, f64, f64, usize)> = Vec::new();
    let mut over_layers: Vec<(String, f64, f64, usize)> = Vec::new();
    let mut config_tensor_mismatches: Vec<String> = Vec::new();
    let mut pissa_non_unit_layers: Vec<(String, f64)> = Vec::new();
    let is_pissa_cfg = peft_config.map_or(false, |c| c.is_pissa());

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
                let rf = r as f64;
                let ratio = alpha / rf;
                let eff_scale = if use_rslora { alpha / rf.sqrt() } else { ratio };
                min_ratio = Some(min_ratio.map_or(ratio, |m| m.min(ratio)));
                max_ratio = Some(max_ratio.map_or(ratio, |m| m.max(ratio)));
                min_eff_scale = Some(min_eff_scale.map_or(eff_scale, |m| m.min(eff_scale)));
                max_eff_scale = Some(max_eff_scale.map_or(eff_scale, |m| m.max(eff_scale)));
                if eff_scale < 0.1 {
                    under_layers.push((l.base_name.clone(), eff_scale, alpha, r));
                } else if eff_scale > 16.0 {
                    over_layers.push((l.base_name.clone(), eff_scale, alpha, r));
                }
                if is_pissa_cfg && (eff_scale - 1.0).abs() > 1e-3 {
                    pissa_non_unit_layers.push((l.base_name.clone(), eff_scale));
                }
            }
        }
    }

    if use_rslora && matched_alpha_tensors > 0 && layers.iter().any(|l| l.rank() > 1) {
        alpha_anomaly = true;
        issues.push(IntegrityIssue {
            code: "RSLORA_ALPHA_TENSOR_AMBIGUITY",
            severity: IssueSeverity::Warning,
            subject: ".alpha vs use_rslora".to_string(),
            message: format!(
                "adapter_config.json sets use_rslora=true while {} .alpha scalar tensor(s) exist in .safetensors; standalone loaders ignoring adapter_config.json divide .alpha/r instead of .alpha/sqrt(r)",
                matched_alpha_tensors
            ),
        });
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

    let scale_formula_label = if use_rslora { "alpha/sqrt(r)" } else { "alpha/r" };

    if !under_layers.is_empty() {
        alpha_anomaly = true;
        let (ref sample_name, sample_ratio, sample_alpha, sample_r) = under_layers[0];
        issues.push(IntegrityIssue {
            code: "EXTREME_ALPHA_UNDERSCALE",
            severity: IssueSeverity::Warning,
            subject: sample_name.clone(),
            message: format!(
                "{} layer(s) have extreme {} < 0.1 (e.g. alpha={}, r={}, {}={:.4}), causing silent adapter under-scaling during merge or inference",
                under_layers.len(),
                scale_formula_label,
                sample_alpha,
                sample_r,
                scale_formula_label,
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
                "{} layer(s) have extreme {} > 16.0 (e.g. alpha={}, r={}, {}={:.2}), risking weight norm explosion and logit saturation during merge",
                over_layers.len(),
                scale_formula_label,
                sample_alpha,
                sample_r,
                scale_formula_label,
                sample_ratio
            ),
        });
    }

    if let (Some(min_s), Some(max_s)) = (min_eff_scale, max_eff_scale) {
        let explicit_alpha_pattern = peft_config.map_or(false, |c| !c.alpha_pattern.is_empty());
        if !explicit_alpha_pattern && min_s > 0.0 && (max_s / min_s) > 8.0 {
            alpha_anomaly = true;
            issues.push(IntegrityIssue {
                code: "MISMATCHED_ALPHA_SCALING",
                severity: IssueSeverity::Warning,
                subject: ".alpha".to_string(),
                message: format!(
                    "Wide unpatterned {} spread across layers ({:.3}x to {:.3}x, {:.1}x disparity); verify .alpha tensors were not mixed from incompatible checkpoints",
                    scale_formula_label,
                    min_s,
                    max_s,
                    max_s / min_s
                ),
            });
        }
    }

    // 5. DoRA magnitude vector & directional coupling diagnostics
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

    let matched_dora_count = layers
        .iter()
        .filter(|l| dora_scales.contains_key(&l.base_name))
        .count();
    if matched_dora_count > 0 && matched_dora_count < layers.len() {
        let non_embed_missing: Vec<&str> = layers
            .iter()
            .filter(|l| {
                !dora_scales.contains_key(&l.base_name)
                    && !l.a_key.contains("lora_embedding_")
                    && classify_token_module(&l.base_name) != Some("embedding")
            })
            .map(|l| l.base_name.as_str())
            .collect();
        if !non_embed_missing.is_empty() {
            alpha_anomaly = true;
            issues.push(IntegrityIssue {
                code: "PARTIAL_DORA_COVERAGE",
                severity: IssueSeverity::Warning,
                subject: non_embed_missing[0].to_string(),
                message: format!(
                    "Partial DoRA magnitude vector coverage: {}/{} LoRA layers carry lora_magnitude_vector ({} linear layer(s) missing, e.g. {})",
                    matched_dora_count,
                    layers.len(),
                    non_embed_missing.len(),
                    non_embed_missing[0]
                ),
            });
        }
    }

    let mut dora_diagnostics: Vec<DoraLayerDiagnostics> = Vec::new();
    for (base_name, (_key, mag_tensor)) in dora_scales {
        let mut shape_ok = false;
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
            } else {
                shape_ok = true;
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
                severity: IssueSeverity::Critical,
                subject: base_name.clone(),
                message: format!(
                    "DoRA magnitude vector contains {} zero, negative, or non-finite entries out of {} (zeroes out or corrupts entire output channels)",
                    bad_vals,
                    mag_tensor.data.len()
                ),
            });
        }

        if let Some(layer) = layer_by_name.get(base_name.as_str()) {
            if shape_ok && !mag_tensor.data.is_empty() {
                let d_out = layer.d_out();
                let finite_vals: Vec<f64> = mag_tensor
                    .data
                    .iter()
                    .copied()
                    .filter(|v| v.is_finite() && *v > 1e-8)
                    .collect();
                let (mag_mean, mag_std, mag_min, mag_max, mag_cv) = if !finite_vals.is_empty() {
                    let nf = finite_vals.len() as f64;
                    let mean = finite_vals.iter().sum::<f64>() / nf;
                    let var = finite_vals
                        .iter()
                        .map(|&v| {
                            let d = v - mean;
                            d * d
                        })
                        .sum::<f64>()
                        / nf;
                    let std = var.sqrt();
                    let min_v = finite_vals.iter().copied().fold(f64::INFINITY, f64::min);
                    let max_v = finite_vals
                        .iter()
                        .copied()
                        .fold(f64::NEG_INFINITY, f64::max);
                    let cv = if mean.abs() > 1e-12 {
                        std / mean.abs()
                    } else {
                        0.0
                    };
                    (mean, std, min_v, max_v, cv)
                } else {
                    (0.0, 0.0, 0.0, 0.0, 0.0)
                };

                let row_norms = compute_layer_row_directional_norms(layer);
                let dir_mean = row_norms.iter().sum::<f64>() / (d_out.max(1) as f64);
                let dead_dir_rows = row_norms
                    .iter()
                    .filter(|&&u| !u.is_finite() || u <= 1e-8)
                    .count();
                let corr = if bad_vals == 0 {
                    pearson_correlation(&mag_tensor.data, &row_norms)
                } else {
                    0.0
                };

                if bad_vals == 0 && (mag_cv > 1.5 || (mag_min > 1e-8 && (mag_max / mag_min) > 50.0))
                {
                    alpha_anomaly = true;
                    issues.push(IntegrityIssue {
                        code: "DORA_EXTREME_DISPERSION",
                        severity: IssueSeverity::Warning,
                        subject: base_name.clone(),
                        message: format!(
                            "DoRA magnitude vector exhibits extreme channel dispersion (min={:.4}, max={:.4}, max/min={:.1}x, cv={:.3}), risking output channel saturation",
                            mag_min,
                            mag_max,
                            mag_max / mag_min.max(1e-12),
                            mag_cv
                        ),
                    });
                } else if bad_vals == 0
                    && (dir_mean <= 1e-7 || dead_dir_rows * 2 >= d_out)
                    && mag_cv > 0.05
                {
                    alpha_anomaly = true;
                    issues.push(IntegrityIssue {
                        code: "DORA_DIRECTIONAL_STAGNATION",
                        severity: IssueSeverity::Warning,
                        subject: base_name.clone(),
                        message: format!(
                            "DoRA directional update B*A is stagnant ({}/{} dead output rows, mean row norm {:.2e}) while lora_magnitude_vector diverged (cv={:.3}); low-rank directional subspace failed to train",
                            dead_dir_rows,
                            d_out,
                            dir_mean,
                            mag_cv
                        ),
                    });
                }

                dora_diagnostics.push(DoraLayerDiagnostics {
                    base_name: base_name.clone(),
                    d_out,
                    mag_mean,
                    mag_std,
                    mag_min,
                    mag_max,
                    mag_cv,
                    invalid_channels: bad_vals,
                    directional_row_norm_mean: dir_mean,
                    dead_directional_rows: dead_dir_rows,
                    mag_dir_correlation: corr,
                });
            }
        }
    }

    // 6. PiSSA (Principal Singular Values & Singular Vectors Adaptation) residual-adapter diagnostics
    if is_pissa_cfg && pissa_init_map.is_empty() {
        alpha_anomaly = true;
        issues.push(IntegrityIssue {
            code: "PISSA_UNCONVERTED_RESIDUAL_HAZARD",
            severity: IssueSeverity::Warning,
            subject: "adapter_config.json".to_string(),
            message: format!(
                "init_lora_weights=\"{}\" indicates an unconverted PiSSA residual adapter without bundled pissa_init reference weights; loading onto an unmodified base model W_0 instead of residual W_res doubles the top-r principal singular components",
                peft_config
                    .and_then(|c| c.init_lora_weights.as_deref())
                    .unwrap_or("pissa")
            ),
        });
    }

    if !pissa_non_unit_layers.is_empty() {
        alpha_anomaly = true;
        let (ref sample_l, sample_s) = pissa_non_unit_layers[0];
        issues.push(IntegrityIssue {
            code: "PISSA_NON_UNIT_SCALING",
            severity: IssueSeverity::Warning,
            subject: sample_l.clone(),
            message: format!(
                "{} PiSSA layer(s) have non-unit effective scale ({}={:.4}, e.g. {}); unconverted PiSSA requires unit principal scale (1.0) so W_res + B_init*A_init reconstructs W_0",
                pissa_non_unit_layers.len(),
                scale_formula_label,
                sample_s,
                sample_l
            ),
        });
    }

    let mut pissa_diagnostics: Vec<PissaLayerDiagnostics> = Vec::new();
    if !pissa_init_map.is_empty() {
        let matched_pissa_count = layers
            .iter()
            .filter(|l| pissa_init_map.contains_key(&l.base_name))
            .count();
        if matched_pissa_count > 0 && matched_pissa_count < layers.len() {
            let missing_name = layers
                .iter()
                .find(|l| !pissa_init_map.contains_key(&l.base_name))
                .map(|l| l.base_name.as_str())
                .unwrap_or("unknown");
            alpha_anomaly = true;
            issues.push(IntegrityIssue {
                code: "PISSA_PARTIAL_INIT_COVERAGE",
                severity: IssueSeverity::Warning,
                subject: missing_name.to_string(),
                message: format!(
                    "Partial pissa_init reference coverage: {}/{} LoRA layers have matching pissa_init tensors ({} missing, e.g. {})",
                    matched_pissa_count,
                    layers.len(),
                    layers.len() - matched_pissa_count,
                    missing_name
                ),
            });
        }

        let mut stagnant_pissa_layers: Vec<String> = Vec::new();
        for l in layers {
            if let Some(init_l) = pissa_init_map.get(&l.base_name) {
                if init_l.d_out() != l.d_out() || init_l.d_in() != l.d_in() {
                    alpha_anomaly = true;
                    issues.push(IntegrityIssue {
                        code: "PISSA_SHAPE_MISMATCH",
                        severity: IssueSeverity::Critical,
                        subject: l.base_name.clone(),
                        message: format!(
                            "pissa_init shape ({}x{}) does not match active LoRA layer shape ({}x{})",
                            init_l.d_out(),
                            init_l.d_in(),
                            l.d_out(),
                            l.d_in()
                        ),
                    });
                    continue;
                }
                let rf = l.rank().max(1) as f64;
                let denom = if use_rslora { rf.sqrt() } else { rf };
                let eff_s = alphas
                    .get(&l.base_name)
                    .copied()
                    .or_else(|| peft_config.and_then(|c| lookup_config_alpha(c, &l.base_name)))
                    .filter(|v| v.is_finite() && *v > 0.0)
                    .map(|a| a / denom)
                    .unwrap_or(1.0);
                // Reference pissa_init represents B_init * A_init = U_r S_r V_r^T at unit principal scale (or matching eff_s)
                if let Some(d_rep) = diff_layer(init_l, 1.0, l, eff_s) {
                    if d_rep.relative_drift <= 1e-6 && d_rep.frobenius_norm_a > 1e-8 {
                        stagnant_pissa_layers.push(l.base_name.clone());
                    }
                    pissa_diagnostics.push(PissaLayerDiagnostics {
                        base_name: l.base_name.clone(),
                        rank: l.rank(),
                        init_rank: init_l.rank(),
                        principal_norm: d_rep.frobenius_norm_a,
                        active_norm: d_rep.frobenius_norm_b,
                        task_delta_norm: d_rep.frobenius_delta,
                        relative_task_update: d_rep.relative_drift,
                        principal_subspace_retention: d_rep.subspace_similarity,
                    });
                }
            }
        }

        if !stagnant_pissa_layers.is_empty() {
            alpha_anomaly = true;
            issues.push(IntegrityIssue {
                code: "PISSA_ZERO_TASK_UPDATE",
                severity: IssueSeverity::Warning,
                subject: stagnant_pissa_layers[0].clone(),
                message: format!(
                    "{} PiSSA layer(s) show zero/near-zero net task update relative to pissa_init (||s*B*A - B_init*A_init||_F / ||B_init*A_init||_F <= 1e-6, e.g. {}); fine-tuning failed to update principal components",
                    stagnant_pissa_layers.len(),
                    stagnant_pissa_layers[0]
                ),
            });
        }
    }

    // 7. Quantized checkpoint scale/codebook metadata integrity (LS023)
    issues.extend(crate::quant::audit_quantized_safetensors_metadata(all_tensors));

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
            min_effective_scale: min_eff_scale,
            max_effective_scale: max_eff_scale,
            dora_tensor_count: dora_scales.len(),
            init_lora_weights: peft_config.and_then(|c| c.init_lora_weights.clone()),
            pissa_init_tensor_pairs: pissa_init_map.len(),
            status: alpha_status,
        },
        dora_diagnostics,
        pissa_diagnostics,
        quant_summary: None,
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
    lines.push(format!(
        "  \"max_mp_relative_spectral_error\": {:.6},",
        summary.max_mp_relative_spectral_error
    ));
    lines.push(format!(
        "  \"mean_99_relative_nuclear_error\": {:.6},",
        summary.mean_99_relative_nuclear_error
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
        "    \"min_effective_scale\": {},",
        opt_f64_json(integrity.alpha_summary.min_effective_scale)
    ));
    lines.push(format!(
        "    \"max_effective_scale\": {},",
        opt_f64_json(integrity.alpha_summary.max_effective_scale)
    ));
    lines.push(format!(
        "    \"init_lora_weights\": {},",
        integrity
            .alpha_summary
            .init_lora_weights
            .as_deref()
            .map_or("null".to_string(), |s| format!("\"{}\"", json_escape_audit(s)))
    ));
    lines.push(format!(
        "    \"pissa_init_tensor_pairs\": {},",
        integrity.alpha_summary.pissa_init_tensor_pairs
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

    lines.push("  \"dora_diagnostics\": [".to_string());
    for (idx, d) in integrity.dora_diagnostics.iter().enumerate() {
        let comma = if idx + 1 < integrity.dora_diagnostics.len() {
            ","
        } else {
            ""
        };
        lines.push(format!(
            "    {{\"layer\": \"{}\", \"d_out\": {}, \"mag_mean\": {:.6}, \"mag_std\": {:.6}, \"mag_min\": {:.6}, \"mag_max\": {:.6}, \"mag_cv\": {:.6}, \"invalid_channels\": {}, \"directional_row_norm_mean\": {:.6}, \"dead_directional_rows\": {}, \"mag_dir_correlation\": {:.6}}}{}",
            json_escape_audit(&d.base_name),
            d.d_out,
            d.mag_mean,
            d.mag_std,
            d.mag_min,
            d.mag_max,
            d.mag_cv,
            d.invalid_channels,
            d.directional_row_norm_mean,
            d.dead_directional_rows,
            d.mag_dir_correlation,
            comma
        ));
    }
    lines.push("  ],".to_string());

    lines.push("  \"pissa_diagnostics\": [".to_string());
    for (idx, p) in integrity.pissa_diagnostics.iter().enumerate() {
        let comma = if idx + 1 < integrity.pissa_diagnostics.len() {
            ","
        } else {
            ""
        };
        lines.push(format!(
            "    {{\"layer\": \"{}\", \"rank\": {}, \"init_rank\": {}, \"principal_norm\": {:.6}, \"active_norm\": {:.6}, \"task_delta_norm\": {:.6}, \"relative_task_update\": {:.6}, \"principal_subspace_retention\": {:.6}}}{}",
            json_escape_audit(&p.base_name),
            p.rank,
            p.init_rank,
            p.principal_norm,
            p.active_norm,
            p.task_delta_norm,
            p.relative_task_update,
            p.principal_subspace_retention,
            comma
        ));
    }
    lines.push("  ],".to_string());

    if let Some(ref qs) = integrity.quant_summary {
        lines.push("  \"quantization_diagnostics\": {".to_string());
        lines.push(format!("    \"scheme\": \"{}\",", qs.scheme.as_str()));
        lines.push(format!("    \"block_size\": {},", qs.block_size));
        lines.push(format!("    \"mean_sqnr_db\": {:.4},", qs.mean_sqnr_db));
        lines.push(format!("    \"min_sqnr_db\": {:.4},", qs.min_sqnr_db));
        lines.push(format!(
            "    \"mean_relative_frobenius_error\": {:.6},",
            qs.mean_relative_frobenius_error
        ));
        lines.push(format!(
            "    \"max_relative_frobenius_error\": {:.6},",
            qs.max_relative_frobenius_error
        ));
        lines.push(format!(
            "    \"max_relative_spectral_error\": {:.6},",
            qs.max_relative_spectral_error
        ));
        lines.push(format!(
            "    \"mean_balanced_relative_frobenius_error\": {:.6}",
            qs.mean_balanced_relative_frobenius_error
        ));
        lines.push("  },".to_string());
    } else {
        lines.push("  \"quantization_diagnostics\": null,".to_string());
    }

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

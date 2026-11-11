use crate::linalg::qr_core_singular_values;
use crate::safetensors::LoraLayer;

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

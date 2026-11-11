use std::collections::BTreeMap;

use crate::audit::estimate_marchenko_pastur_rank;
use crate::linalg::{
    multiply_q_cols_by_small, jacobi_svd, matmul, svd_to_lora_factors, thin_qr_columns, transpose,
    SvdDecomposition,
};
use crate::safetensors::{convert_layers_key_format, LoraLayer, Tensor2D};

#[derive(Debug, Clone, Copy)]
pub enum CompressionPolicy {
    /// Truncate every layer to a fixed target rank `k`.
    FixedRank(usize),
    /// Dynamically select the smallest rank `k in [min_rank, max_rank]` per layer
    /// that preserves at least `target_energy` (e.g. `0.99` = 99% of Frobenius norm squared).
    AdaptiveEnergy {
        target_energy: f64,
        min_rank: usize,
        max_rank: usize,
    },
    /// Automatically prune stochastic SGD noise singular values using the Gavish-Donoho (2014)
    /// Marchenko-Pastur optimal hard threshold with zero manual hyperparameter tuning.
    MarchenkoPastur {
        min_rank: usize,
        max_rank: usize,
    },
}

#[derive(Debug, Clone)]
pub struct LayerCompressionMetric {
    pub base_name: String,
    pub original_rank: usize,
    pub compressed_rank: usize,
    pub retained_energy_ratio: f64,
    pub relative_frobenius_error: f64,
    pub relative_spectral_error: f64,
    pub original_bytes: usize,
    pub compressed_bytes: usize,
}

#[derive(Debug, Clone)]
pub struct CompressionResult {
    pub layers: Vec<LoraLayer>,
    pub metrics: Vec<LayerCompressionMetric>,
    pub total_original_bytes: usize,
    pub total_compressed_bytes: usize,
    pub overall_retained_energy: f64,
    pub rank_pattern: BTreeMap<String, usize>,
}

/// Compress a single `LoraLayer` using optimal Eckart-Young QR-Core SVD truncation.
pub fn compress_layer(
    layer: &LoraLayer,
    policy: CompressionPolicy,
) -> (LoraLayer, LayerCompressionMetric) {
    let orig_r = layer.rank();
    let (qb_cols, rb) = thin_qr_columns(&layer.b);
    let at = transpose(&layer.a);
    let (qa_cols, ra) = thin_qr_columns(&at);
    let rat = transpose(&ra);
    let core = matmul(&rb, &rat);
    let core_svd = jacobi_svd(&core);

    let total_energy: f64 = core_svd.sigma.iter().map(|&s| s * s).sum();

    let target_k = match policy {
        CompressionPolicy::FixedRank(k) => k.min(orig_r).max(1),
        CompressionPolicy::AdaptiveEnergy {
            target_energy,
            min_rank,
            max_rank,
        } => {
            if total_energy <= 1e-18 {
                min_rank.min(orig_r).max(1)
            } else {
                let mut cum = 0.0;
                let mut chosen = orig_r;
                for (i, &s) in core_svd.sigma.iter().enumerate() {
                    cum += s * s;
                    if cum / total_energy >= target_energy {
                        chosen = i + 1;
                        break;
                    }
                }
                chosen.max(min_rank).min(max_rank).min(orig_r).max(1)
            }
        }
        CompressionPolicy::MarchenkoPastur { min_rank, max_rank } => {
            let (mp_k, _) =
                estimate_marchenko_pastur_rank(&core_svd.sigma, layer.d_out(), layer.d_in());
            mp_k.max(min_rank).min(max_rank).min(orig_r).max(1)
        }
    };

    let u_trunc = multiply_q_cols_by_small(&qb_cols, layer.d_out(), &core_svd.u, target_k, layer.b.dtype);
    let v_trunc = multiply_q_cols_by_small(&qa_cols, layer.d_in(), &core_svd.v, target_k, layer.a.dtype);

    let full_svd = SvdDecomposition {
        u: u_trunc,
        sigma: core_svd.sigma.clone(),
        v: v_trunc,
    };

    let (b_new, a_new) = svd_to_lora_factors(&full_svd, target_k, layer.a.dtype);

    let kept_energy: f64 = core_svd.sigma.iter().take(target_k).map(|&s| s * s).sum();
    let tail_energy = (total_energy - kept_energy).max(0.0);

    let retained_energy_ratio = if total_energy > 1e-18 {
        (kept_energy / total_energy).min(1.0)
    } else {
        1.0
    };

    let relative_frobenius_error = if total_energy > 1e-18 {
        (tail_energy / total_energy).sqrt()
    } else {
        0.0
    };

    let sigma_1 = core_svd.sigma.first().copied().unwrap_or(0.0);
    let sigma_next = core_svd.sigma.get(target_k).copied().unwrap_or(0.0);
    let relative_spectral_error = if sigma_1 > 1e-18 {
        sigma_next / sigma_1
    } else {
        0.0
    };

    let compressed_layer = LoraLayer {
        base_name: layer.base_name.clone(),
        a_key: layer.a_key.clone(),
        b_key: layer.b_key.clone(),
        a: a_new,
        b: b_new,
    };

    let original_bytes = layer.raw_byte_size();
    let compressed_bytes = compressed_layer.raw_byte_size();

    let metric = LayerCompressionMetric {
        base_name: layer.base_name.clone(),
        original_rank: orig_r,
        compressed_rank: target_k,
        retained_energy_ratio,
        relative_frobenius_error,
        relative_spectral_error,
        original_bytes,
        compressed_bytes,
    };

    (compressed_layer, metric)
}

/// Compress all layers in a LoRA adapter according to `policy`.
pub fn compress_adapter(layers: &[LoraLayer], policy: CompressionPolicy) -> CompressionResult {
    let mut out_layers = Vec::with_capacity(layers.len());
    let mut metrics = Vec::with_capacity(layers.len());
    let mut rank_pattern = BTreeMap::new();

    for layer in layers {
        let (comp, m) = compress_layer(layer, policy);
        rank_pattern.insert(layer.base_name.clone(), m.compressed_rank);
        out_layers.push(comp);
        metrics.push(m);
    }

    let total_original_bytes: usize = metrics.iter().map(|m| m.original_bytes).sum();
    let total_compressed_bytes: usize = metrics.iter().map(|m| m.compressed_bytes).sum();
    let overall_retained_energy = if !metrics.is_empty() {
        metrics.iter().map(|m| m.retained_energy_ratio).sum::<f64>() / (metrics.len() as f64)
    } else {
        1.0
    };

    CompressionResult {
        layers: out_layers,
        metrics,
        total_original_bytes,
        total_compressed_bytes,
        overall_retained_energy,
        rank_pattern,
    }
}

/// Extract a low-rank LoRA adapter directly from the dense weight difference `W_tuned - W_base`
/// between two `.safetensors` models using Eckart-Young SVD truncation or Marchenko-Pastur denoising.
pub fn extract_lora_from_dense_models(
    base_tensors: &BTreeMap<String, Tensor2D>,
    tuned_tensors: &BTreeMap<String, Tensor2D>,
    policy: CompressionPolicy,
    format: &str,
) -> Result<CompressionResult, String> {
    let mut out_layers = Vec::new();
    let mut metrics = Vec::new();
    let mut rank_pattern = BTreeMap::new();

    for (key, w_base) in base_tensors {
        let Some(w_tuned) = tuned_tensors.get(key) else {
            continue;
        };
        if w_base.rows != w_tuned.rows || w_base.cols != w_tuned.cols {
            continue;
        }
        if w_base.rows < 2 || w_base.cols < 2 {
            continue;
        }

        let mut delta = Tensor2D::zeros(w_base.rows, w_base.cols, w_tuned.dtype);
        let mut norm_sq = 0.0;
        for ((d, &tb), &bb) in delta
            .data
            .iter_mut()
            .zip(w_tuned.data.iter())
            .zip(w_base.data.iter())
        {
            let diff = tb - bb;
            *d = diff;
            norm_sq += diff * diff;
        }
        if norm_sq <= 1e-18 {
            continue;
        }

        let svd = jacobi_svd(&delta);
        let max_possible = svd.sigma.len().max(1);
        let total_energy: f64 = svd.sigma.iter().map(|&s| s * s).sum();

        let target_k = match policy {
            CompressionPolicy::FixedRank(k) => k.min(max_possible).max(1),
            CompressionPolicy::AdaptiveEnergy {
                target_energy,
                min_rank,
                max_rank,
            } => {
                let mut cum = 0.0;
                let mut chosen = max_possible;
                for (i, &s) in svd.sigma.iter().enumerate() {
                    cum += s * s;
                    if cum / total_energy >= target_energy {
                        chosen = i + 1;
                        break;
                    }
                }
                chosen.max(min_rank).min(max_rank).min(max_possible).max(1)
            }
            CompressionPolicy::MarchenkoPastur { min_rank, max_rank } => {
                let (mp_k, _) =
                    estimate_marchenko_pastur_rank(&svd.sigma, w_base.rows, w_base.cols);
                mp_k.max(min_rank).min(max_rank).min(max_possible).max(1)
            }
        };

        let (b_new, a_new) = svd_to_lora_factors(&svd, target_k, w_tuned.dtype);
        let base_name = key.strip_suffix(".weight").unwrap_or(key.as_str()).to_string();

        let mut layer = LoraLayer {
            base_name: base_name.clone(),
            a_key: format!("{}.lora_A.weight", base_name),
            b_key: format!("{}.lora_B.weight", base_name),
            a: a_new,
            b: b_new,
        };
        convert_layers_key_format(std::slice::from_mut(&mut layer), format)?;

        let kept_energy: f64 = svd.sigma.iter().take(target_k).map(|&s| s * s).sum();
        let tail_energy = (total_energy - kept_energy).max(0.0);
        let retained_energy_ratio = (kept_energy / total_energy).min(1.0);
        let relative_frobenius_error = (tail_energy / total_energy).sqrt();
        let sigma_1 = svd.sigma.first().copied().unwrap_or(0.0);
        let sigma_next = svd.sigma.get(target_k).copied().unwrap_or(0.0);
        let relative_spectral_error = if sigma_1 > 1e-18 {
            sigma_next / sigma_1
        } else {
            0.0
        };

        let original_bytes = w_base.data.len() * w_tuned.dtype.byte_size();
        let compressed_bytes = layer.raw_byte_size();

        rank_pattern.insert(base_name.clone(), target_k);
        metrics.push(LayerCompressionMetric {
            base_name,
            original_rank: max_possible,
            compressed_rank: target_k,
            retained_energy_ratio,
            relative_frobenius_error,
            relative_spectral_error,
            original_bytes,
            compressed_bytes,
        });
        out_layers.push(layer);
    }

    if out_layers.is_empty() {
        return Err("No modified 2D weight matrices found between base and tuned checkpoints".to_string());
    }

    let total_original_bytes: usize = metrics.iter().map(|m| m.original_bytes).sum();
    let total_compressed_bytes: usize = metrics.iter().map(|m| m.compressed_bytes).sum();
    let overall_retained_energy =
        metrics.iter().map(|m| m.retained_energy_ratio).sum::<f64>() / (metrics.len() as f64);

    Ok(CompressionResult {
        layers: out_layers,
        metrics,
        total_original_bytes,
        total_compressed_bytes,
        overall_retained_energy,
        rank_pattern,
    })
}

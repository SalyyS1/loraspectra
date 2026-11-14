use std::cmp::Ordering;
use std::collections::{BTreeMap, BinaryHeap};

use crate::audit::estimate_marchenko_pastur_rank;
use crate::linalg::{
    jacobi_svd, matmul, multiply_q_cols_by_small, svd_to_lora_factors, thin_qr_columns, transpose,
    SvdDecomposition,
};
use crate::safetensors::{convert_layers_key_format, DType, LoraLayer, Tensor2D};

/// Target specification for global heterogeneous per-layer rank budget allocation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BudgetTarget {
    /// Fit all compressed LoRA tensors within `max_bytes` raw tensor payload bytes.
    MaxBytes(usize),
    /// Fit all compressed LoRA tensors within `ratio * original_bytes` (`0.0 < ratio <= 1.0`).
    Ratio(f64),
    /// Fit all compressed LoRA tensors within the byte budget of a uniform `avg_rank` adapter.
    TargetMeanRank(f64),
}

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
    /// Globally allocate heterogeneous per-layer ranks `k_l in [min_rank, max_rank]` across all
    /// layers to maximize total effective retained spectral energy `sum_l (s_l * sigma_{l,j})^2`
    /// subject to a global byte/parameter budget via marginal spectral energy-per-byte water-filling.
    TargetBudget {
        target: BudgetTarget,
        min_rank: usize,
        max_rank: usize,
    },
}

/// Parse a human-friendly byte size specification such as `"4194304"`, `"512K"`, `"16M"`, `"1.5MiB"`, or `"2G"`.
pub fn parse_byte_budget(spec: &str) -> Result<usize, String> {
    let s = spec.trim();
    if s.is_empty() {
        return Err("Empty byte budget".to_string());
    }
    let split_pos = s
        .find(|c: char| !(c.is_ascii_digit() || c == '.' || c == '+' || c == '-'))
        .unwrap_or(s.len());
    let num_part = s[..split_pos].trim();
    let unit_part = s[split_pos..].trim().to_ascii_lowercase();
    let val: f64 = num_part
        .parse::<f64>()
        .map_err(|_| format!("Invalid numeric value in byte budget '{}'", spec))?;
    if !val.is_finite() || val <= 0.0 {
        return Err(format!("Byte budget must be positive, got '{}'", spec));
    }
    let mult: f64 = match unit_part.as_str() {
        "" | "b" => 1.0,
        "k" | "kb" | "kib" => 1024.0,
        "m" | "mb" | "mib" => 1024.0 * 1024.0,
        "g" | "gb" | "gib" => 1024.0 * 1024.0 * 1024.0,
        other => {
            return Err(format!(
                "Unsupported byte unit '{}' in '{}' (use B, K/KiB, M/MiB, or G/GiB)",
                other, spec
            ))
        }
    };
    let bytes = (val * mult).floor();
    if bytes < 1.0 || bytes > (usize::MAX as f64) {
        return Err(format!("Byte budget out of range: '{}'", spec));
    }
    Ok(bytes as usize)
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
    pub global_retained_energy: f64,
    pub rank_pattern: BTreeMap<String, usize>,
}

#[derive(Debug, Clone)]
struct MarginalStep {
    density: f64,
    step_idx: usize,
    layer_idx: usize,
}

impl PartialEq for MarginalStep {
    fn eq(&self, other: &Self) -> bool {
        self.density.to_bits() == other.density.to_bits()
            && self.step_idx == other.step_idx
            && self.layer_idx == other.layer_idx
    }
}

impl Eq for MarginalStep {}

impl PartialOrd for MarginalStep {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for MarginalStep {
    fn cmp(&self, other: &Self) -> Ordering {
        self.density
            .total_cmp(&other.density)
            .then_with(|| other.step_idx.cmp(&self.step_idx))
            .then_with(|| other.layer_idx.cmp(&self.layer_idx))
    }
}

/// Globally allocate heterogeneous per-layer ranks `k_l in [min_rank, max_rank]` across `L` layers
/// to maximize total weighted spectral energy `sum_l sum_{j < k_l} (scales[l] * sigmas[l][j])^2`
/// subject to `sum_l k_l * step_bytes[l] <= budget_bytes` in `O(L * r * log L)` time.
pub fn allocate_optimal_ranks(
    sigmas: &[Vec<f64>],
    step_bytes: &[usize],
    scales: &[f64],
    budget_bytes: usize,
    min_rank: usize,
    max_rank: usize,
) -> Vec<usize> {
    let n = sigmas.len();
    if n == 0 {
        return Vec::new();
    }

    let mut allocated = Vec::with_capacity(n);
    let mut cap_ranks = Vec::with_capacity(n);
    let mut used_bytes = 0usize;

    for l in 0..n {
        let orig_r = sigmas[l].len().max(1);
        let cap_r = max_rank.max(1).min(orig_r);
        let floor_r = min_rank.max(1).min(cap_r);
        allocated.push(floor_r);
        cap_ranks.push(cap_r);
        used_bytes = used_bytes.saturating_add(floor_r.saturating_mul(step_bytes[l]));
    }

    if used_bytes >= budget_bytes {
        return allocated;
    }

    let mut heap: BinaryHeap<MarginalStep> = BinaryHeap::with_capacity(n);
    for l in 0..n {
        let next_idx = allocated[l];
        if next_idx < cap_ranks[l] {
            let s = scales.get(l).copied().unwrap_or(1.0).abs();
            let sigma = sigmas[l].get(next_idx).copied().unwrap_or(0.0);
            let gain = (s * sigma) * (s * sigma);
            if gain > 1e-30 {
                let cost = step_bytes[l].max(1) as f64;
                heap.push(MarginalStep {
                    density: gain / cost,
                    step_idx: next_idx,
                    layer_idx: l,
                });
            }
        }
    }

    while let Some(top) = heap.pop() {
        let l = top.layer_idx;
        let cost = step_bytes[l];
        if used_bytes.saturating_add(cost) <= budget_bytes {
            used_bytes += cost;
            allocated[l] += 1;
            let next_idx = allocated[l];
            if next_idx < cap_ranks[l] {
                let s = scales.get(l).copied().unwrap_or(1.0).abs();
                let sigma = sigmas[l].get(next_idx).copied().unwrap_or(0.0);
                let gain = (s * sigma) * (s * sigma);
                if gain > 1e-30 {
                    heap.push(MarginalStep {
                        density: gain / (cost.max(1) as f64),
                        step_idx: next_idx,
                        layer_idx: l,
                    });
                }
            }
        }
    }

    allocated
}

fn select_local_layer_rank(
    sigma: &[f64],
    d_out: usize,
    d_in: usize,
    orig_r: usize,
    policy: CompressionPolicy,
) -> usize {
    let total_energy: f64 = sigma.iter().map(|&s| s * s).sum();
    match policy {
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
                for (i, &s) in sigma.iter().enumerate() {
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
            let (mp_k, _) = estimate_marchenko_pastur_rank(sigma, d_out, d_in);
            mp_k.max(min_rank).min(max_rank).min(orig_r).max(1)
        }
        CompressionPolicy::TargetBudget {
            target,
            min_rank,
            max_rank,
        } => {
            let step_bytes = (d_out + d_in).max(1) * 2;
            let orig_bytes = orig_r * step_bytes;
            let budget = match target {
                BudgetTarget::MaxBytes(b) => b,
                BudgetTarget::Ratio(r) => ((orig_bytes as f64) * r.clamp(0.0, 1.0)).floor() as usize,
                BudgetTarget::TargetMeanRank(avg_r) => {
                    (avg_r.max(1.0).min(orig_r as f64) * (step_bytes as f64)).floor() as usize
                }
            };
            let alloc = allocate_optimal_ranks(
                &[sigma.to_vec()],
                &[step_bytes],
                &[1.0],
                budget,
                min_rank,
                max_rank,
            );
            alloc[0]
        }
    }
}

struct LayerQrCore {
    qb_cols: Vec<Vec<f64>>,
    qa_cols: Vec<Vec<f64>>,
    core_svd: SvdDecomposition,
}

fn decompose_layer_qr_core(layer: &LoraLayer) -> LayerQrCore {
    let (qb_cols, rb) = thin_qr_columns(&layer.b);
    let at = transpose(&layer.a);
    let (qa_cols, ra) = thin_qr_columns(&at);
    let rat = transpose(&ra);
    let core = matmul(&rb, &rat);
    let core_svd = jacobi_svd(&core);
    LayerQrCore {
        qb_cols,
        qa_cols,
        core_svd,
    }
}

fn finalize_compressed_layer(
    layer: &LoraLayer,
    decomp: &LayerQrCore,
    target_k: usize,
) -> (LoraLayer, LayerCompressionMetric, f64, f64) {
    let orig_r = layer.rank();
    let k = target_k.min(orig_r).max(1);
    let total_energy: f64 = decomp.core_svd.sigma.iter().map(|&s| s * s).sum();

    let u_trunc = multiply_q_cols_by_small(
        &decomp.qb_cols,
        layer.d_out(),
        &decomp.core_svd.u,
        k,
        layer.b.dtype,
    );
    let v_trunc = multiply_q_cols_by_small(
        &decomp.qa_cols,
        layer.d_in(),
        &decomp.core_svd.v,
        k,
        layer.a.dtype,
    );

    let full_svd = SvdDecomposition {
        u: u_trunc,
        sigma: decomp.core_svd.sigma.clone(),
        v: v_trunc,
    };

    let (b_new, a_new) = svd_to_lora_factors(&full_svd, k, layer.a.dtype);

    let kept_energy: f64 = decomp.core_svd.sigma.iter().take(k).map(|&s| s * s).sum();
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

    let sigma_1 = decomp.core_svd.sigma.first().copied().unwrap_or(0.0);
    let sigma_next = decomp.core_svd.sigma.get(k).copied().unwrap_or(0.0);
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
        compressed_rank: k,
        retained_energy_ratio,
        relative_frobenius_error,
        relative_spectral_error,
        original_bytes,
        compressed_bytes,
    };

    (compressed_layer, metric, total_energy, kept_energy)
}

/// Compress a single `LoraLayer` using optimal Eckart-Young QR-Core SVD truncation.
pub fn compress_layer(
    layer: &LoraLayer,
    policy: CompressionPolicy,
) -> (LoraLayer, LayerCompressionMetric) {
    let decomp = decompose_layer_qr_core(layer);
    let target_k = select_local_layer_rank(
        &decomp.core_svd.sigma,
        layer.d_out(),
        layer.d_in(),
        layer.rank(),
        policy,
    );
    let (comp, metric, _, _) = finalize_compressed_layer(layer, &decomp, target_k);
    (comp, metric)
}

fn layer_rank_step_bytes(layer: &LoraLayer, target_dtype: Option<DType>) -> usize {
    match target_dtype {
        Some(dt) => (layer.d_in() + layer.d_out()) * dt.byte_size(),
        None => layer.d_in() * layer.a.dtype.byte_size() + layer.d_out() * layer.b.dtype.byte_size(),
    }
}

/// Compress all layers in a LoRA adapter with optional per-layer effective scaling weights and output `DType`.
pub fn compress_adapter_with_options(
    layers: &[LoraLayer],
    layer_scales: Option<&[f64]>,
    policy: CompressionPolicy,
    target_dtype: Option<DType>,
) -> CompressionResult {
    let n = layers.len();
    let decomps: Vec<LayerQrCore> = layers.iter().map(decompose_layer_qr_core).collect();
    let default_scales: Vec<f64>;
    let scales: &[f64] = match layer_scales {
        Some(s) if s.len() == n => s,
        _ => {
            default_scales = vec![1.0; n];
            &default_scales
        }
    };

    let target_ranks: Vec<usize> = match policy {
        CompressionPolicy::TargetBudget {
            target,
            min_rank,
            max_rank,
        } => {
            let sigmas: Vec<Vec<f64>> = decomps.iter().map(|d| d.core_svd.sigma.clone()).collect();
            let step_bytes: Vec<usize> = layers
                .iter()
                .map(|l| layer_rank_step_bytes(l, target_dtype))
                .collect();
            let baseline_bytes: usize = layers
                .iter()
                .zip(step_bytes.iter())
                .map(|(l, &sb)| l.rank() * sb)
                .sum();
            let budget_bytes = match target {
                BudgetTarget::MaxBytes(b) => b,
                BudgetTarget::Ratio(ratio) => {
                    ((baseline_bytes as f64) * ratio.clamp(0.0, 1.0)).floor() as usize
                }
                BudgetTarget::TargetMeanRank(avg_r) => layers
                    .iter()
                    .zip(step_bytes.iter())
                    .map(|(l, &sb)| {
                        let eff_r = avg_r.max(1.0).min(l.rank() as f64);
                        eff_r * (sb as f64)
                    })
                    .sum::<f64>()
                    .floor() as usize,
            };
            allocate_optimal_ranks(
                &sigmas,
                &step_bytes,
                scales,
                budget_bytes,
                min_rank,
                max_rank,
            )
        }
        _ => layers
            .iter()
            .zip(decomps.iter())
            .map(|(l, d)| {
                select_local_layer_rank(&d.core_svd.sigma, l.d_out(), l.d_in(), l.rank(), policy)
            })
            .collect(),
    };

    let mut out_layers = Vec::with_capacity(n);
    let mut metrics = Vec::with_capacity(n);
    let mut rank_pattern = BTreeMap::new();
    let mut global_total_e = 0.0;
    let mut global_kept_e = 0.0;

    for ((layer, decomp), (&target_k, &scale)) in layers
        .iter()
        .zip(decomps.iter())
        .zip(target_ranks.iter().zip(scales.iter()))
    {
        let (comp, m, tot_e, kept_e) = finalize_compressed_layer(layer, decomp, target_k);
        let w = scale * scale;
        global_total_e += w * tot_e;
        global_kept_e += w * kept_e;
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
    let global_retained_energy = if global_total_e > 1e-18 {
        (global_kept_e / global_total_e).min(1.0)
    } else {
        1.0
    };

    CompressionResult {
        layers: out_layers,
        metrics,
        total_original_bytes,
        total_compressed_bytes,
        overall_retained_energy,
        global_retained_energy,
        rank_pattern,
    }
}

/// Compress all layers in a LoRA adapter according to `policy`.
pub fn compress_adapter(layers: &[LoraLayer], policy: CompressionPolicy) -> CompressionResult {
    compress_adapter_with_options(layers, None, policy, None)
}

fn strip_model_wrapper_prefix(key: &str) -> &str {
    for prefix in [
        "base_model.model.model.diffusion_model.",
        "base_model.model.language_model.model.",
        "base_model.model.llm.model.",
        "model.diffusion_model.",
        "diffusion_model.",
        "language_model.model.",
        "language_model.",
        "llm.model.",
        "base_model.model.",
        "transformer.",
        "model.",
    ] {
        if let Some(rest) = key.strip_prefix(prefix) {
            return rest;
        }
    }
    key
}

/// Extract a low-rank LoRA adapter directly from the dense weight difference `W_tuned - W_base`
/// between two `.safetensors` models using Eckart-Young SVD truncation, Marchenko-Pastur denoising,
/// or global budget-optimal heterogeneous rank allocation.
/// Automatically normalizes wrapper prefixes (`model.diffusion_model.`, `transformer.`, etc.) so
/// ComfyUI/SRPO and Kohya/Diffusers checkpoints match seamlessly.
pub fn extract_lora_from_dense_models(
    base_tensors: &BTreeMap<String, Tensor2D>,
    tuned_tensors: &BTreeMap<String, Tensor2D>,
    policy: CompressionPolicy,
    format: &str,
) -> Result<CompressionResult, String> {
    struct DenseCandidate {
        base_name: String,
        rows: usize,
        cols: usize,
        dtype: DType,
        svd: SvdDecomposition,
    }

    let mut canonical_tuned: BTreeMap<&str, (&String, &Tensor2D)> = BTreeMap::new();
    for (t_key, t_val) in tuned_tensors {
        canonical_tuned.insert(strip_model_wrapper_prefix(t_key), (t_key, t_val));
    }

    let mut candidates: Vec<DenseCandidate> = Vec::new();

    for (key, w_base) in base_tensors {
        let stripped_key = strip_model_wrapper_prefix(key);
        let (canonical_key_str, w_tuned) = if let Some(exact) = tuned_tensors.get(key) {
            (key.as_str(), exact)
        } else if let Some(&(_t_key, matched)) = canonical_tuned.get(stripped_key) {
            (stripped_key, matched)
        } else {
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
        let base_name = canonical_key_str
            .strip_suffix(".weight")
            .unwrap_or(canonical_key_str)
            .to_string();

        candidates.push(DenseCandidate {
            base_name,
            rows: w_base.rows,
            cols: w_base.cols,
            dtype: w_tuned.dtype,
            svd,
        });
    }

    if candidates.is_empty() {
        return Err(
            "No modified 2D weight matrices found between base and tuned checkpoints".to_string(),
        );
    }

    let target_ranks: Vec<usize> = match policy {
        CompressionPolicy::TargetBudget {
            target,
            min_rank,
            max_rank,
        } => {
            let sigmas: Vec<Vec<f64>> = candidates.iter().map(|c| c.svd.sigma.clone()).collect();
            let step_bytes: Vec<usize> = candidates
                .iter()
                .map(|c| (c.rows + c.cols) * c.dtype.byte_size())
                .collect();
            let dense_bytes: usize = candidates
                .iter()
                .map(|c| c.rows * c.cols * c.dtype.byte_size())
                .sum();
            let scales = vec![1.0; candidates.len()];
            let budget_bytes = match target {
                BudgetTarget::MaxBytes(b) => b,
                BudgetTarget::Ratio(ratio) => {
                    ((dense_bytes as f64) * ratio.clamp(0.0, 1.0)).floor() as usize
                }
                BudgetTarget::TargetMeanRank(avg_r) => candidates
                    .iter()
                    .zip(step_bytes.iter())
                    .map(|(c, &sb)| {
                        let max_r = c.svd.sigma.len().max(1) as f64;
                        avg_r.max(1.0).min(max_r) * (sb as f64)
                    })
                    .sum::<f64>()
                    .floor() as usize,
            };
            allocate_optimal_ranks(&sigmas, &step_bytes, &scales, budget_bytes, min_rank, max_rank)
        }
        _ => candidates
            .iter()
            .map(|c| {
                let max_possible = c.svd.sigma.len().max(1);
                select_local_layer_rank(&c.svd.sigma, c.rows, c.cols, max_possible, policy)
            })
            .collect(),
    };

    let mut out_layers = Vec::with_capacity(candidates.len());
    let mut metrics = Vec::with_capacity(candidates.len());
    let mut rank_pattern = BTreeMap::new();
    let mut global_total_e = 0.0;
    let mut global_kept_e = 0.0;

    for (cand, &target_k) in candidates.into_iter().zip(target_ranks.iter()) {
        let max_possible = cand.svd.sigma.len().max(1);
        let k = target_k.min(max_possible).max(1);
        let total_energy: f64 = cand.svd.sigma.iter().map(|&s| s * s).sum();
        let (b_new, a_new) = svd_to_lora_factors(&cand.svd, k, cand.dtype);

        let mut layer = LoraLayer {
            base_name: cand.base_name.clone(),
            a_key: format!("{}.lora_A.weight", cand.base_name),
            b_key: format!("{}.lora_B.weight", cand.base_name),
            a: a_new,
            b: b_new,
        };
        convert_layers_key_format(std::slice::from_mut(&mut layer), format)?;

        let kept_energy: f64 = cand.svd.sigma.iter().take(k).map(|&s| s * s).sum();
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
        let sigma_1 = cand.svd.sigma.first().copied().unwrap_or(0.0);
        let sigma_next = cand.svd.sigma.get(k).copied().unwrap_or(0.0);
        let relative_spectral_error = if sigma_1 > 1e-18 {
            sigma_next / sigma_1
        } else {
            0.0
        };

        let original_bytes = cand.rows * cand.cols * cand.dtype.byte_size();
        let compressed_bytes = layer.raw_byte_size();
        global_total_e += total_energy;
        global_kept_e += kept_energy;

        rank_pattern.insert(cand.base_name.clone(), k);
        metrics.push(LayerCompressionMetric {
            base_name: cand.base_name,
            original_rank: max_possible,
            compressed_rank: k,
            retained_energy_ratio,
            relative_frobenius_error,
            relative_spectral_error,
            original_bytes,
            compressed_bytes,
        });
        out_layers.push(layer);
    }

    let total_original_bytes: usize = metrics.iter().map(|m| m.original_bytes).sum();
    let total_compressed_bytes: usize = metrics.iter().map(|m| m.compressed_bytes).sum();
    let overall_retained_energy =
        metrics.iter().map(|m| m.retained_energy_ratio).sum::<f64>() / (metrics.len() as f64);
    let global_retained_energy = if global_total_e > 1e-18 {
        (global_kept_e / global_total_e).min(1.0)
    } else {
        1.0
    };

    Ok(CompressionResult {
        layers: out_layers,
        metrics,
        total_original_bytes,
        total_compressed_bytes,
        overall_retained_energy,
        global_retained_energy,
        rank_pattern,
    })
}

use std::collections::{BTreeMap, BTreeSet};

use crate::linalg::{dot_slice, qr_core_singular_values, thin_qr_columns, transpose};
use crate::safetensors::{DType, LoraLayer, PeftAdapterConfig, Tensor2D};

/// Input bundle for a single adapter participating in a multi-adapter `merge-check`.
#[derive(Debug, Clone)]
pub struct AdapterMergeInput {
    pub name: String,
    pub layers: Vec<LoraLayer>,
    pub alphas: BTreeMap<String, f64>,
    pub peft_config: Option<PeftAdapterConfig>,
    pub weight: f64,
}

#[derive(Debug, Clone)]
pub struct IncompatibleShapeFinding {
    pub base_name: String,
    pub adapter_a: String,
    pub shape_a: (usize, usize),
    pub adapter_b: String,
    pub shape_b: (usize, usize),
}

#[derive(Debug, Clone)]
pub struct PairwiseAdapterCollision {
    pub adapter_i: usize,
    pub adapter_j: usize,
    pub shared_layers: usize,
    pub mean_subspace_overlap: f64,
    pub max_subspace_overlap: f64,
    pub mean_frobenius_cosine: f64,
    pub min_frobenius_cosine: f64,
    pub destructive_layers: usize,
    pub inflation_layers: usize,
}

#[derive(Debug, Clone)]
pub struct LayerMergeCheckReport {
    pub base_name: String,
    pub adapter_indices: Vec<usize>,
    pub d_out: usize,
    pub d_in: usize,
    pub ranks: Vec<usize>,
    pub adapter_norms: Vec<f64>,
    pub linear_merged_norm: f64,
    pub orthogonal_expected_norm: f64,
    pub max_single_norm: f64,
    pub orthogonal_inflation_ratio: f64,
    pub peak_inflation_ratio: f64,
    pub merged_spectral_norm: f64,
    pub max_single_spectral_norm: f64,
    pub spectral_inflation_ratio: f64,
    pub subspace_collision_ratio: f64,
    pub max_pair_subspace_overlap: f64,
    pub min_pair_frobenius_cosine: f64,
    pub max_pair_frobenius_cosine: f64,
    pub worst_pair_indices: (usize, usize),
    pub union_rank_99: usize,
    pub status: &'static str,
}

#[derive(Debug, Clone)]
pub struct MultiMergeAdvisor {
    pub risk_level: &'static str,
    pub recommended_mode: &'static str,
    pub recommended_rank: usize,
    pub preserve_norm: bool,
    pub suggested_weights: Vec<f64>,
    pub rationale: String,
    pub cli_command: String,
}

#[derive(Debug, Clone)]
pub struct MultiMergeCheckSummary {
    pub adapter_names: Vec<String>,
    pub input_weights: Vec<f64>,
    pub shared_layers: Vec<LayerMergeCheckReport>,
    pub single_adapter_layers: Vec<(String, usize)>,
    pub incompatible_shapes: Vec<IncompatibleShapeFinding>,
    pub pairwise: Vec<PairwiseAdapterCollision>,
    pub mean_subspace_collision: f64,
    pub max_subspace_collision: f64,
    pub mean_peak_inflation: f64,
    pub max_peak_inflation: f64,
    pub mean_spectral_inflation: f64,
    pub max_spectral_inflation: f64,
    pub mean_orthogonal_inflation: f64,
    pub destructive_layer_count: usize,
    pub inflation_layer_count: usize,
    pub collision_layer_count: usize,
    pub safe_layer_count: usize,
    pub advisor: MultiMergeAdvisor,
}

/// Resolve per-layer inference scaling (`alpha / r` or `alpha / sqrt(r)` for rsLoRA)
/// from `.alpha` scalar tensors or companion `adapter_config.json`.
pub fn resolve_effective_layer_scale(
    layer: &LoraLayer,
    alphas: &BTreeMap<String, f64>,
    peft_config: Option<&PeftAdapterConfig>,
) -> f64 {
    let r = layer.rank().max(1) as f64;
    let use_rslora = peft_config.map(|c| c.use_rslora).unwrap_or(false);
    let denom = if use_rslora { r.sqrt() } else { r };

    if let Some(&alpha) = alphas.get(&layer.base_name) {
        if alpha.is_finite() && alpha > 0.0 {
            return alpha / denom;
        }
    }

    if let Some(cfg) = peft_config {
        let mut matched_alpha: Option<(&str, f64)> = None;
        for (pat, &val) in &cfg.alpha_pattern {
            if layer.base_name.ends_with(pat.as_str()) || layer.base_name.contains(pat.as_str()) {
                match matched_alpha {
                    Some((prev_pat, _)) if prev_pat.len() >= pat.len() => {}
                    _ => matched_alpha = Some((pat.as_str(), val)),
                }
            }
        }
        if let Some((_, val)) = matched_alpha {
            if val.is_finite() && val > 0.0 {
                return val / denom;
            }
        }
        if let Some(global_alpha) = cfg.lora_alpha {
            if global_alpha.is_finite() && global_alpha > 0.0 {
                return global_alpha / denom;
            }
        }
    }

    1.0
}

fn active_orthonormal_columns(m: &Tensor2D) -> Vec<Vec<f64>> {
    let (q_cols, r_mat) = thin_qr_columns(m);
    let mut active = Vec::with_capacity(m.cols);
    for c in 0..m.cols {
        if r_mat.get(c, c).abs() > 1e-12 {
            active.push(q_cols[c].clone());
        }
    }
    active
}

fn basis_overlap_sq(qa: &[Vec<f64>], qb: &[Vec<f64>]) -> f64 {
    let mut sum_sq = 0.0;
    for u in qa {
        for v in qb {
            let d = dot_slice(u, v);
            sum_sq += d * d;
        }
    }
    sum_sq
}

fn energy_rank_99(sigma: &[f64]) -> usize {
    let total: f64 = sigma.iter().map(|&s| s * s).sum();
    if total <= 1e-18 {
        return 1;
    }
    let mut cum = 0.0;
    for (idx, &s) in sigma.iter().enumerate() {
        cum += s * s;
        if cum / total >= 0.99 {
            return idx + 1;
        }
    }
    sigma.len().max(1)
}

/// Low-rank Frobenius inner product `<s_p * B_p * A_p, s_q * B_q * A_q>_F`
/// evaluated via `s_p * s_q * Tr((B_p^T * B_q) * (A_q * A_p^T))` in `O(r_p * r_q * (d_in + d_out))`.
fn low_rank_frobenius_inner(
    lp: &LoraLayer,
    sp: f64,
    lq: &LoraLayer,
    sq: f64,
) -> f64 {
    let rp = lp.rank();
    let rq = lq.rank();
    let d_in = lp.d_in();
    let d_out = lp.d_out();

    // P = A_p * A_q^T (rp x rq)
    let mut p_mat = vec![0.0; rp * rq];
    for i in 0..rp {
        let row_p = &lp.a.data[i * d_in..(i + 1) * d_in];
        for j in 0..rq {
            let row_q = &lq.a.data[j * d_in..(j + 1) * d_in];
            p_mat[i * rq + j] = dot_slice(row_p, row_q);
        }
    }

    // Q = B_p^T * B_q (rp x rq)
    let mut q_mat = vec![0.0; rp * rq];
    for k in 0..d_out {
        let bp_row = &lp.b.data[k * rp..(k + 1) * rp];
        let bq_row = &lq.b.data[k * rq..(k + 1) * rq];
        for i in 0..rp {
            let bpi = bp_row[i];
            if bpi == 0.0 {
                continue;
            }
            let q_out = &mut q_mat[i * rq..(i + 1) * rq];
            for j in 0..rq {
                q_out[j] += bpi * bq_row[j];
            }
        }
    }

    let mut trace = 0.0;
    for idx in 0..(rp * rq) {
        trace += p_mat[idx] * q_mat[idx];
    }
    sp * sq * trace
}

/// Analyze cross-adapter subspace collision, destructive sign cancellation, and Frobenius/spectral
/// norm inflation across `N >= 2` LoRA adapters prior to TIES/DARE/linear/ortho-svd merging.
pub fn analyze_multi_adapter_merge(
    inputs: &[AdapterMergeInput],
) -> Result<MultiMergeCheckSummary, String> {
    let n = inputs.len();
    if n < 2 {
        return Err("merge-check requires at least 2 adapters".to_string());
    }

    let adapter_names: Vec<String> = inputs.iter().map(|a| a.name.clone()).collect();
    let input_weights: Vec<f64> = inputs.iter().map(|a| a.weight).collect();

    let mut layer_maps: Vec<BTreeMap<&str, &LoraLayer>> = Vec::with_capacity(n);
    let mut all_layer_names: BTreeSet<&str> = BTreeSet::new();

    for inp in inputs {
        let mut m = BTreeMap::new();
        for l in &inp.layers {
            m.insert(l.base_name.as_str(), l);
            all_layer_names.insert(l.base_name.as_str());
        }
        layer_maps.push(m);
    }

    // Pairwise accumulators for (i < j)
    let mut pair_accum: BTreeMap<(usize, usize), (usize, f64, f64, f64, f64, usize, usize)> =
        BTreeMap::new();
    for i in 0..n {
        for j in (i + 1)..n {
            pair_accum.insert((i, j), (0, 0.0, 0.0, 0.0, 1.0, 0, 0));
        }
    }

    let mut shared_layers = Vec::new();
    let mut single_adapter_layers = Vec::new();
    let mut incompatible_shapes = Vec::new();

    for base_name in all_layer_names {
        let mut present: Vec<(usize, &LoraLayer, f64)> = Vec::new();
        for (idx, m) in layer_maps.iter().enumerate() {
            if let Some(&layer) = m.get(base_name) {
                let eff_scale = resolve_effective_layer_scale(
                    layer,
                    &inputs[idx].alphas,
                    inputs[idx].peft_config.as_ref(),
                ) * inputs[idx].weight;
                present.push((idx, layer, eff_scale));
            }
        }

        if present.len() <= 1 {
            if let Some(&(only_idx, _, _)) = present.first() {
                single_adapter_layers.push((base_name.to_string(), only_idx));
            }
            continue;
        }

        let (first_idx, first_layer, _) = present[0];
        let d_out = first_layer.d_out();
        let d_in = first_layer.d_in();
        let mut shape_mismatch = false;
        for &(other_idx, other_layer, _) in &present[1..] {
            if other_layer.d_out() != d_out || other_layer.d_in() != d_in {
                incompatible_shapes.push(IncompatibleShapeFinding {
                    base_name: base_name.to_string(),
                    adapter_a: inputs[first_idx].name.clone(),
                    shape_a: (d_out, d_in),
                    adapter_b: inputs[other_idx].name.clone(),
                    shape_b: (other_layer.d_out(), other_layer.d_in()),
                });
                shape_mismatch = true;
            }
        }
        if shape_mismatch {
            continue;
        }

        let m_count = present.len();
        let mut gram = vec![0.0; m_count * m_count];
        for p in 0..m_count {
            let (_, lp, sp) = present[p];
            let diag = low_rank_frobenius_inner(lp, sp, lp, sp).max(0.0);
            gram[p * m_count + p] = diag;
            for q in (p + 1)..m_count {
                let (_, lq, sq) = present[q];
                let cross = low_rank_frobenius_inner(lp, sp, lq, sq);
                gram[p * m_count + q] = cross;
                gram[q * m_count + p] = cross;
            }
        }

        let adapter_norms: Vec<f64> = (0..m_count)
            .map(|p| gram[p * m_count + p].max(0.0).sqrt())
            .collect();
        let sum_diag: f64 = (0..m_count).map(|p| gram[p * m_count + p].max(0.0)).sum();
        let sum_all: f64 = gram.iter().sum::<f64>().max(0.0);

        let linear_merged_norm = sum_all.sqrt();
        let orthogonal_expected_norm = sum_diag.sqrt();
        let max_single_norm = adapter_norms.iter().copied().fold(0.0_f64, f64::max);

        let orthogonal_inflation_ratio = if orthogonal_expected_norm > 1e-12 {
            linear_merged_norm / orthogonal_expected_norm
        } else {
            1.0
        };
        let peak_inflation_ratio = if max_single_norm > 1e-12 {
            linear_merged_norm / max_single_norm
        } else {
            1.0
        };

        // Active input/output orthonormal bases for each participating adapter
        let mut in_bases: Vec<Vec<Vec<f64>>> = Vec::with_capacity(m_count);
        let mut out_bases: Vec<Vec<Vec<f64>>> = Vec::with_capacity(m_count);
        let mut max_single_spectral_norm = 0.0_f64;

        for &(_, layer, scale) in &present {
            let at = transpose(&layer.a);
            in_bases.push(active_orthonormal_columns(&at));
            out_bases.push(active_orthonormal_columns(&layer.b));

            let s1 = qr_core_singular_values(&layer.b, &layer.a)
                .first()
                .copied()
                .unwrap_or(0.0)
                * scale.abs();
            if s1 > max_single_spectral_norm {
                max_single_spectral_norm = s1;
            }
        }

        // Stacked low-rank factors B_cat = [s_1 B_1, ..., s_M B_M] and A_cat = [A_1; ...; A_M]
        let r_sum: usize = present.iter().map(|(_, l, _)| l.rank()).sum();
        let mut b_cat = Tensor2D::zeros(d_out, r_sum, DType::F32);
        let mut a_cat = Tensor2D::zeros(r_sum, d_in, DType::F32);
        let mut col_offset = 0usize;
        for &(_, layer, scale) in &present {
            let r = layer.rank();
            for row in 0..d_out {
                for c in 0..r {
                    b_cat.set(row, col_offset + c, layer.b.get(row, c) * scale);
                }
            }
            for c in 0..r {
                let src_row = &layer.a.data[c * d_in..(c + 1) * d_in];
                let dst_row = &mut a_cat.data[(col_offset + c) * d_in..(col_offset + c + 1) * d_in];
                dst_row.copy_from_slice(src_row);
            }
            col_offset += r;
        }

        let merged_sigma = qr_core_singular_values(&b_cat, &a_cat);
        let merged_spectral_norm = merged_sigma.first().copied().unwrap_or(0.0);
        let spectral_inflation_ratio = if max_single_spectral_norm > 1e-12 {
            merged_spectral_norm / max_single_spectral_norm
        } else {
            1.0
        };
        let union_rank_99 = energy_rank_99(&merged_sigma);

        // Pairwise subspace overlap & Frobenius cosine across all pairs in this layer
        let mut sum_in_overlap = 0.0;
        let mut sum_in_denom = 0.0;
        let mut sum_out_overlap = 0.0;
        let mut sum_out_denom = 0.0;

        let mut max_pair_subspace_overlap = 0.0_f64;
        let mut min_pair_frobenius_cosine = 1.0_f64;
        let mut max_pair_frobenius_cosine = -1.0_f64;
        let mut worst_pair_indices = (present[0].0, present[1].0);
        let mut worst_score = -1.0_f64;

        for p in 0..m_count {
            for q in (p + 1)..m_count {
                let idx_i = present[p].0.min(present[q].0);
                let idx_j = present[p].0.max(present[q].0);

                let ov_in_raw = basis_overlap_sq(&in_bases[p], &in_bases[q]);
                let min_in_k = in_bases[p].len().min(in_bases[q].len());
                let ov_in = if min_in_k > 0 {
                    (ov_in_raw / (min_in_k as f64)).clamp(0.0, 1.0)
                } else {
                    0.0
                };
                sum_in_overlap += ov_in_raw;
                sum_in_denom += min_in_k as f64;

                let ov_out_raw = basis_overlap_sq(&out_bases[p], &out_bases[q]);
                let min_out_k = out_bases[p].len().min(out_bases[q].len());
                let ov_out = if min_out_k > 0 {
                    (ov_out_raw / (min_out_k as f64)).clamp(0.0, 1.0)
                } else {
                    0.0
                };
                sum_out_overlap += ov_out_raw;
                sum_out_denom += min_out_k as f64;

                let pair_overlap = 0.5 * (ov_in + ov_out);
                if pair_overlap > max_pair_subspace_overlap {
                    max_pair_subspace_overlap = pair_overlap;
                }

                let denom_frob = adapter_norms[p] * adapter_norms[q];
                let frob_cos = if denom_frob > 1e-15 {
                    (gram[p * m_count + q] / denom_frob).clamp(-1.0, 1.0)
                } else {
                    0.0
                };
                if frob_cos < min_pair_frobenius_cosine {
                    min_pair_frobenius_cosine = frob_cos;
                }
                if frob_cos > max_pair_frobenius_cosine {
                    max_pair_frobenius_cosine = frob_cos;
                }

                let severity_score = pair_overlap + (-frob_cos).max(0.0);
                if severity_score > worst_score {
                    worst_score = severity_score;
                    worst_pair_indices = (idx_i, idx_j);
                }

                let pair_norm_sq = (gram[p * m_count + p]
                    + gram[q * m_count + q]
                    + 2.0 * gram[p * m_count + q])
                    .max(0.0);
                let pair_peak = adapter_norms[p].max(adapter_norms[q]);
                let pair_peak_infl = if pair_peak > 1e-12 {
                    pair_norm_sq.sqrt() / pair_peak
                } else {
                    1.0
                };

                if let Some(acc) = pair_accum.get_mut(&(idx_i, idx_j)) {
                    acc.0 += 1;
                    acc.1 += pair_overlap;
                    if pair_overlap > acc.2 {
                        acc.2 = pair_overlap;
                    }
                    acc.3 += frob_cos;
                    if frob_cos < acc.4 {
                        acc.4 = frob_cos;
                    }
                    if frob_cos < -0.10 {
                        acc.5 += 1;
                    }
                    if frob_cos > 0.35 && pair_peak_infl > 1.35 {
                        acc.6 += 1;
                    }
                }
            }
        }

        let kappa_in = if sum_in_denom > 0.0 {
            (sum_in_overlap / sum_in_denom).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let kappa_out = if sum_out_denom > 0.0 {
            (sum_out_overlap / sum_out_denom).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let subspace_collision_ratio = 0.5 * (kappa_in + kappa_out);

        let status = if min_pair_frobenius_cosine < -0.10 || orthogonal_inflation_ratio < 0.85 {
            "DESTRUCTIVE_CANCELLATION"
        } else if spectral_inflation_ratio > 1.30 || orthogonal_inflation_ratio > 1.15 {
            "NORM_INFLATION"
        } else if subspace_collision_ratio >= 0.45 || max_pair_subspace_overlap >= 0.55 {
            "HIGH_SUBSPACE_COLLISION"
        } else if subspace_collision_ratio >= 0.20 || max_pair_subspace_overlap >= 0.25 {
            "MODERATE_OVERLAP"
        } else {
            "ORTHOGONAL_SAFE"
        };

        shared_layers.push(LayerMergeCheckReport {
            base_name: base_name.to_string(),
            adapter_indices: present.iter().map(|&(i, _, _)| i).collect(),
            d_out,
            d_in,
            ranks: present.iter().map(|&(_, l, _)| l.rank()).collect(),
            adapter_norms,
            linear_merged_norm,
            orthogonal_expected_norm,
            max_single_norm,
            orthogonal_inflation_ratio,
            peak_inflation_ratio,
            merged_spectral_norm,
            max_single_spectral_norm,
            spectral_inflation_ratio,
            subspace_collision_ratio,
            max_pair_subspace_overlap,
            min_pair_frobenius_cosine,
            max_pair_frobenius_cosine,
            worst_pair_indices,
            union_rank_99,
            status,
        });
    }

    let mut pairwise = Vec::with_capacity(pair_accum.len());
    for ((i, j), (cnt, sum_ov, max_ov, sum_fc, min_fc, destr, infl)) in pair_accum {
        let denom = (cnt.max(1)) as f64;
        pairwise.push(PairwiseAdapterCollision {
            adapter_i: i,
            adapter_j: j,
            shared_layers: cnt,
            mean_subspace_overlap: if cnt > 0 { sum_ov / denom } else { 0.0 },
            max_subspace_overlap: max_ov,
            mean_frobenius_cosine: if cnt > 0 { sum_fc / denom } else { 0.0 },
            min_frobenius_cosine: if cnt > 0 { min_fc } else { 0.0 },
            destructive_layers: destr,
            inflation_layers: infl,
        });
    }

    let s_count = shared_layers.len();
    let s_denom = (s_count.max(1)) as f64;
    let mean_subspace_collision = if s_count > 0 {
        shared_layers
            .iter()
            .map(|l| l.subspace_collision_ratio)
            .sum::<f64>()
            / s_denom
    } else {
        0.0
    };
    let max_subspace_collision = shared_layers
        .iter()
        .map(|l| l.subspace_collision_ratio)
        .fold(0.0_f64, f64::max);

    let mean_peak_inflation = if s_count > 0 {
        shared_layers
            .iter()
            .map(|l| l.peak_inflation_ratio)
            .sum::<f64>()
            / s_denom
    } else {
        1.0
    };
    let max_peak_inflation = shared_layers
        .iter()
        .map(|l| l.peak_inflation_ratio)
        .fold(0.0_f64, f64::max);

    let mean_spectral_inflation = if s_count > 0 {
        shared_layers
            .iter()
            .map(|l| l.spectral_inflation_ratio)
            .sum::<f64>()
            / s_denom
    } else {
        1.0
    };
    let max_spectral_inflation = shared_layers
        .iter()
        .map(|l| l.spectral_inflation_ratio)
        .fold(0.0_f64, f64::max);

    let mean_orthogonal_inflation = if s_count > 0 {
        shared_layers
            .iter()
            .map(|l| l.orthogonal_inflation_ratio)
            .sum::<f64>()
            / s_denom
    } else {
        1.0
    };

    let mut destructive_layer_count = 0usize;
    let mut inflation_layer_count = 0usize;
    let mut collision_layer_count = 0usize;
    let mut safe_layer_count = 0usize;

    for l in &shared_layers {
        match l.status {
            "DESTRUCTIVE_CANCELLATION" => destructive_layer_count += 1,
            "NORM_INFLATION" => inflation_layer_count += 1,
            "HIGH_SUBSPACE_COLLISION" => collision_layer_count += 1,
            _ => safe_layer_count += 1,
        }
    }

    let max_union_rank = shared_layers
        .iter()
        .map(|l| l.union_rank_99)
        .max()
        .unwrap_or_else(|| {
            inputs
                .iter()
                .flat_map(|a| a.layers.iter().map(|l| l.rank()))
                .max()
                .unwrap_or(16)
        });
    let recommended_rank = max_union_rank.clamp(4, 128);

    let advisor = build_advisor(
        &adapter_names,
        &input_weights,
        s_count,
        &incompatible_shapes,
        destructive_layer_count,
        inflation_layer_count,
        collision_layer_count,
        max_peak_inflation,
        max_spectral_inflation,
        mean_subspace_collision,
        recommended_rank,
    );

    Ok(MultiMergeCheckSummary {
        adapter_names,
        input_weights,
        shared_layers,
        single_adapter_layers,
        incompatible_shapes,
        pairwise,
        mean_subspace_collision,
        max_subspace_collision,
        mean_peak_inflation,
        max_peak_inflation,
        mean_spectral_inflation,
        max_spectral_inflation,
        mean_orthogonal_inflation,
        destructive_layer_count,
        inflation_layer_count,
        collision_layer_count,
        safe_layer_count,
        advisor,
    })
}

#[allow(clippy::too_many_arguments)]
fn build_advisor(
    adapter_names: &[String],
    input_weights: &[f64],
    shared_count: usize,
    incompatible_shapes: &[IncompatibleShapeFinding],
    destructive_count: usize,
    inflation_count: usize,
    collision_count: usize,
    max_peak_inflation: f64,
    max_spectral_inflation: f64,
    mean_subspace_collision: f64,
    recommended_rank: usize,
) -> MultiMergeAdvisor {
    let n = adapter_names.len();
    if !incompatible_shapes.is_empty() {
        return MultiMergeAdvisor {
            risk_level: "INCOMPATIBLE",
            recommended_mode: "ortho-svd",
            recommended_rank,
            preserve_norm: true,
            suggested_weights: input_weights.to_vec(),
            rationale: format!(
                "Detected {} layer shape mismatch(es) across adapters; verify all adapters target the same base model architecture before merging.",
                incompatible_shapes.len()
            ),
            cli_command: String::new(),
        };
    }

    let preserve_norm = inflation_count > 0 || max_spectral_inflation > 1.25;

    let (risk_level, recommended_mode) = if destructive_count > 0 {
        ("HIGH", "ties")
    } else if collision_count > 0 || mean_subspace_collision >= 0.30 {
        ("HIGH", "ortho-svd")
    } else if inflation_count > 0 {
        ("MODERATE", "ortho-svd")
    } else if shared_count == 0 || mean_subspace_collision < 0.12 {
        ("LOW", "linear")
    } else {
        ("MODERATE", "ortho-svd")
    };

    // Scale down weights if unnormalized linear/ties merge would inflate spectral operator norm
    let suggested_weights: Vec<f64> = if max_spectral_inflation > 1.30 {
        let scale = 1.0 / max_spectral_inflation;
        input_weights.iter().map(|&w| w * scale).collect()
    } else if recommended_mode == "linear"
        && input_weights.iter().all(|&w| (w - 1.0).abs() < 1e-6)
        && max_spectral_inflation > 1.15
    {
        let inv = 1.0 / (n as f64).sqrt();
        vec![inv; n]
    } else {
        input_weights.to_vec()
    };

    let rationale = if destructive_count > 0 {
        format!(
            "{} shared layer(s) exhibit opposing gradient directions (negative Frobenius cosine); TIES sign election with --preserve-norm prevents destructive task cancellation.",
            destructive_count
        )
    } else if collision_count > 0 || mean_subspace_collision >= 0.30 {
        format!(
            "High multi-adapter subspace overlap (mean {:.1}%, {} colliding layer(s)); Gram-Schmidt orthogonalized SVD (ortho-svd) at union rank {} avoids singular mode interference.",
            mean_subspace_collision * 100.0,
            collision_count,
            recommended_rank
        )
    } else if inflation_count > 0 {
        format!(
            "Peak Frobenius inflation reaches {:.2}x (spectral {:.2}x) across {} layer(s); enable --preserve-norm or scaled weights to prevent post-merge logit saturation.",
            max_peak_inflation, max_spectral_inflation, inflation_count
        )
    } else if shared_count == 0 {
        "Adapters target disjoint module subsets with zero shared-layer overlap; direct linear union is lossless.".to_string()
    } else {
        format!(
            "Adapters span nearly orthogonal subspaces (mean collision {:.1}%, peak norm ratio {:.2}x); linear or ortho-svd merge at rank {} is safe.",
            mean_subspace_collision * 100.0,
            max_peak_inflation,
            recommended_rank
        )
    };

    let inputs_str = adapter_names.join(" ");
    let weights_str: String = suggested_weights
        .iter()
        .map(|w| format!("{:.2}", w))
        .collect::<Vec<_>>()
        .join(":");
    let norm_flag = if preserve_norm { " --preserve-norm" } else { "" };
    let cli_command = format!(
        "loraspectra merge {} merged_adapter.safetensors --mode {} --rank {} --weights {}{}",
        inputs_str, recommended_mode, recommended_rank, weights_str, norm_flag
    );

    MultiMergeAdvisor {
        risk_level,
        recommended_mode,
        recommended_rank,
        preserve_norm,
        suggested_weights,
        rationale,
        cli_command,
    }
}

fn json_escape(s: &str) -> String {
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

pub fn render_merge_check_table(summary: &MultiMergeCheckSummary) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "Multi-Adapter Merge Check ({} adapters, {} shared layers, {} single-adapter layers)\n",
        summary.adapter_names.len(),
        summary.shared_layers.len(),
        summary.single_adapter_layers.len()
    ));
    for (idx, (name, w)) in summary
        .adapter_names
        .iter()
        .zip(summary.input_weights.iter())
        .enumerate()
    {
        out.push_str(&format!("  [{}] {} (weight={:.3})\n", idx, name, w));
    }

    if !summary.incompatible_shapes.is_empty() {
        out.push_str("\nINCOMPATIBLE LAYER SHAPES:\n");
        for inc in &summary.incompatible_shapes {
            out.push_str(&format!(
                "  {} : {} ({}x{}) vs {} ({}x{})\n",
                inc.base_name,
                inc.adapter_a,
                inc.shape_a.0,
                inc.shape_a.1,
                inc.adapter_b,
                inc.shape_b.0,
                inc.shape_b.1
            ));
        }
    }

    if !summary.pairwise.is_empty() {
        out.push_str("\nPAIRWISE ADAPTER MATRIX:\n");
        out.push_str(&format!(
            "{:<12} {:>8} {:>12} {:>12} {:>12} {:>10}\n",
            "PAIR", "SHARED", "MEAN_SUBSP", "MAX_SUBSP", "MIN_FROB", "CONFLICTS"
        ));
        out.push_str(&format!("{}\n", "-".repeat(72)));
        for p in &summary.pairwise {
            out.push_str(&format!(
                "[{}]<->[{}]   {:>8} {:>11.1}% {:>11.1}% {:>+12.4} {:>10}\n",
                p.adapter_i,
                p.adapter_j,
                p.shared_layers,
                p.mean_subspace_overlap * 100.0,
                p.max_subspace_overlap * 100.0,
                p.min_frobenius_cosine,
                p.destructive_layers + p.inflation_layers
            ));
        }
    }

    if !summary.shared_layers.is_empty() {
        out.push_str("\nSHARED LAYER INTERFERENCE & NORM INFLATION:\n");
        out.push_str(&format!(
            "{:<38} {:>6} {:>9} {:>9} {:>9} {:>9} {:>7} {:<24}\n",
            "LAYER", "ADAPT", "SUBSPACE", "PEAK_INF", "SPEC_INF", "MIN_COS", "R_99%", "STATUS"
        ));
        out.push_str(&format!("{}\n", "-".repeat(118)));
        for l in &summary.shared_layers {
            out.push_str(&format!(
                "{:<38} {:>6} {:>8.1}% {:>8.2}x {:>8.2}x {:>+9.4} {:>7} {:<24}\n",
                l.base_name,
                l.adapter_indices.len(),
                l.subspace_collision_ratio * 100.0,
                l.peak_inflation_ratio,
                l.spectral_inflation_ratio,
                l.min_pair_frobenius_cosine,
                l.union_rank_99,
                l.status
            ));
        }
        out.push_str(&format!("{}\n", "-".repeat(118)));
    }

    out.push_str(&format!(
        "Summary: risk={} | destructive={} | norm_inflation={} | subspace_collision={} | safe={}\n",
        summary.advisor.risk_level,
        summary.destructive_layer_count,
        summary.inflation_layer_count,
        summary.collision_layer_count,
        summary.safe_layer_count
    ));
    out.push_str(&format!(
        "Metrics: mean_subspace={:.1}% (max {:.1}%) | max_peak_inflation={:.2}x | max_spectral_inflation={:.2}x\n",
        summary.mean_subspace_collision * 100.0,
        summary.max_subspace_collision * 100.0,
        summary.max_peak_inflation,
        summary.max_spectral_inflation
    ));
    out.push_str(&format!("Rationale: {}\n", summary.advisor.rationale));
    if !summary.advisor.cli_command.is_empty() {
        out.push_str(&format!("Recommended Command: {}\n", summary.advisor.cli_command));
    }
    out
}

pub fn render_merge_check_json(summary: &MultiMergeCheckSummary) -> String {
    let mut out = String::new();
    out.push_str("{\n");

    let names_json: Vec<String> = summary
        .adapter_names
        .iter()
        .map(|s| format!("\"{}\"", json_escape(s)))
        .collect();
    out.push_str(&format!("  \"adapters\": [{}],\n", names_json.join(", ")));

    let weights_json: Vec<String> = summary
        .input_weights
        .iter()
        .map(|w| format!("{:.6}", w))
        .collect();
    out.push_str(&format!(
        "  \"input_weights\": [{}],\n",
        weights_json.join(", ")
    ));
    out.push_str(&format!(
        "  \"shared_layer_count\": {},\n",
        summary.shared_layers.len()
    ));
    out.push_str(&format!(
        "  \"single_adapter_layer_count\": {},\n",
        summary.single_adapter_layers.len()
    ));
    out.push_str(&format!(
        "  \"incompatible_shape_count\": {},\n",
        summary.incompatible_shapes.len()
    ));
    out.push_str(&format!(
        "  \"destructive_layer_count\": {},\n",
        summary.destructive_layer_count
    ));
    out.push_str(&format!(
        "  \"inflation_layer_count\": {},\n",
        summary.inflation_layer_count
    ));
    out.push_str(&format!(
        "  \"collision_layer_count\": {},\n",
        summary.collision_layer_count
    ));
    out.push_str(&format!(
        "  \"safe_layer_count\": {},\n",
        summary.safe_layer_count
    ));
    out.push_str(&format!(
        "  \"mean_subspace_collision\": {:.6},\n",
        summary.mean_subspace_collision
    ));
    out.push_str(&format!(
        "  \"max_subspace_collision\": {:.6},\n",
        summary.max_subspace_collision
    ));
    out.push_str(&format!(
        "  \"mean_peak_inflation\": {:.6},\n",
        summary.mean_peak_inflation
    ));
    out.push_str(&format!(
        "  \"max_peak_inflation\": {:.6},\n",
        summary.max_peak_inflation
    ));
    out.push_str(&format!(
        "  \"mean_spectral_inflation\": {:.6},\n",
        summary.mean_spectral_inflation
    ));
    out.push_str(&format!(
        "  \"max_spectral_inflation\": {:.6},\n",
        summary.max_spectral_inflation
    ));
    out.push_str(&format!(
        "  \"mean_orthogonal_inflation\": {:.6},\n",
        summary.mean_orthogonal_inflation
    ));

    let sug_w: Vec<String> = summary
        .advisor
        .suggested_weights
        .iter()
        .map(|w| format!("{:.4}", w))
        .collect();
    out.push_str("  \"advisor\": {\n");
    out.push_str(&format!(
        "    \"risk_level\": \"{}\",\n",
        summary.advisor.risk_level
    ));
    out.push_str(&format!(
        "    \"recommended_mode\": \"{}\",\n",
        summary.advisor.recommended_mode
    ));
    out.push_str(&format!(
        "    \"recommended_rank\": {},\n",
        summary.advisor.recommended_rank
    ));
    out.push_str(&format!(
        "    \"preserve_norm\": {},\n",
        summary.advisor.preserve_norm
    ));
    out.push_str(&format!(
        "    \"suggested_weights\": [{}],\n",
        sug_w.join(", ")
    ));
    out.push_str(&format!(
        "    \"rationale\": \"{}\",\n",
        json_escape(&summary.advisor.rationale)
    ));
    out.push_str(&format!(
        "    \"cli_command\": \"{}\"\n",
        json_escape(&summary.advisor.cli_command)
    ));
    out.push_str("  },\n");

    out.push_str("  \"pairwise\": [\n");
    for (idx, p) in summary.pairwise.iter().enumerate() {
        let comma = if idx + 1 < summary.pairwise.len() {
            ","
        } else {
            ""
        };
        out.push_str(&format!(
            "    {{\"adapter_i\": {}, \"adapter_j\": {}, \"shared_layers\": {}, \"mean_subspace_overlap\": {:.6}, \"max_subspace_overlap\": {:.6}, \"mean_frobenius_cosine\": {:.6}, \"min_frobenius_cosine\": {:.6}, \"destructive_layers\": {}, \"inflation_layers\": {}}}{}\n",
            p.adapter_i,
            p.adapter_j,
            p.shared_layers,
            p.mean_subspace_overlap,
            p.max_subspace_overlap,
            p.mean_frobenius_cosine,
            p.min_frobenius_cosine,
            p.destructive_layers,
            p.inflation_layers,
            comma
        ));
    }
    out.push_str("  ],\n");

    out.push_str("  \"incompatible_shapes\": [\n");
    for (idx, inc) in summary.incompatible_shapes.iter().enumerate() {
        let comma = if idx + 1 < summary.incompatible_shapes.len() {
            ","
        } else {
            ""
        };
        out.push_str(&format!(
            "    {{\"base_name\": \"{}\", \"adapter_a\": \"{}\", \"shape_a\": [{}, {}], \"adapter_b\": \"{}\", \"shape_b\": [{}, {}]}}{}\n",
            json_escape(&inc.base_name),
            json_escape(&inc.adapter_a),
            inc.shape_a.0,
            inc.shape_a.1,
            json_escape(&inc.adapter_b),
            inc.shape_b.0,
            inc.shape_b.1,
            comma
        ));
    }
    out.push_str("  ],\n");

    out.push_str("  \"layers\": [\n");
    for (idx, l) in summary.shared_layers.iter().enumerate() {
        let comma = if idx + 1 < summary.shared_layers.len() {
            ","
        } else {
            ""
        };
        let idxs: Vec<String> = l.adapter_indices.iter().map(|i| i.to_string()).collect();
        let ranks: Vec<String> = l.ranks.iter().map(|r| r.to_string()).collect();
        let norms: Vec<String> = l.adapter_norms.iter().map(|v| format!("{:.6}", v)).collect();
        out.push_str(&format!(
            "    {{\"base_name\": \"{}\", \"adapter_indices\": [{}], \"d_out\": {}, \"d_in\": {}, \"ranks\": [{}], \"adapter_norms\": [{}], \"linear_merged_norm\": {:.6}, \"orthogonal_expected_norm\": {:.6}, \"max_single_norm\": {:.6}, \"orthogonal_inflation_ratio\": {:.6}, \"peak_inflation_ratio\": {:.6}, \"merged_spectral_norm\": {:.6}, \"max_single_spectral_norm\": {:.6}, \"spectral_inflation_ratio\": {:.6}, \"subspace_collision_ratio\": {:.6}, \"max_pair_subspace_overlap\": {:.6}, \"min_pair_frobenius_cosine\": {:.6}, \"max_pair_frobenius_cosine\": {:.6}, \"worst_pair\": [{}, {}], \"union_rank_99\": {}, \"status\": \"{}\"}}{}\n",
            json_escape(&l.base_name),
            idxs.join(", "),
            l.d_out,
            l.d_in,
            ranks.join(", "),
            norms.join(", "),
            l.linear_merged_norm,
            l.orthogonal_expected_norm,
            l.max_single_norm,
            l.orthogonal_inflation_ratio,
            l.peak_inflation_ratio,
            l.merged_spectral_norm,
            l.max_single_spectral_norm,
            l.spectral_inflation_ratio,
            l.subspace_collision_ratio,
            l.max_pair_subspace_overlap,
            l.min_pair_frobenius_cosine,
            l.max_pair_frobenius_cosine,
            l.worst_pair_indices.0,
            l.worst_pair_indices.1,
            l.union_rank_99,
            l.status,
            comma
        ));
    }
    out.push_str("  ]\n");
    out.push_str("}");
    out
}

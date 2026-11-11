use std::collections::BTreeMap;

use crate::linalg::{
    axpy_slice, dot_slice, jacobi_svd, matmul, qr_core_singular_values, qr_core_svd_truncated,
    scale_slice, svd_to_lora_factors, transpose,
};
use crate::safetensors::{LoraLayer, Tensor2D};

#[derive(Debug, Clone, Copy)]
pub enum MergeStrategy {
    Linear {
        w1: f64,
        w2: f64,
        target_rank: usize,
    },
    Ties {
        w1: f64,
        w2: f64,
        density: f64,
        target_rank: usize,
    },
    OrthoSvd {
        w1: f64,
        w2: f64,
        target_rank: usize,
    },
}

impl MergeStrategy {
    pub fn with_weights(self, w1: f64, w2: f64) -> Self {
        match self {
            MergeStrategy::Linear { target_rank, .. } => MergeStrategy::Linear {
                w1,
                w2,
                target_rank,
            },
            MergeStrategy::Ties {
                density,
                target_rank,
                ..
            } => MergeStrategy::Ties {
                w1,
                w2,
                density,
                target_rank,
            },
            MergeStrategy::OrthoSvd { target_rank, .. } => MergeStrategy::OrthoSvd {
                w1,
                w2,
                target_rank,
            },
        }
    }

    pub fn weights(self) -> (f64, f64) {
        match self {
            MergeStrategy::Linear { w1, w2, .. }
            | MergeStrategy::Ties { w1, w2, .. }
            | MergeStrategy::OrthoSvd { w1, w2, .. } => (w1, w2),
        }
    }
}

/// Per-block weight override rule (e.g. `attn=1.0:0.2` or `mlp=0.2:1.0`) for fine-grained
/// Layer-Block Weighting (LBW) during LoRA merging.
#[derive(Debug, Clone)]
pub struct BlockWeightRule {
    pub pattern: String,
    pub w1: f64,
    pub w2: f64,
}

/// Parse a comma-separated layer-block weight specification such as `"attn=1.0:0.2,mlp=0.2:1.0"`.
pub fn parse_layer_weights(spec: &str) -> Result<Vec<BlockWeightRule>, String> {
    let mut rules = Vec::new();
    for part in spec.split(',') {
        let trimmed = part.trim();
        if trimmed.is_empty() {
            continue;
        }
        let (pat, weights_str) = trimmed
            .split_once('=')
            .ok_or_else(|| format!("Invalid layer-weight rule '{}': expected pattern=w1:w2", trimmed))?;
        let (w1_s, w2_s) = weights_str
            .split_once(':')
            .ok_or_else(|| format!("Invalid weight pair '{}' in '{}': expected w1:w2", weights_str, trimmed))?;
        let w1: f64 = w1_s
            .trim()
            .parse()
            .map_err(|_| format!("Invalid float '{}' in rule '{}'", w1_s, trimmed))?;
        let w2: f64 = w2_s
            .trim()
            .parse()
            .map_err(|_| format!("Invalid float '{}' in rule '{}'", w2_s, trimmed))?;
        rules.push(BlockWeightRule {
            pattern: pat.trim().to_string(),
            w1,
            w2,
        });
    }
    Ok(rules)
}

#[derive(Debug, Clone)]
pub struct LayerInterferenceReport {
    pub base_name: String,
    pub max_left_principal_cos: f64,
    pub max_right_principal_cos: f64,
    pub grassmann_left_overlap: f64,
    pub grassmann_right_overlap: f64,
    pub frobenius_cosine: f64,
    pub conflict_severity: &'static str,
}

/// Analyze principal subspace angles and Frobenius cosine similarity between two LoRA layers
/// in `O(r^2 * (d_out + d_in) + r^3)` time without forming `d_out x d_in` dense matrices.
pub fn analyze_layer_interference(
    layer1: &LoraLayer,
    layer2: &LoraLayer,
) -> LayerInterferenceReport {
    let r1 = layer1.rank();
    let r2 = layer2.rank();
    let svd1 = qr_core_svd_truncated(&layer1.b, &layer1.a, r1);
    let svd2 = qr_core_svd_truncated(&layer2.b, &layer2.a, r2);

    // U1^T * U2 (r1 x r2) and V1^T * V2 (r1 x r2)
    let u1_t = transpose(&svd1.u);
    let u_cross = matmul(&u1_t, &svd2.u);
    let u_cross_svd = jacobi_svd(&u_cross);

    let v1_t = transpose(&svd1.v);
    let v_cross = matmul(&v1_t, &svd2.v);
    let v_cross_svd = jacobi_svd(&v_cross);

    let max_left_principal_cos = u_cross_svd
        .sigma
        .first()
        .copied()
        .unwrap_or(0.0)
        .clamp(0.0, 1.0);
    let max_right_principal_cos = v_cross_svd
        .sigma
        .first()
        .copied()
        .unwrap_or(0.0)
        .clamp(0.0, 1.0);

    let min_r = r1.min(r2).max(1) as f64;
    let grassmann_left_overlap = (u_cross.frobenius_norm_sq() / min_r).clamp(0.0, 1.0);
    let grassmann_right_overlap = (v_cross.frobenius_norm_sq() / min_r).clamp(0.0, 1.0);

    // Exact Frobenius inner product <DeltaW_1, DeltaW_2>_F = sum_{i,j} sigma1_i * sigma2_j * (U1^T U2)_{i,j} * (V1^T V2)_{i,j}
    let mut inner_prod = 0.0;
    for i in 0..r1 {
        let s1 = svd1.sigma[i];
        for j in 0..r2 {
            let s2 = svd2.sigma[j];
            inner_prod += s1 * s2 * u_cross.get(i, j) * v_cross.get(i, j);
        }
    }

    let norm1 = svd1.sigma.iter().map(|&s| s * s).sum::<f64>().sqrt();
    let norm2 = svd2.sigma.iter().map(|&s| s * s).sum::<f64>().sqrt();
    let frobenius_cosine = if norm1 > 1e-15 && norm2 > 1e-15 {
        (inner_prod / (norm1 * norm2)).clamp(-1.0, 1.0)
    } else {
        0.0
    };

    let conflict_severity = if frobenius_cosine < -0.15
        || (max_left_principal_cos > 0.70 && max_right_principal_cos > 0.70)
    {
        "HIGH_INTERFERENCE"
    } else if max_left_principal_cos > 0.40 || max_right_principal_cos > 0.40 {
        "MODERATE_OVERLAP"
    } else {
        "ORTHOGONAL_SAFE"
    };

    LayerInterferenceReport {
        base_name: layer1.base_name.clone(),
        max_left_principal_cos,
        max_right_principal_cos,
        grassmann_left_overlap,
        grassmann_right_overlap,
        frobenius_cosine,
        conflict_severity,
    }
}

/// Concatenate two low-rank representations `[w1*B1 | w2*B2]` and `[A1; A2]` and compress
/// to `target_rank` via QR-Core SVD.
fn combine_and_compress_factors(
    base_name: &str,
    a_key: &str,
    b_key: &str,
    b1: &Tensor2D,
    a1: &Tensor2D,
    b2: &Tensor2D,
    a2: &Tensor2D,
    w1: f64,
    w2: f64,
    target_rank: usize,
) -> LoraLayer {
    let d_out = b1.rows;
    let d_in = a1.cols;
    let r1 = a1.rows;
    let r2 = a2.rows;
    let r_sum = r1 + r2;

    let mut b_cat = Tensor2D::zeros(d_out, r_sum, b1.dtype);
    for row in 0..d_out {
        for c in 0..r1 {
            b_cat.set(row, c, w1 * b1.get(row, c));
        }
        for c in 0..r2 {
            b_cat.set(row, r1 + c, w2 * b2.get(row, c));
        }
    }

    let mut a_cat = Tensor2D::zeros(r_sum, d_in, a1.dtype);
    for r in 0..r1 {
        for col in 0..d_in {
            a_cat.set(r, col, a1.get(r, col));
        }
    }
    for r in 0..r2 {
        for col in 0..d_in {
            a_cat.set(r1 + r, col, a2.get(r, col));
        }
    }

    let keep = target_rank.min(r_sum).max(1);
    let svd = qr_core_svd_truncated(&b_cat, &a_cat, keep);
    let (b_merged, a_merged) = svd_to_lora_factors(&svd, keep, a1.dtype);

    LoraLayer {
        base_name: base_name.to_string(),
        a_key: a_key.to_string(),
        b_key: b_key.to_string(),
        a: a_merged,
        b: b_merged,
    }
}

fn magnitude_prune_tensor(t: &Tensor2D, density: f64) -> Tensor2D {
    let d = density.clamp(0.05, 1.0);
    if d >= 0.999 {
        return t.clone();
    }
    let mut mags: Vec<f64> = t.data.iter().map(|x| x.abs()).collect();
    mags.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let keep_idx = ((mags.len() as f64) * d).floor() as usize;
    let threshold = mags.get(keep_idx.min(mags.len().saturating_sub(1))).copied().unwrap_or(0.0);

    let mut out = t.clone();
    for val in &mut out.data {
        if val.abs() < threshold {
            *val = 0.0;
        }
    }
    out
}

/// Merge two matched `LoraLayer` instances according to `strategy`.
pub fn merge_layer(
    layer1: &LoraLayer,
    layer2: &LoraLayer,
    strategy: MergeStrategy,
) -> LoraLayer {
    match strategy {
        MergeStrategy::Linear {
            w1,
            w2,
            target_rank,
        } => combine_and_compress_factors(
            &layer1.base_name,
            &layer1.a_key,
            &layer1.b_key,
            &layer1.b,
            &layer1.a,
            &layer2.b,
            &layer2.a,
            w1,
            w2,
            target_rank,
        ),
        MergeStrategy::Ties {
            w1,
            w2,
            density,
            target_rank,
        } => {
            let b1_p = magnitude_prune_tensor(&layer1.b, density);
            let a1_p = magnitude_prune_tensor(&layer1.a, density);
            let b2_p = magnitude_prune_tensor(&layer2.b, density);
            let a2_p = magnitude_prune_tensor(&layer2.a, density);
            combine_and_compress_factors(
                &layer1.base_name,
                &layer1.a_key,
                &layer1.b_key,
                &b1_p,
                &a1_p,
                &b2_p,
                &a2_p,
                w1,
                w2,
                target_rank,
            )
        }
        MergeStrategy::OrthoSvd {
            w1,
            w2,
            target_rank,
        } => {
            let r1 = layer1.rank();
            let r2 = layer2.rank();
            let d_out = layer1.d_out();
            let d_in = layer1.d_in();

            let svd1 = qr_core_svd_truncated(&layer1.b, &layer1.a, r1);
            let svd2 = qr_core_svd_truncated(&layer2.b, &layer2.a, r2);

            // Extract columns of U1, V1 and U2, V2
            let mut u1_cols: Vec<Vec<f64>> = vec![vec![0.0; d_out]; r1];
            for row in 0..d_out {
                for c in 0..r1 {
                    u1_cols[c][row] = svd1.u.get(row, c);
                }
            }
            let mut v1_cols: Vec<Vec<f64>> = vec![vec![0.0; d_in]; r1];
            for row in 0..d_in {
                for c in 0..r1 {
                    v1_cols[c][row] = svd1.v.get(row, c);
                }
            }

            let mut u2_cols: Vec<Vec<f64>> = vec![vec![0.0; d_out]; r2];
            for row in 0..d_out {
                for c in 0..r2 {
                    u2_cols[c][row] = svd2.u.get(row, c);
                }
            }
            let mut v2_cols: Vec<Vec<f64>> = vec![vec![0.0; d_in]; r2];
            for row in 0..d_in {
                for c in 0..r2 {
                    v2_cols[c][row] = svd2.v.get(row, c);
                }
            }

            // Identify dominant modes of Adapter 1 (capturing 95% of Adapter 1 spectral energy)
            let e1_total: f64 = svd1.sigma.iter().map(|&s| s * s).sum();
            let mut e1_cum = 0.0;
            let mut dom_k1 = r1;
            if e1_total > 1e-18 {
                for (i, &s) in svd1.sigma.iter().enumerate() {
                    e1_cum += s * s;
                    if e1_cum / e1_total >= 0.95 {
                        dom_k1 = i + 1;
                        break;
                    }
                }
            }

            // Orthogonalize Adapter 2's right-singular input vectors against conflicting dominant
            // right-singular modes of Adapter 1 so Adapter 2 does not destructively interfere
            // with Adapter 1's input subspace while retaining its full orthogonal capacity.
            let mut sigma2_adj = svd2.sigma.clone();
            for j in 0..r2 {
                for i in 0..dom_k1 {
                    let u_dot = dot_slice(&u1_cols[i], &u2_cols[j]);
                    let v_dot = dot_slice(&v1_cols[i], &v2_cols[j]);
                    // If Adapter 2 mode j has destructive sign alignment (u_dot * v_dot < 0)
                    // or high subspace collision with Adapter 1's dominant mode i, remove the
                    // colliding right-subspace component from v2_cols[j].
                    if u_dot * v_dot < -0.02 || v_dot.abs() > 0.25 {
                        axpy_slice(&mut v2_cols[j], -v_dot, &v1_cols[i]);
                    }
                }
                let residual_norm = dot_slice(&v2_cols[j], &v2_cols[j]).sqrt();
                if residual_norm > 1e-12 {
                    // Scale singular value by the orthogonal residual norm and re-normalize vector
                    sigma2_adj[j] *= residual_norm;
                    scale_slice(&mut v2_cols[j], 1.0 / residual_norm);
                } else {
                    sigma2_adj[j] = 0.0;
                    v2_cols[j].fill(0.0);
                }
            }

            // Build balanced low-rank factors for Adapter 1 and De-conflicted Adapter 2
            let (b1_bal, a1_bal) = svd_to_lora_factors(&svd1, r1, layer1.a.dtype);

            let mut b2_ortho = Tensor2D::zeros(d_out, r2, layer2.b.dtype);
            let mut a2_ortho = Tensor2D::zeros(r2, d_in, layer2.a.dtype);
            for j in 0..r2 {
                let s_sqrt = sigma2_adj[j].max(0.0).sqrt();
                for row in 0..d_out {
                    b2_ortho.set(row, j, u2_cols[j][row] * s_sqrt);
                }
                for col in 0..d_in {
                    a2_ortho.set(j, col, s_sqrt * v2_cols[j][col]);
                }
            }

            combine_and_compress_factors(
                &layer1.base_name,
                &layer1.a_key,
                &layer1.b_key,
                &b1_bal,
                &a1_bal,
                &b2_ortho,
                &a2_ortho,
                w1,
                w2,
                target_rank,
            )
        }
    }
}

fn scale_single_layer(layer: &LoraLayer, weight: f64) -> LoraLayer {
    let mut out = layer.clone();
    let mag_sqrt = weight.abs().sqrt();
    let sign = if weight < 0.0 { -1.0 } else { 1.0 };
    for v in &mut out.b.data {
        *v *= sign * mag_sqrt;
    }
    for v in &mut out.a.data {
        *v *= mag_sqrt;
    }
    out
}

fn bound_layer_spectral_norm(merged: &mut LoraLayer, l1: &LoraLayer, l2: &LoraLayer) {
    let s1 = qr_core_singular_values(&l1.b, &l1.a)
        .first()
        .copied()
        .unwrap_or(0.0);
    let s2 = qr_core_singular_values(&l2.b, &l2.a)
        .first()
        .copied()
        .unwrap_or(0.0);
    let max_source_sigma = s1.max(s2);
    if max_source_sigma <= 1e-12 {
        return;
    }
    let s_merged = qr_core_singular_values(&merged.b, &merged.a)
        .first()
        .copied()
        .unwrap_or(0.0);
    if s_merged > max_source_sigma {
        let ratio_sqrt = (max_source_sigma / s_merged).sqrt();
        for v in &mut merged.b.data {
            *v *= ratio_sqrt;
        }
        for v in &mut merged.a.data {
            *v *= ratio_sqrt;
        }
    }
}

pub fn merge_adapters(
    layers1: &[LoraLayer],
    layers2: &[LoraLayer],
    strategy: MergeStrategy,
) -> Result<Vec<LoraLayer>, String> {
    merge_adapters_with_options(layers1, layers2, strategy, &[], false)
}

pub fn merge_adapters_with_block_weights(
    layers1: &[LoraLayer],
    layers2: &[LoraLayer],
    base_strategy: MergeStrategy,
    rules: &[BlockWeightRule],
) -> Result<Vec<LoraLayer>, String> {
    merge_adapters_with_options(layers1, layers2, base_strategy, rules, false)
}

pub fn merge_adapters_with_options(
    layers1: &[LoraLayer],
    layers2: &[LoraLayer],
    base_strategy: MergeStrategy,
    rules: &[BlockWeightRule],
    preserve_spectral_norm: bool,
) -> Result<Vec<LoraLayer>, String> {
    let mut map1: BTreeMap<&str, &LoraLayer> = BTreeMap::new();
    let mut map2: BTreeMap<&str, &LoraLayer> = BTreeMap::new();
    let mut all_names: Vec<&str> = Vec::new();

    for l1 in layers1 {
        map1.insert(l1.base_name.as_str(), l1);
        all_names.push(l1.base_name.as_str());
    }
    for l2 in layers2 {
        map2.insert(l2.base_name.as_str(), l2);
        if !map1.contains_key(l2.base_name.as_str()) {
            all_names.push(l2.base_name.as_str());
        }
    }

    let mut merged = Vec::with_capacity(all_names.len());
    for name in all_names {
        let mut layer_strategy = base_strategy;
        for rule in rules {
            if name.contains(&rule.pattern) {
                layer_strategy = base_strategy.with_weights(rule.w1, rule.w2);
            }
        }
        let (w1, w2) = layer_strategy.weights();

        match (map1.get(name), map2.get(name)) {
            (Some(l1), Some(l2)) => {
                if l1.d_out() != l2.d_out() || l1.d_in() != l2.d_in() {
                    return Err(format!(
                        "Incompatible layer pair: {} ({}, {}) vs {} ({}, {})",
                        l1.base_name,
                        l1.d_out(),
                        l1.d_in(),
                        l2.base_name,
                        l2.d_out(),
                        l2.d_in()
                    ));
                }
                let mut m = merge_layer(l1, l2, layer_strategy);
                if preserve_spectral_norm {
                    bound_layer_spectral_norm(&mut m, l1, l2);
                }
                merged.push(m);
            }
            (Some(l1), None) => {
                merged.push(scale_single_layer(l1, w1));
            }
            (None, Some(l2)) => {
                merged.push(scale_single_layer(l2, w2));
            }
            (None, None) => unreachable!(),
        }
    }
    Ok(merged)
}

/// Jointly merge `N >= 2` LoRA adapters with per-adapter weights `[w_1, w_2, ..., w_N]`.
///
/// Intermediate steps retain lossless rank (`r_1 + ... + r_m`) so that each subsequent adapter
/// is orthogonalized against the exact joint dominant singular subspace of all preceding adapters
/// before a single final Eckart-Young QR-Core SVD compression to `target_rank`.
pub fn merge_multiple_adapters(
    adapters: &[Vec<LoraLayer>],
    weights: &[f64],
    mode: &str,
    target_rank: usize,
    rules: &[BlockWeightRule],
    preserve_spectral_norm: bool,
) -> Result<Vec<LoraLayer>, String> {
    if adapters.len() < 2 {
        return Err("merge_multiple_adapters requires at least 2 adapters".to_string());
    }
    if adapters.len() != weights.len() {
        return Err(format!(
            "Adapter count ({}) does not match weight count ({})",
            adapters.len(),
            weights.len()
        ));
    }

    // Track max singular value per layer across all N source adapters for --preserve-norm
    let mut max_source_sigma: BTreeMap<String, f64> = BTreeMap::new();
    if preserve_spectral_norm {
        for adapter in adapters {
            for layer in adapter {
                let s1 = qr_core_singular_values(&layer.b, &layer.a)
                    .first()
                    .copied()
                    .unwrap_or(0.0);
                let entry = max_source_sigma.entry(layer.base_name.clone()).or_insert(0.0);
                if s1 > *entry {
                    *entry = s1;
                }
            }
        }
    }

    let make_strategy = |w1: f64, w2: f64, rank: usize| -> MergeStrategy {
        match mode {
            "linear" => MergeStrategy::Linear {
                w1,
                w2,
                target_rank: rank,
            },
            "ties" => MergeStrategy::Ties {
                w1,
                w2,
                density: 0.6,
                target_rank: rank,
            },
            _ => MergeStrategy::OrthoSvd {
                w1,
                w2,
                target_rank: rank,
            },
        }
    };

    let n = adapters.len();
    let mut acc: Vec<LoraLayer> = Vec::new();

    for idx in 1..n {
        let is_last = idx == n - 1;
        let left = if idx == 1 { &adapters[0] } else { &acc };
        let right = &adapters[idx];
        let w_left = if idx == 1 { weights[0] } else { 1.0 };
        let w_right = weights[idx];

        let max_r_left = left.iter().map(|l| l.rank()).max().unwrap_or(target_rank);
        let max_r_right = right.iter().map(|l| l.rank()).max().unwrap_or(target_rank);
        let step_rank = if is_last {
            target_rank
        } else {
            (max_r_left + max_r_right).min(256)
        };

        let strategy = make_strategy(w_left, w_right, step_rank);
        let step_rules = if idx == 1 { rules } else { &[] };
        acc = merge_adapters_with_options(left, right, strategy, step_rules, false)?;
    }

    if preserve_spectral_norm {
        for layer in &mut acc {
            if let Some(&bound) = max_source_sigma.get(&layer.base_name) {
                if bound > 1e-12 {
                    let s_merged = qr_core_singular_values(&layer.b, &layer.a)
                        .first()
                        .copied()
                        .unwrap_or(0.0);
                    if s_merged > bound {
                        let ratio_sqrt = (bound / s_merged).sqrt();
                        for v in &mut layer.b.data {
                            *v *= ratio_sqrt;
                        }
                        for v in &mut layer.a.data {
                            *v *= ratio_sqrt;
                        }
                    }
                }
            }
        }
    }

    Ok(acc)
}

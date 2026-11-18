//! Cross-Checkpoint Orthogonal Subspace Projector (O-LoRA, Wang et al., EMNLP 2023;
//! InfLoRA, Liang & Li, CVPR 2024; Nullspace Task Arithmetic) and SARIF rules (`LS041`, `LS042`).
//!
//! Projects a candidate LoRA adapter `DeltaW_c = s_c * B_c * A_c` onto the orthogonal complement
//! of one or more protected anchor checkpoints `{DeltaW_1, ..., DeltaW_M}` in pure low-rank time
//! `O((M * r_a + r_c)^2 * (d_in + d_out))` without allocating dense `d_out x d_in` matrices.
//!
//! Under exact nullspace projection (`strength = 1.0`), the linear operator `P_perp` is self-adjoint
//! and idempotent in Frobenius inner-product space (`Right`: `X * (I - Q_V Q_V^T)`,
//! `Left`: `(I - Q_U Q_U^T) * X`, `Bilateral`: `(I - Q_U Q_U^T) * X * (I - Q_V Q_V^T)`),
//! guaranteeing the Pythagorean energy identity `||DeltaW_c||_F^2 = ||DeltaW_perp||_F^2 + ||DeltaW_par||_F^2`
//! and eliminating interference on protected anchor feature subspaces to IEEE-754 `f64` precision.

use std::collections::BTreeMap;

use crate::linalg::{
    axpy_slice, dot_slice, matmul, qr_core_singular_values, qr_core_svd_truncated, scale_slice,
    svd_to_lora_factors,
};
use crate::merge_check::{
    resolve_effective_layer_scale, AdapterMergeInput, IncompatibleShapeFinding,
};
use crate::safetensors::{convert_layers_key_format, DType, LoraLayer, Tensor2D};

/// SARIF v2.1.0 rule IDs for cross-checkpoint orthogonal projection.
pub const LS041_SIGNAL_ANNIHILATION: &str = "LS041";
pub const LS042_ANCHOR_SUBSPACE_SATURATION: &str = "LS042";

/// Subspace domain over which anchor orthogonality is enforced.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrthoDomain {
    /// Input-feature nullspace projection `A_perp = A_c * (I - lambda * Q_V * Q_V^T)` (O-LoRA / InfLoRA).
    /// Ensures `DeltaW_perp * x = 0` for all `x` in the active anchor input subspace `span(Q_V)`.
    Right,
    /// Output-feature nullspace projection `B_perp = (I - lambda * Q_U * Q_U^T) * B_c`.
    Left,
    /// Double-sided principal subspace projection `(I - lambda * Q_U * Q_U^T) * DeltaW_c * (I - lambda * Q_V * Q_V^T)`.
    Bilateral,
}

impl OrthoDomain {
    pub fn as_str(&self) -> &'static str {
        match self {
            OrthoDomain::Right => "right",
            OrthoDomain::Left => "left",
            OrthoDomain::Bilateral => "bilateral",
        }
    }

    pub fn from_name(s: &str) -> Result<Self, String> {
        match s.trim().to_ascii_lowercase().as_str() {
            "right" | "input" | "olora" | "inflora" => Ok(OrthoDomain::Right),
            "left" | "output" => Ok(OrthoDomain::Left),
            "bilateral" | "both" | "two-sided" | "double" => Ok(OrthoDomain::Bilateral),
            other => Err(format!(
                "Unsupported orthogonal domain '{}': expected right, left, or bilateral",
                other
            )),
        }
    }
}

/// Configuration for `project_adapter_orthogonal`.
#[derive(Debug, Clone)]
pub struct OrthoProjectConfig {
    pub domain: OrthoDomain,
    /// Cumulative Frobenius energy fraction `(0.0, 1.0]` retained per anchor layer when building
    /// protected basis `Q_V` / `Q_U` (default `0.99` avoids locking SGD noise tail dimensions).
    pub anchor_energy_threshold: f64,
    /// Deflation strength `lambda` in `[0.0, 1.0]` (`1.0` = exact nullspace projection).
    pub strength: f64,
    /// Optional target output rank `k` (defaults to candidate layer's nominal rank).
    pub target_rank: Option<usize>,
    /// Rescale projected non-zero layer so `||DeltaW_perp||_F == ||DeltaW_c||_F`.
    pub preserve_norm: bool,
    /// Fold candidate effective scale `alpha / r` (or `alpha / sqrt(r)`) directly into output `(B, A)`.
    pub bake_alpha: bool,
    /// Output key naming convention (`peft`, `kohya`, or `diffusers`).
    pub key_format: String,
    /// Optional output dtype override.
    pub output_dtype: Option<DType>,
}

impl Default for OrthoProjectConfig {
    fn default() -> Self {
        Self {
            domain: OrthoDomain::Right,
            anchor_energy_threshold: 0.99,
            strength: 1.0,
            target_rank: None,
            preserve_norm: false,
            bake_alpha: false,
            key_format: "peft".to_string(),
            output_dtype: None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrthoSeverity {
    Error,
    Warning,
}

impl OrthoSeverity {
    pub fn as_str(&self) -> &'static str {
        match self {
            OrthoSeverity::Error => "error",
            OrthoSeverity::Warning => "warning",
        }
    }
}

#[derive(Debug, Clone)]
pub struct OrthoDiagnostic {
    pub rule_id: &'static str,
    pub severity: OrthoSeverity,
    pub layer_name: String,
    pub artifact_uri: String,
    pub message: String,
    pub retained_ratio: f64,
    pub ambient_saturation_ratio: f64,
}

#[derive(Debug, Clone)]
pub struct OrthoLayerReport {
    pub base_name: String,
    pub d_out: usize,
    pub d_in: usize,
    pub candidate_rank: usize,
    pub output_rank: usize,
    pub anchor_count: usize,
    pub anchor_union_dim_u: usize,
    pub anchor_union_dim_v: usize,
    pub ambient_saturation_ratio: f64,
    pub pre_energy: f64,
    pub post_projection_energy: f64,
    pub removed_interference_energy: f64,
    pub retained_ratio: f64,
    pub pre_max_anchor_cosine: f64,
    pub post_max_anchor_cosine: f64,
    pub post_effective_rank: f64,
}

#[derive(Debug, Clone)]
pub struct OrthoProjectSummary {
    pub candidate_name: String,
    pub anchor_names: Vec<String>,
    pub domain: OrthoDomain,
    pub anchor_energy_threshold: f64,
    pub strength: f64,
    pub preserve_norm: bool,
    pub projected_layers: Vec<LoraLayer>,
    pub reports: Vec<OrthoLayerReport>,
    pub passthrough_layers: Vec<String>,
    pub incompatible_shapes: Vec<IncompatibleShapeFinding>,
    pub diagnostics: Vec<OrthoDiagnostic>,
    pub total_pre_energy: f64,
    pub total_post_projection_energy: f64,
    pub total_removed_interference_energy: f64,
    pub global_retained_ratio: f64,
    pub max_pre_anchor_cosine: f64,
    pub max_post_anchor_cosine: f64,
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

/// Exact low-rank Frobenius inner product `<s1 * B1 * A1, s2 * B2 * A2>_F` in `O(r1 * r2 * (d_out + d_in))`.
fn low_rank_frobenius_inner(
    b1: &Tensor2D,
    a1: &Tensor2D,
    s1: f64,
    b2: &Tensor2D,
    a2: &Tensor2D,
    s2: f64,
) -> f64 {
    let b1_t = crate::linalg::transpose(b1);
    let mb = matmul(&b1_t, b2); // (r1 x r2)
    let a2_t = crate::linalg::transpose(a2);
    let ma = matmul(a1, &a2_t); // (r1 x r2)
    let mut sum = 0.0;
    for (&xb, &xa) in mb.data.iter().zip(ma.data.iter()) {
        sum += xb * xa;
    }
    s1 * s2 * sum
}

fn low_rank_frobenius_cosine(
    b1: &Tensor2D,
    a1: &Tensor2D,
    s1: f64,
    n1: f64,
    b2: &Tensor2D,
    a2: &Tensor2D,
    s2: f64,
    n2: f64,
) -> f64 {
    let denom = n1 * n2;
    if denom <= 1e-18 {
        return 0.0;
    }
    (low_rank_frobenius_inner(b1, a1, s1, b2, a2, s2).abs() / denom).clamp(0.0, 1.0)
}

/// Exact low-rank difference squared norm `||s1 * B1 * A1 - s2 * B2 * A2||_F^2` via QR-Core SVD.
fn low_rank_diff_energy(
    b1: &Tensor2D,
    a1: &Tensor2D,
    s1: f64,
    b2: &Tensor2D,
    a2: &Tensor2D,
    s2: f64,
) -> f64 {
    let d_out = b1.rows;
    let d_in = a1.cols;
    let r1 = a1.rows;
    let r2 = a2.rows;
    let r_sum = r1 + r2;
    let mut b_cat = Tensor2D::zeros(d_out, r_sum, DType::F32);
    let mut a_cat = Tensor2D::zeros(r_sum, d_in, DType::F32);

    for row in 0..d_out {
        for c in 0..r1 {
            b_cat.set(row, c, s1 * b1.get(row, c));
        }
        for c in 0..r2 {
            b_cat.set(row, r1 + c, -s2 * b2.get(row, c));
        }
    }
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

    let sigmas = qr_core_singular_values(&b_cat, &a_cat);
    sigmas.iter().map(|&s| s * s).sum()
}

/// Orthonormalize a set of column vectors in `R^dim` via two-pass DGKS Modified Gram-Schmidt,
/// discarding numerically linearly dependent columns (`norm <= 1e-11`).
fn orthonormalize_union_columns(raw_cols: &[Vec<f64>], dim: usize) -> Vec<Vec<f64>> {
    let mut basis: Vec<Vec<f64>> = Vec::new();
    for src in raw_cols {
        if src.len() != dim {
            continue;
        }
        let orig_norm = dot_slice(src, src).sqrt();
        if orig_norm <= 1e-14 {
            continue;
        }
        let mut v = src.clone();
        for q in &basis {
            let proj = dot_slice(q, &v);
            axpy_slice(&mut v, -proj, q);
        }
        for q in &basis {
            let corr = dot_slice(q, &v);
            if corr.abs() > 1e-16 {
                axpy_slice(&mut v, -corr, q);
            }
        }
        let rem_norm = dot_slice(&v, &v).sqrt();
        if rem_norm > (orig_norm * 1e-10).max(1e-12) {
            scale_slice(&mut v, 1.0 / rem_norm);
            basis.push(v);
            if basis.len() >= dim {
                break;
            }
        }
    }
    basis
}

fn select_energy_rank(sigma: &[f64], energy_threshold: f64) -> usize {
    let total_e: f64 = sigma.iter().map(|&s| s * s).sum();
    if total_e <= 1e-24 || sigma.is_empty() {
        return 0;
    }
    let sigma_max = sigma[0];
    let tol = (sigma_max * 1e-12).max(1e-15);
    let mut acc = 0.0;
    let target = (energy_threshold.clamp(1e-6, 1.0) * total_e).min(total_e);
    let mut k = 0usize;
    for &s in sigma {
        if s <= tol {
            break;
        }
        acc += s * s;
        k += 1;
        if acc + 1e-15 * total_e >= target {
            break;
        }
    }
    k
}

fn von_neumann_effective_rank(sigma: &[f64]) -> f64 {
    let total: f64 = sigma.iter().map(|&s| s * s).sum();
    if total <= 1e-24 {
        return 0.0;
    }
    let mut h = 0.0;
    for &s in sigma {
        let e = s * s;
        if e > 1e-28 {
            let p = (e / total).clamp(0.0, 1.0);
            if p > 0.0 {
                h -= p * p.ln();
            }
        }
    }
    h.exp()
}

/// Project `candidate` onto the orthogonal complement of `anchors`.
pub fn project_adapter_orthogonal(
    candidate: &AdapterMergeInput,
    anchors: &[AdapterMergeInput],
    cfg: &OrthoProjectConfig,
) -> Result<OrthoProjectSummary, String> {
    if anchors.is_empty() {
        return Err("At least one anchor adapter is required for orthogonal projection".to_string());
    }
    if !cfg.anchor_energy_threshold.is_finite()
        || cfg.anchor_energy_threshold <= 0.0
        || cfg.anchor_energy_threshold > 1.0
    {
        return Err("anchor_energy_threshold must be in (0.0, 1.0]".to_string());
    }
    if !cfg.strength.is_finite() || !(0.0..=1.0).contains(&cfg.strength) {
        return Err("strength must be in [0.0, 1.0]".to_string());
    }
    if let Some(k) = cfg.target_rank {
        if k == 0 {
            return Err("target_rank must be >= 1".to_string());
        }
    }

    // Index anchor layers by base_name
    let mut anchor_maps: Vec<BTreeMap<&str, &LoraLayer>> = Vec::with_capacity(anchors.len());
    for anc in anchors {
        let mut m = BTreeMap::new();
        for l in &anc.layers {
            m.insert(l.base_name.as_str(), l);
        }
        anchor_maps.push(m);
    }

    let mut raw_out_layers = Vec::with_capacity(candidate.layers.len());
    let mut reports = Vec::new();
    let mut passthrough_layers = Vec::new();
    let mut incompatible_shapes = Vec::new();
    let mut diagnostics = Vec::new();

    let mut total_pre_e = 0.0;
    let mut total_post_e = 0.0;
    let mut total_rem_e = 0.0;
    let mut max_pre_cos = 0.0_f64;
    let mut max_post_cos = 0.0_f64;

    for cand_layer in &candidate.layers {
        let d_out = cand_layer.d_out();
        let d_in = cand_layer.d_in();
        let out_dtype = cfg.output_dtype.unwrap_or(cand_layer.a.dtype);
        let s_cand = resolve_effective_layer_scale(
            cand_layer,
            &candidate.alphas,
            candidate.peft_config.as_ref(),
        );

        // Gather matching anchor layers for this module
        let mut matched_anchors: Vec<(usize, &LoraLayer, f64)> = Vec::new();
        let mut shape_conflict = false;
        for (a_idx, amap) in anchor_maps.iter().enumerate() {
            if let Some(&anc_layer) = amap.get(cand_layer.base_name.as_str()) {
                if anc_layer.d_out() != d_out || anc_layer.d_in() != d_in {
                    incompatible_shapes.push(IncompatibleShapeFinding {
                        base_name: cand_layer.base_name.clone(),
                        adapter_a: candidate.name.clone(),
                        shape_a: (d_out, d_in),
                        adapter_b: anchors[a_idx].name.clone(),
                        shape_b: (anc_layer.d_out(), anc_layer.d_in()),
                    });
                    shape_conflict = true;
                } else {
                    let s_anc = resolve_effective_layer_scale(
                        anc_layer,
                        &anchors[a_idx].alphas,
                        anchors[a_idx].peft_config.as_ref(),
                    );
                    matched_anchors.push((a_idx, anc_layer, s_anc));
                }
            }
        }

        if shape_conflict {
            continue;
        }

        if matched_anchors.is_empty() {
            passthrough_layers.push(cand_layer.base_name.clone());
            let mut copy_l = cand_layer.clone();
            copy_l.a.dtype = out_dtype;
            copy_l.b.dtype = out_dtype;
            if cfg.bake_alpha && (s_cand - 1.0).abs() > 1e-12 {
                let sqrt_s = s_cand.abs().sqrt();
                let sign_s = if s_cand < 0.0 { -1.0 } else { 1.0 };
                for v in &mut copy_l.b.data {
                    *v *= sign_s * sqrt_s;
                }
                for v in &mut copy_l.a.data {
                    *v *= sqrt_s;
                }
            }
            raw_out_layers.push(copy_l);
            continue;
        }

        // Candidate baseline spectrum & Frobenius energy
        let cand_sigmas = qr_core_singular_values(&cand_layer.b, &cand_layer.a);
        let pre_energy: f64 = cand_sigmas
            .iter()
            .map(|&s| (s_cand * s) * (s_cand * s))
            .sum();
        let pre_norm = pre_energy.sqrt();

        // Extract active anchor truncated factors & collect left/right singular vectors
        let mut raw_u_cols: Vec<Vec<f64>> = Vec::new();
        let mut raw_v_cols: Vec<Vec<f64>> = Vec::new();
        let mut truncated_anchors: Vec<(Tensor2D, Tensor2D, f64, f64)> = Vec::new();

        for &(_a_idx, anc_layer, s_anc) in &matched_anchors {
            let anc_svd = qr_core_svd_truncated(&anc_layer.b, &anc_layer.a, anc_layer.rank());
            let keep_k = select_energy_rank(&anc_svd.sigma, cfg.anchor_energy_threshold);
            if keep_k == 0 {
                continue;
            }
            if matches!(cfg.domain, OrthoDomain::Left | OrthoDomain::Bilateral) {
                for j in 0..keep_k.min(anc_svd.u.cols) {
                    let mut col = vec![0.0; d_out];
                    for r_idx in 0..d_out {
                        col[r_idx] = anc_svd.u.get(r_idx, j);
                    }
                    raw_u_cols.push(col);
                }
            }
            if matches!(cfg.domain, OrthoDomain::Right | OrthoDomain::Bilateral) {
                for j in 0..keep_k.min(anc_svd.v.cols) {
                    let mut col = vec![0.0; d_in];
                    for r_idx in 0..d_in {
                        col[r_idx] = anc_svd.v.get(r_idx, j);
                    }
                    raw_v_cols.push(col);
                }
            }

            let (b_anc_k, a_anc_k) = svd_to_lora_factors(&anc_svd, keep_k, DType::F32);
            let anc_e: f64 = anc_svd.sigma[..keep_k]
                .iter()
                .map(|&s| (s_anc * s) * (s_anc * s))
                .sum();
            truncated_anchors.push((b_anc_k, a_anc_k, s_anc, anc_e.sqrt()));
        }

        let q_u = if matches!(cfg.domain, OrthoDomain::Left | OrthoDomain::Bilateral) {
            orthonormalize_union_columns(&raw_u_cols, d_out)
        } else {
            Vec::new()
        };
        let q_v = if matches!(cfg.domain, OrthoDomain::Right | OrthoDomain::Bilateral) {
            orthonormalize_union_columns(&raw_v_cols, d_in)
        } else {
            Vec::new()
        };

        // Pre-projection maximum Frobenius cosine against active anchor subspaces
        let mut pre_layer_max_cos = 0.0_f64;
        for (b_ak, a_ak, s_ak, norm_ak) in &truncated_anchors {
            let cos = low_rank_frobenius_cosine(
                &cand_layer.b,
                &cand_layer.a,
                s_cand,
                pre_norm,
                b_ak,
                a_ak,
                *s_ak,
                *norm_ak,
            );
            if cos > pre_layer_max_cos {
                pre_layer_max_cos = cos;
            }
        }

        // Deflate B_c and/or A_c with strength lambda (two-pass DGKS when lambda == 1.0)
        let mut b_proj = cand_layer.b.clone();
        b_proj.dtype = DType::F32;
        if !q_u.is_empty() && cfg.strength > 0.0 {
            let r_c = b_proj.cols;
            for c_idx in 0..r_c {
                let mut col = vec![0.0; d_out];
                for r_idx in 0..d_out {
                    col[r_idx] = b_proj.get(r_idx, c_idx);
                }
                for qu in &q_u {
                    let proj = dot_slice(qu, &col);
                    axpy_slice(&mut col, -cfg.strength * proj, qu);
                }
                if (cfg.strength - 1.0).abs() < 1e-12 {
                    for qu in &q_u {
                        let corr = dot_slice(qu, &col);
                        if corr.abs() > 1e-16 {
                            axpy_slice(&mut col, -corr, qu);
                        }
                    }
                }
                for r_idx in 0..d_out {
                    b_proj.set(r_idx, c_idx, col[r_idx]);
                }
            }
        }

        let mut a_proj = cand_layer.a.clone();
        a_proj.dtype = DType::F32;
        if !q_v.is_empty() && cfg.strength > 0.0 {
            let r_c = a_proj.rows;
            for r_idx in 0..r_c {
                let row_slice = &mut a_proj.data[r_idx * d_in..(r_idx + 1) * d_in];
                for qv in &q_v {
                    let proj = dot_slice(qv, row_slice);
                    axpy_slice(row_slice, -cfg.strength * proj, qv);
                }
                if (cfg.strength - 1.0).abs() < 1e-12 {
                    for qv in &q_v {
                        let corr = dot_slice(qv, row_slice);
                        if corr.abs() > 1e-16 {
                            axpy_slice(row_slice, -corr, qv);
                        }
                    }
                }
            }
        }

        // Canonicalize (b_proj, a_proj) via QR-Core SVD into clean balanced factors
        let target_k = cfg
            .target_rank
            .unwrap_or_else(|| cand_layer.rank())
            .min(d_out.min(d_in))
            .max(1);

        let proj_svd = if target_k <= b_proj.cols {
            qr_core_svd_truncated(&b_proj, &a_proj, target_k)
        } else {
            // Pad columns if user requested higher nominal rank than candidate
            let svd_raw = qr_core_svd_truncated(&b_proj, &a_proj, b_proj.cols);
            svd_raw
        };

        let keep_k = target_k.min(proj_svd.u.cols).max(1);
        let (mut b_clean, mut a_clean) = svd_to_lora_factors(&proj_svd, keep_k, out_dtype);

        let post_sigmas = &proj_svd.sigma[..keep_k.min(proj_svd.sigma.len())];
        let post_energy: f64 = post_sigmas
            .iter()
            .map(|&s| (s_cand * s) * (s_cand * s))
            .sum();
        let post_norm = post_energy.sqrt();
        let removed_energy = low_rank_diff_energy(
            &cand_layer.b,
            &cand_layer.a,
            s_cand,
            &b_clean,
            &a_clean,
            s_cand,
        );
        let retained_ratio = if pre_energy > 1e-24 {
            (post_energy / pre_energy).clamp(0.0, 1.0)
        } else {
            1.0
        };

        let mut post_layer_max_cos = 0.0_f64;
        for (b_ak, a_ak, s_ak, norm_ak) in &truncated_anchors {
            let cos = low_rank_frobenius_cosine(
                &b_clean, &a_clean, s_cand, post_norm, b_ak, a_ak, *s_ak, *norm_ak,
            );
            if cos > post_layer_max_cos {
                post_layer_max_cos = cos;
            }
        }

        // Optional norm preservation & alpha baking on output weights
        let mut factor_mult = 1.0_f64;
        if cfg.preserve_norm && post_norm > 1e-12 && pre_norm > 0.0 {
            factor_mult *= pre_norm / post_norm;
        }
        if cfg.bake_alpha && (s_cand - 1.0).abs() > 1e-12 {
            factor_mult *= s_cand.abs();
        }
        if (factor_mult - 1.0).abs() > 1e-12 {
            let sqrt_m = factor_mult.sqrt();
            let sign_m = if cfg.bake_alpha && s_cand < 0.0 {
                -1.0
            } else {
                1.0
            };
            for v in &mut b_clean.data {
                *v *= sign_m * sqrt_m;
            }
            for v in &mut a_clean.data {
                *v *= sqrt_m;
            }
        }

        // Ambient saturation ratio & SARIF checks (LS041, LS042)
        let sat_u = (q_u.len() as f64) / (d_out as f64);
        let sat_v = (q_v.len() as f64) / (d_in as f64);
        let ambient_sat = match cfg.domain {
            OrthoDomain::Right => sat_v,
            OrthoDomain::Left => sat_u,
            OrthoDomain::Bilateral => sat_u.max(sat_v),
        };
        let free_null_dim = match cfg.domain {
            OrthoDomain::Right => d_in.saturating_sub(q_v.len()),
            OrthoDomain::Left => d_out.saturating_sub(q_u.len()),
            OrthoDomain::Bilateral => d_out
                .saturating_sub(q_u.len())
                .min(d_in.saturating_sub(q_v.len())),
        };

        if pre_energy > 1e-18 && retained_ratio < 0.30 {
            let sev = if retained_ratio < 0.05 {
                OrthoSeverity::Error
            } else {
                OrthoSeverity::Warning
            };
            diagnostics.push(OrthoDiagnostic {
                rule_id: LS041_SIGNAL_ANNIHILATION,
                severity: sev,
                layer_name: cand_layer.base_name.clone(),
                artifact_uri: candidate.name.clone(),
                message: format!(
                    "Layer '{}' retained only {:.2}% of Frobenius energy after orthogonal projection against {} anchor(s); {:.2}% collided with protected anchor subspaces.",
                    cand_layer.base_name,
                    retained_ratio * 100.0,
                    matched_anchors.len(),
                    (1.0 - retained_ratio) * 100.0
                ),
                retained_ratio,
                ambient_saturation_ratio: ambient_sat,
            });
        }

        if ambient_sat >= 0.75 || free_null_dim < cand_layer.rank() {
            diagnostics.push(OrthoDiagnostic {
                rule_id: LS042_ANCHOR_SUBSPACE_SATURATION,
                severity: OrthoSeverity::Warning,
                layer_name: cand_layer.base_name.clone(),
                artifact_uri: candidate.name.clone(),
                message: format!(
                    "Layer '{}' anchor subspace occupies {:.1}% of ambient dimension (dim_u={}/{}, dim_v={}/{}), leaving {} orthogonal dimension(s) for candidate rank {}.",
                    cand_layer.base_name,
                    ambient_sat * 100.0,
                    q_u.len(),
                    d_out,
                    q_v.len(),
                    d_in,
                    free_null_dim,
                    cand_layer.rank()
                ),
                retained_ratio,
                ambient_saturation_ratio: ambient_sat,
            });
        }

        total_pre_e += pre_energy;
        total_post_e += post_energy;
        total_rem_e += removed_energy;
        if pre_layer_max_cos > max_pre_cos {
            max_pre_cos = pre_layer_max_cos;
        }
        if post_layer_max_cos > max_post_cos {
            max_post_cos = post_layer_max_cos;
        }

        let post_eff_rank = von_neumann_effective_rank(post_sigmas);
        let out_rank = b_clean.cols;

        raw_out_layers.push(LoraLayer {
            base_name: cand_layer.base_name.clone(),
            a_key: cand_layer.a_key.clone(),
            b_key: cand_layer.b_key.clone(),
            a: a_clean,
            b: b_clean,
        });

        reports.push(OrthoLayerReport {
            base_name: cand_layer.base_name.clone(),
            d_out,
            d_in,
            candidate_rank: cand_layer.rank(),
            output_rank: out_rank,
            anchor_count: matched_anchors.len(),
            anchor_union_dim_u: q_u.len(),
            anchor_union_dim_v: q_v.len(),
            ambient_saturation_ratio: ambient_sat,
            pre_energy,
            post_projection_energy: post_energy,
            removed_interference_energy: removed_energy,
            retained_ratio,
            pre_max_anchor_cosine: pre_layer_max_cos,
            post_max_anchor_cosine: post_layer_max_cos,
            post_effective_rank: post_eff_rank,
        });
    }

    convert_layers_key_format(&mut raw_out_layers, &cfg.key_format)?;
    let global_retained = if total_pre_e > 1e-24 {
        (total_post_e / total_pre_e).clamp(0.0, 1.0)
    } else {
        1.0
    };

    Ok(OrthoProjectSummary {
        candidate_name: candidate.name.clone(),
        anchor_names: anchors.iter().map(|a| a.name.clone()).collect(),
        domain: cfg.domain,
        anchor_energy_threshold: cfg.anchor_energy_threshold,
        strength: cfg.strength,
        preserve_norm: cfg.preserve_norm,
        projected_layers: raw_out_layers,
        reports,
        passthrough_layers,
        incompatible_shapes,
        diagnostics,
        total_pre_energy: total_pre_e,
        total_post_projection_energy: total_post_e,
        total_removed_interference_energy: total_rem_e,
        global_retained_ratio: global_retained,
        max_pre_anchor_cosine: max_pre_cos,
        max_post_anchor_cosine: max_post_cos,
    })
}

pub fn render_ortho_project_table(summary: &OrthoProjectSummary) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "Cross-Checkpoint Orthogonal Projector (domain={}, anchor_energy={:.2}, strength={:.2}, anchors={})\n",
        summary.domain.as_str(),
        summary.anchor_energy_threshold,
        summary.strength,
        summary.anchor_names.len()
    ));
    out.push_str(&format!(
        "{:<36} {:>9} {:>9} {:>9} {:>9} {:>10} {:>10}\n",
        "LAYER", "ANCH_DIM", "SAT%", "RETAIN%", "EFF_RANK", "PRE_COS", "POST_COS"
    ));
    out.push_str(&"-".repeat(100));
    out.push('\n');

    for r in &summary.reports {
        let short_name = if r.base_name.len() > 36 {
            format!("...{}", &r.base_name[r.base_name.len() - 33..])
        } else {
            r.base_name.clone()
        };
        let anc_dim = match summary.domain {
            OrthoDomain::Right => format!("v={}", r.anchor_union_dim_v),
            OrthoDomain::Left => format!("u={}", r.anchor_union_dim_u),
            OrthoDomain::Bilateral => format!("{}/{}", r.anchor_union_dim_u, r.anchor_union_dim_v),
        };
        out.push_str(&format!(
            "{:<36} {:>9} {:>8.1}% {:>8.2}% {:>9.2} {:>10.6} {:>10.2e}\n",
            short_name,
            anc_dim,
            r.ambient_saturation_ratio * 100.0,
            r.retained_ratio * 100.0,
            r.post_effective_rank,
            r.pre_max_anchor_cosine,
            r.post_max_anchor_cosine
        ));
    }

    out.push_str(&"-".repeat(100));
    out.push('\n');
    out.push_str(&format!(
        "SUMMARY ({} projected, {} passthrough): retained_energy={:.2}%, max_anchor_cosine {:.6} -> {:.2e}, diagnostics={}\n",
        summary.reports.len(),
        summary.passthrough_layers.len(),
        summary.global_retained_ratio * 100.0,
        summary.max_pre_anchor_cosine,
        summary.max_post_anchor_cosine,
        summary.diagnostics.len()
    ));
    out
}

pub fn render_ortho_project_json(summary: &OrthoProjectSummary) -> String {
    let anchors_json: Vec<String> = summary
        .anchor_names
        .iter()
        .map(|a| format!("\"{}\"", json_escape(a)))
        .collect();

    let layers_json: Vec<String> = summary
        .reports
        .iter()
        .map(|r| {
            format!(
                r#"    {{
      "base_name": "{}",
      "d_out": {},
      "d_in": {},
      "candidate_rank": {},
      "output_rank": {},
      "anchor_count": {},
      "anchor_union_dim_u": {},
      "anchor_union_dim_v": {},
      "ambient_saturation_ratio": {:.6},
      "pre_energy": {:.8},
      "post_projection_energy": {:.8},
      "removed_interference_energy": {:.8},
      "retained_ratio": {:.6},
      "pre_max_anchor_cosine": {:.8},
      "post_max_anchor_cosine": {:.12},
      "post_effective_rank": {:.6}
    }}"#,
                json_escape(&r.base_name),
                r.d_out,
                r.d_in,
                r.candidate_rank,
                r.output_rank,
                r.anchor_count,
                r.anchor_union_dim_u,
                r.anchor_union_dim_v,
                r.ambient_saturation_ratio,
                r.pre_energy,
                r.post_projection_energy,
                r.removed_interference_energy,
                r.retained_ratio,
                r.pre_max_anchor_cosine,
                r.post_max_anchor_cosine,
                r.post_effective_rank
            )
        })
        .collect();

    let diags_json: Vec<String> = summary
        .diagnostics
        .iter()
        .map(|d| {
            format!(
                r#"    {{
      "rule_id": "{}",
      "severity": "{}",
      "layer_name": "{}",
      "message": "{}"
    }}"#,
                d.rule_id,
                d.severity.as_str(),
                json_escape(&d.layer_name),
                json_escape(&d.message)
            )
        })
        .collect();

    format!(
        r#"{{
  "candidate_name": "{}",
  "anchor_names": [{}],
  "domain": "{}",
  "anchor_energy_threshold": {:.4},
  "strength": {:.4},
  "preserve_norm": {},
  "total_pre_energy": {:.8},
  "total_post_projection_energy": {:.8},
  "total_removed_interference_energy": {:.8},
  "global_retained_ratio": {:.6},
  "max_pre_anchor_cosine": {:.8},
  "max_post_anchor_cosine": {:.12},
  "layers": [
{}
  ],
  "diagnostics": [
{}
  ]
}}
"#,
        json_escape(&summary.candidate_name),
        anchors_json.join(", "),
        summary.domain.as_str(),
        summary.anchor_energy_threshold,
        summary.strength,
        summary.preserve_norm,
        summary.total_pre_energy,
        summary.total_post_projection_energy,
        summary.total_removed_interference_energy,
        summary.global_retained_ratio,
        summary.max_pre_anchor_cosine,
        summary.max_post_anchor_cosine,
        layers_json.join(",\n"),
        diags_json.join(",\n")
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Rng64;

    fn random_lora_layer(name: &str, d_out: usize, d_in: usize, r: usize, rng: &mut Rng64) -> LoraLayer {
        let mut a = Tensor2D::zeros(r, d_in, DType::F32);
        let mut b = Tensor2D::zeros(d_out, r, DType::F32);
        let sa = 1.0 / (d_in as f64).sqrt();
        let sb = 1.0 / (r as f64).sqrt();
        for v in &mut a.data {
            *v = rng.next_gaussian() * sa;
        }
        for v in &mut b.data {
            *v = rng.next_gaussian() * sb;
        }
        LoraLayer {
            base_name: name.to_string(),
            a_key: format!("{}.lora_A.weight", name),
            b_key: format!("{}.lora_B.weight", name),
            a,
            b,
        }
    }

    #[test]
    fn test_right_left_and_bilateral_projection_zero_anchor_interference_and_pythagorean_ledger() {
        let mut rng = Rng64::new(202611);
        let d_out = 48;
        let d_in = 40;
        let r = 6;

        let anchor_1 = random_lora_layer("layers.0.self_attn.q_proj", d_out, d_in, 4, &mut rng);
        let anchor_2 = random_lora_layer("layers.0.self_attn.q_proj", d_out, d_in, 4, &mut rng);

        // Mix both B and A of anchor_1 into candidate so pre-projection Frobenius cosine is substantial
        let mut cand = random_lora_layer("layers.0.self_attn.q_proj", d_out, d_in, r, &mut rng);
        for row in 0..4 {
            for col in 0..d_in {
                cand.a.set(row, col, cand.a.get(row, col) + 0.8 * anchor_1.a.get(row, col));
            }
        }
        for row in 0..d_out {
            for col in 0..4 {
                cand.b.set(row, col, cand.b.get(row, col) + 0.8 * anchor_1.b.get(row, col));
            }
        }

        let cand_in = AdapterMergeInput {
            name: "candidate.safetensors".to_string(),
            layers: vec![cand],
            alphas: BTreeMap::new(),
            peft_config: None,
            weight: 1.0,
        };
        let anc_inputs = vec![
            AdapterMergeInput {
                name: "safety_anchor.safetensors".to_string(),
                layers: vec![anchor_1.clone()],
                alphas: BTreeMap::new(),
                peft_config: None,
                weight: 1.0,
            },
            AdapterMergeInput {
                name: "math_anchor.safetensors".to_string(),
                layers: vec![anchor_2.clone()],
                alphas: BTreeMap::new(),
                peft_config: None,
                weight: 1.0,
            },
        ];

        for domain in [OrthoDomain::Right, OrthoDomain::Left, OrthoDomain::Bilateral] {
            let cfg = OrthoProjectConfig {
                domain,
                anchor_energy_threshold: 1.0, // full anchor span -> exact zero post-projection cosine
                strength: 1.0,
                ..OrthoProjectConfig::default()
            };
            let summary = project_adapter_orthogonal(&cand_in, &anc_inputs, &cfg)
                .expect("orthogonal projection");
            let rep = &summary.reports[0];

            assert!(rep.pre_max_anchor_cosine > 0.05);
            assert!(
                rep.post_max_anchor_cosine < 1e-13,
                "Domain {:?} left non-zero anchor cosine: {}",
                domain,
                rep.post_max_anchor_cosine
            );

            // Verify Pythagorean energy conservation: ||DeltaW_c||_F^2 == ||DeltaW_perp||_F^2 + ||DeltaW_par||_F^2
            let sum_e = rep.post_projection_energy + rep.removed_interference_energy;
            let rel_pythagorean_err = (sum_e - rep.pre_energy).abs() / rep.pre_energy.max(1e-12);
            assert!(
                rel_pythagorean_err < 1e-11,
                "Domain {:?} violated Pythagorean energy ledger: pre={} vs post+rem={} (rel_err={})",
                domain,
                rep.pre_energy,
                sum_e,
                rel_pythagorean_err
            );

            // For Right projection, verify exact input-subspace nullity DeltaW_perp * A_anchor^T == 0
            if domain == OrthoDomain::Right {
                let proj_l = &summary.projected_layers[0];
                let dw_perp = matmul(&proj_l.b, &proj_l.a);
                let a1_t = crate::linalg::transpose(&anchor_1.a);
                let cross = matmul(&dw_perp, &a1_t);
                let max_abs = cross.data.iter().map(|x| x.abs()).fold(0.0_f64, f64::max);
                assert!(
                    max_abs < 1e-13,
                    "Right nullspace check DeltaW_perp * A_1^T != 0: max_abs={}",
                    max_abs
                );
            }
        }
    }

    #[test]
    fn test_diagnostics_ls041_signal_annihilation_and_ls042_anchor_saturation() {
        let mut rng = Rng64::new(888);
        let d_out = 16;
        let d_in = 16;

        // Anchor occupies 13 of 16 dimensions (81.25% >= 75% -> triggers LS042)
        let anchor = random_lora_layer("layers.0.mlp.down_proj", d_out, d_in, 13, &mut rng);
        // Candidate is an exact copy of the first 4 rows/cols of anchor -> 100% annihilated (-> triggers LS041 Error)
        let mut cand_a = Tensor2D::zeros(4, d_in, DType::F32);
        let mut cand_b = Tensor2D::zeros(d_out, 4, DType::F32);
        for r in 0..4 {
            for c in 0..d_in {
                cand_a.set(r, c, anchor.a.get(r, c));
            }
        }
        for r in 0..d_out {
            for c in 0..4 {
                cand_b.set(r, c, anchor.b.get(r, c));
            }
        }
        let cand_layer = LoraLayer {
            base_name: "layers.0.mlp.down_proj".to_string(),
            a_key: "layers.0.mlp.down_proj.lora_A.weight".to_string(),
            b_key: "layers.0.mlp.down_proj.lora_B.weight".to_string(),
            a: cand_a,
            b: cand_b,
        };

        let cand_in = AdapterMergeInput {
            name: "clone_cand.safetensors".to_string(),
            layers: vec![cand_layer],
            alphas: BTreeMap::new(),
            peft_config: None,
            weight: 1.0,
        };
        let anc_in = AdapterMergeInput {
            name: "saturated_anchor.safetensors".to_string(),
            layers: vec![anchor],
            alphas: BTreeMap::new(),
            peft_config: None,
            weight: 1.0,
        };

        let summary = project_adapter_orthogonal(
            &cand_in,
            &[anc_in],
            &OrthoProjectConfig {
                domain: OrthoDomain::Right,
                anchor_energy_threshold: 1.0,
                strength: 1.0,
                ..OrthoProjectConfig::default()
            },
        )
        .expect("project");

        assert!(summary.global_retained_ratio < 1e-12);
        let rules: Vec<&str> = summary.diagnostics.iter().map(|d| d.rule_id).collect();
        assert!(rules.contains(&LS041_SIGNAL_ANNIHILATION));
        assert!(rules.contains(&LS042_ANCHOR_SUBSPACE_SATURATION));
        let ls041 = summary
            .diagnostics
            .iter()
            .find(|d| d.rule_id == LS041_SIGNAL_ANNIHILATION)
            .unwrap();
        assert_eq!(ls041.severity, OrthoSeverity::Error);
    }
}

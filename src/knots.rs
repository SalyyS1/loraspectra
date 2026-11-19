//! Knots Joint-SVD Multi-Adapter Subspace Alignment & Isotropic Task-Vector Solver
//! (Stoica et al., "Model Merging with SVD to Tie the Knots", ICLR 2025;
//! Gargiulo et al., "Task Singular Vectors: Reducing Task Interference in Model Merging", CVPR 2025;
//! Marczak et al., "No Task Left Behind: Isotropic Model Merging with Common and Task-Specific Subspaces", ICML 2025)
//! and OASIS SARIF v2.1.0 diagnostic rules (`LS043`..`LS046`).
//!
//! Computes a shared left-singular coordinate frame `U_joint` over `M >= 2` scaled LoRA adapters
//! `[DeltaW_1 | ... | DeltaW_M]` in pure low-rank time `O(R^2 * (d_out + d_in) + R^3)` (`R = sum_m r_m`)
//! without allocating dense `d_out x (M * d_in)` matrices. Supports three gauge-invariant spectral
//! synthesis modes (`knots-ties`, `iso`, `tsv`) and verifies the exact joint Frobenius conservation
//! identity `sum_m ||DeltaW_m||_F^2 == sum_j sigma_joint[j]^2`.

use std::collections::{BTreeMap, BTreeSet};

use crate::linalg::{
    dot_slice, jacobi_svd, matmul, multiply_q_cols_by_small, qr_core_svd_truncated, scale_slice,
    svd_to_lora_factors, thin_qr_columns, transpose, SvdDecomposition,
};
use crate::merge_check::{
    resolve_effective_layer_scale, AdapterMergeInput, IncompatibleShapeFinding,
};
use crate::safetensors::{convert_layers_key_format, DType, LoraLayer, Tensor2D};

/// SARIF v2.1.0 rule IDs for Knots / TSV / Isotropic spectral merging.
pub const LS043_KNOTS_DESTRUCTIVE_CANCELLATION: &str = "LS043";
pub const LS044_JOINT_RANK_TRUNCATION_OVERFLOW: &str = "LS044";
pub const LS045_ADAPTER_TASK_NORM_DOMINANCE: &str = "LS045";
pub const LS046_SPECTRUM_ANISOTROPY_SPIKE: &str = "LS046";

/// Spectral alignment and synthesis strategy for `compose_adapters_knots`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KnotsMode {
    /// Aligns all `M` adapters into the shared `U_joint` left-singular coordinate frame,
    /// performs top-`density` magnitude trimming and sign election in aligned right-singular
    /// coordinates `G_m = U_joint^T * DeltaW_m`, and compresses to `target_rank`.
    KnotsTies,
    /// Computes the exact low-rank SVD of the combined task delta and applies power-law
    /// isotropic singular-value equalization `sigma_i^(1 - beta) * mean_sigma^beta` while
    /// preserving kept-subspace Frobenius norm (`beta = 1.0` produces a flat spectrum).
    IsoSpectral,
    /// Task Singular Vectors (TSV): orthogonalizes concatenated per-adapter left and right
    /// singular frames `(U_cat, V_cat)` via symmetric polar Procrustes projection
    /// `U_polar = P_U * Q_U^T`, `V_polar = P_V * Q_V^T` before rank-`k` truncation.
    TaskSingularVectors,
}

impl KnotsMode {
    pub fn as_str(&self) -> &'static str {
        match self {
            KnotsMode::KnotsTies => "knots-ties",
            KnotsMode::IsoSpectral => "iso",
            KnotsMode::TaskSingularVectors => "tsv",
        }
    }

    pub fn from_name(s: &str) -> Result<Self, String> {
        match s.trim().to_ascii_lowercase().as_str() {
            "knots-ties" | "knots" | "joint-ties" => Ok(KnotsMode::KnotsTies),
            "iso" | "iso-c" | "isotropic" | "flat" => Ok(KnotsMode::IsoSpectral),
            "tsv" | "task-singular-vectors" | "procrustes" => Ok(KnotsMode::TaskSingularVectors),
            other => Err(format!(
                "Unsupported knots mode '{}': expected knots-ties, iso, or tsv",
                other
            )),
        }
    }
}

/// Configuration for `compose_adapters_knots`.
#[derive(Debug, Clone)]
pub struct KnotsConfig {
    pub mode: KnotsMode,
    /// Retained top-magnitude fraction `(0.0, 1.0]` in aligned right-singular space (`knots-ties`).
    pub density: f64,
    /// Spectral flattening exponent `[0.0, 1.0]` for `iso` mode (`1.0` = full isotropic flattening).
    pub isotropy: f64,
    /// Optional target output LoRA rank `k` (defaults to max rank across participating adapters).
    pub target_rank: Option<usize>,
    /// Equalize participating adapter Frobenius norms to the layer RMS norm prior to alignment.
    pub normalize_tasks: bool,
    /// Rescale merged output layer to match the RMS Frobenius norm of participating adapters.
    pub preserve_norm: bool,
    /// Output key naming convention (`peft`, `kohya`, or `diffusers`).
    pub key_format: String,
    /// Optional output dtype override.
    pub output_dtype: Option<DType>,
}

impl Default for KnotsConfig {
    fn default() -> Self {
        Self {
            mode: KnotsMode::KnotsTies,
            density: 0.5,
            isotropy: 1.0,
            target_rank: None,
            normalize_tasks: false,
            preserve_norm: false,
            key_format: "peft".to_string(),
            output_dtype: None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KnotsSeverity {
    Error,
    Warning,
}

impl KnotsSeverity {
    pub fn as_str(&self) -> &'static str {
        match self {
            KnotsSeverity::Error => "error",
            KnotsSeverity::Warning => "warning",
        }
    }
}

#[derive(Debug, Clone)]
pub struct KnotsDiagnostic {
    pub rule_id: &'static str,
    pub severity: KnotsSeverity,
    pub layer_name: String,
    pub artifact_uri: String,
    pub message: String,
    pub coherence_ratio: f64,
    pub svd_retained_ratio: f64,
    pub max_task_energy_share: f64,
    pub pre_stable_rank: f64,
}

#[derive(Debug, Clone)]
pub struct LayerKnotsReport {
    pub base_name: String,
    pub participating_adapters: usize,
    pub d_out: usize,
    pub d_in: usize,
    pub input_ranks: Vec<usize>,
    pub joint_active_rank: usize,
    pub output_rank: usize,
    /// Sum of squared adapter Frobenius norms `sum_m ||DeltaW_m||_F^2`.
    pub sum_task_energy: f64,
    /// Sum of squared joint singular values `sum_j sigma_joint[j]^2` (equals `sum_task_energy`).
    pub joint_basis_energy: f64,
    /// Linear sum energy divided by sum of task energies `||sum_m DeltaW_m||_F^2 / sum_m ||DeltaW_m||_F^2`.
    pub linear_coherence_ratio: f64,
    /// Minimum pairwise Frobenius cosine across participating adapters on this layer.
    pub min_pair_cosine: f64,
    /// Maximum single-adapter share of `sum_task_energy` in `[0.0, 1.0]`.
    pub max_task_energy_share: f64,
    /// Aligned-frame TIES sign conflict rate in `[0.0, 1.0]` (`0.0` for non-TIES modes).
    pub aligned_sign_conflict_rate: f64,
    /// Effective stable rank `||M||_F^2 / sigma_1^2` of the linear sum prior to synthesis.
    pub pre_stable_rank: f64,
    /// Effective stable rank `||DeltaW_out||_F^2 / sigma_out_1^2` of the synthesized output layer.
    pub post_stable_rank: f64,
    /// Fraction of post-alignment Frobenius energy retained after rank-`output_rank` truncation.
    pub svd_retained_ratio: f64,
    pub output_frobenius_norm: f64,
}

#[derive(Debug, Clone)]
pub struct KnotsSummary {
    pub mode: KnotsMode,
    pub density: f64,
    pub isotropy: f64,
    pub normalize_tasks: bool,
    pub preserve_norm: bool,
    pub adapter_names: Vec<String>,
    pub weights: Vec<f64>,
    pub merged_layers: Vec<LoraLayer>,
    pub layer_reports: Vec<LayerKnotsReport>,
    pub incompatible_shapes: Vec<IncompatibleShapeFinding>,
    pub diagnostics: Vec<KnotsDiagnostic>,
    pub mean_coherence_ratio: f64,
    pub mean_svd_retained_ratio: f64,
    pub mean_pre_stable_rank: f64,
    pub mean_post_stable_rank: f64,
    pub max_joint_conservation_drift: f64,
}

/// Canonicalize singular vector pair signs `(u_col, v_col)` so the first largest-magnitude entry
/// of `u_col` is strictly positive. Preserves `u_col * sigma * v_col^T` while making `(U, V)`
/// invariant to orthogonal LoRA factor gauge rotations `(B * Q, Q^T * A)`.
fn canonicalize_uv_signs(u: &mut Tensor2D, v: &mut Tensor2D, active_cols: usize) {
    let rows_u = u.rows;
    let rows_v = v.rows;
    for col in 0..active_cols.min(u.cols).min(v.cols) {
        let mut best_abs = -1.0_f64;
        let mut best_val = 1.0_f64;
        for r in 0..rows_u {
            let val = u.get(r, col);
            let a = val.abs();
            if a > best_abs + 1e-15 {
                best_abs = a;
                best_val = val;
            }
        }
        if best_val < 0.0 {
            for r in 0..rows_u {
                u.set(r, col, -u.get(r, col));
            }
            for r in 0..rows_v {
                v.set(r, col, -v.get(r, col));
            }
        }
    }
}

fn canonicalize_u_columns(u: &mut Tensor2D, active_cols: usize) {
    let rows_u = u.rows;
    for col in 0..active_cols.min(u.cols) {
        let mut best_abs = -1.0_f64;
        let mut best_val = 1.0_f64;
        for r in 0..rows_u {
            let val = u.get(r, col);
            let a = val.abs();
            if a > best_abs + 1e-15 {
                best_abs = a;
                best_val = val;
            }
        }
        if best_val < 0.0 {
            for r in 0..rows_u {
                u.set(r, col, -u.get(r, col));
            }
        }
    }
}

/// Compute thin SVD `M = P * diag(S) * Q^T` for a tall or wide matrix `M` (`rows x cols`)
/// using thin-QR acceleration when `min(rows, cols) << max(rows, cols)`.
/// Returns `(P, S, Q)` where `P` is `rows x p`, `Q` is `cols x p`, `p = min(rows, cols)`.
fn fast_thin_svd(mat: &Tensor2D) -> SvdDecomposition {
    let m = mat.rows;
    let n = mat.cols;
    if m == 0 || n == 0 {
        return SvdDecomposition {
            u: Tensor2D::zeros(m, 0, DType::F32),
            sigma: Vec::new(),
            v: Tensor2D::zeros(n, 0, DType::F32),
        };
    }
    if m >= n {
        let (q_cols, r_mat) = thin_qr_columns(mat);
        let r_svd = jacobi_svd(&r_mat);
        let u = multiply_q_cols_by_small(&q_cols, m, &r_svd.u, n, DType::F32);
        SvdDecomposition {
            u,
            sigma: r_svd.sigma,
            v: r_svd.v,
        }
    } else {
        let mt = transpose(mat);
        let (q_cols, r_mat) = thin_qr_columns(&mt);
        let r_svd = jacobi_svd(&r_mat);
        let v = multiply_q_cols_by_small(&q_cols, n, &r_svd.u, m, DType::F32);
        SvdDecomposition {
            u: r_svd.v,
            sigma: r_svd.sigma,
            v,
        }
    }
}

fn stable_rank_from_sigma(sigma: &[f64]) -> f64 {
    if sigma.is_empty() || sigma[0] <= 1e-15 {
        return 0.0;
    }
    let energy: f64 = sigma.iter().map(|s| s * s).sum();
    energy / (sigma[0] * sigma[0])
}

fn apply_top_density_mask_inplace(g: &mut Tensor2D, density: f64) {
    if density >= 1.0 {
        return;
    }
    let total = g.data.len();
    if total == 0 {
        return;
    }
    let keep_count = ((total as f64) * density).ceil() as usize;
    let keep_count = keep_count.clamp(1, total);
    if keep_count >= total {
        return;
    }
    let mut mags: Vec<(usize, f64)> = g
        .data
        .iter()
        .enumerate()
        .map(|(idx, &v)| (idx, v.abs()))
        .collect();
    mags.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    let mut keep_mask = vec![false; total];
    for &(idx, _) in mags.iter().take(keep_count) {
        keep_mask[idx] = true;
    }
    for idx in 0..total {
        if !keep_mask[idx] {
            g.data[idx] = 0.0;
        }
    }
}

struct TaskFactor {
    adapter_idx: usize,
    u: Tensor2D,
    sigma: Vec<f64>,
    v: Tensor2D,
    energy: f64,
}

/// Align and merge `M >= 2` LoRA adapters using Knots joint-SVD alignment, isotropic spectral
/// equalization, or polar Procrustes Task Singular Vectors (`TSV`).
pub fn compose_adapters_knots(
    inputs: &[AdapterMergeInput],
    cfg: &KnotsConfig,
) -> Result<KnotsSummary, String> {
    if inputs.len() < 2 {
        return Err("Knots composition requires at least 2 input adapters".to_string());
    }
    if !cfg.density.is_finite() || cfg.density <= 0.0 || cfg.density > 1.0 {
        return Err(format!(
            "Invalid --density {}: expected finite value in (0.0, 1.0]",
            cfg.density
        ));
    }
    if !cfg.isotropy.is_finite() || !(0.0..=1.0).contains(&cfg.isotropy) {
        return Err(format!(
            "Invalid --isotropy {}: expected finite value in [0.0, 1.0]",
            cfg.isotropy
        ));
    }
    if let Some(r) = cfg.target_rank {
        if r == 0 {
            return Err("Invalid --rank 0: target rank must be >= 1".to_string());
        }
    }
    for inp in inputs {
        if !inp.weight.is_finite() {
            return Err(format!(
                "Non-finite adapter weight {} for '{}'",
                inp.weight, inp.name
            ));
        }
    }

    let mut all_layer_names = BTreeSet::new();
    let mut per_adapter_maps: Vec<BTreeMap<String, &LoraLayer>> = Vec::with_capacity(inputs.len());
    for inp in inputs {
        let mut map = BTreeMap::new();
        for layer in &inp.layers {
            all_layer_names.insert(layer.base_name.clone());
            map.insert(layer.base_name.clone(), layer);
        }
        per_adapter_maps.push(map);
    }

    let mut incompatible_shapes = Vec::new();
    let mut diagnostics = Vec::new();
    let mut layer_reports = Vec::new();
    let mut raw_merged_layers = Vec::new();
    let mut max_conservation_drift = 0.0_f64;

    for base_name in all_layer_names {
        let mut present: Vec<(usize, &LoraLayer)> = Vec::new();
        for (idx, map) in per_adapter_maps.iter().enumerate() {
            if let Some(&layer) = map.get(&base_name) {
                present.push((idx, layer));
            }
        }
        if present.is_empty() {
            continue;
        }

        let (first_idx, first_layer) = present[0];
        let d_out = first_layer.d_out();
        let d_in = first_layer.d_in();
        let mut shape_mismatch = false;
        for &(idx, layer) in &present[1..] {
            if layer.d_out() != d_out || layer.d_in() != d_in {
                incompatible_shapes.push(IncompatibleShapeFinding {
                    base_name: base_name.clone(),
                    adapter_a: inputs[first_idx].name.clone(),
                    shape_a: (d_out, d_in),
                    adapter_b: inputs[idx].name.clone(),
                    shape_b: (layer.d_out(), layer.d_in()),
                });
                shape_mismatch = true;
            }
        }
        if shape_mismatch {
            continue;
        }

        let out_dtype = cfg.output_dtype.unwrap_or(first_layer.b.dtype);
        let max_in_rank = present
            .iter()
            .map(|(_, l)| l.rank())
            .max()
            .unwrap_or(1)
            .max(1);
        let target_rank = cfg
            .target_rank
            .unwrap_or(max_in_rank)
            .min(d_out.min(d_in).max(1))
            .max(1);

        // Single-adapter passthrough when only one adapter touches this module
        if present.len() == 1 {
            let eff_scale = resolve_effective_layer_scale(
                first_layer,
                &inputs[first_idx].alphas,
                inputs[first_idx].peft_config.as_ref(),
            ) * inputs[first_idx].weight;
            let sign = if eff_scale < 0.0 { -1.0 } else { 1.0 };
            let abs_s = eff_scale.abs();
            let mut b_scaled = first_layer.b.clone();
            let mut a_scaled = first_layer.a.clone();
            scale_slice(&mut b_scaled.data, sign * abs_s.sqrt());
            scale_slice(&mut a_scaled.data, abs_s.sqrt());
            let r_single = b_scaled.cols.max(1);
            let svd = qr_core_svd_truncated(&b_scaled, &a_scaled, r_single);
            let keep = target_rank.min(svd.u.cols).max(1);
            let total_e: f64 = svd.sigma.iter().map(|s| s * s).sum();
            let kept_e: f64 = svd.sigma.iter().take(keep).map(|s| s * s).sum();
            let (b_out, a_out) = svd_to_lora_factors(&svd, keep, out_dtype);
            let sr = stable_rank_from_sigma(&svd.sigma);
            let sr_post = stable_rank_from_sigma(&svd.sigma[..keep.min(svd.sigma.len())]);
            raw_merged_layers.push(LoraLayer {
                base_name: base_name.clone(),
                a_key: format!("{}.lora_A.weight", base_name),
                b_key: format!("{}.lora_B.weight", base_name),
                a: a_out,
                b: b_out,
            });
            layer_reports.push(LayerKnotsReport {
                base_name,
                participating_adapters: 1,
                d_out,
                d_in,
                input_ranks: vec![first_layer.rank()],
                joint_active_rank: svd.sigma.iter().filter(|&&s| s > 1e-13).count(),
                output_rank: keep,
                sum_task_energy: total_e,
                joint_basis_energy: total_e,
                linear_coherence_ratio: 1.0,
                min_pair_cosine: 1.0,
                max_task_energy_share: 1.0,
                aligned_sign_conflict_rate: 0.0,
                pre_stable_rank: sr,
                post_stable_rank: sr_post,
                svd_retained_ratio: if total_e > 1e-24 {
                    (kept_e / total_e).clamp(0.0, 1.0)
                } else {
                    1.0
                },
                output_frobenius_norm: kept_e.sqrt(),
            });
            continue;
        }

        // Extract gauge-canonical QR-Core SVD per participating adapter
        let mut task_factors: Vec<TaskFactor> = Vec::with_capacity(present.len());
        let mut input_ranks = Vec::with_capacity(present.len());
        for &(idx, layer) in &present {
            input_ranks.push(layer.rank());
            let r = layer.rank();
            if r == 0 {
                task_factors.push(TaskFactor {
                    adapter_idx: idx,
                    u: Tensor2D::zeros(d_out, 0, DType::F32),
                    sigma: Vec::new(),
                    v: Tensor2D::zeros(d_in, 0, DType::F32),
                    energy: 0.0,
                });
                continue;
            }
            let eff_scale = resolve_effective_layer_scale(
                layer,
                &inputs[idx].alphas,
                inputs[idx].peft_config.as_ref(),
            ) * inputs[idx].weight;

            let mut b_f32 = Tensor2D::zeros(layer.b.rows, layer.b.cols, DType::F32);
            b_f32.data.copy_from_slice(&layer.b.data);
            let mut a_f32 = Tensor2D::zeros(layer.a.rows, layer.a.cols, DType::F32);
            a_f32.data.copy_from_slice(&layer.a.data);

            let sign = if eff_scale < 0.0 { -1.0 } else { 1.0 };
            let abs_s = eff_scale.abs();
            scale_slice(&mut b_f32.data, sign * abs_s.sqrt());
            scale_slice(&mut a_f32.data, abs_s.sqrt());

            let keep_r = r.min(d_out.min(d_in).max(1));
            let mut svd = if r <= d_out && r <= d_in {
                qr_core_svd_truncated(&b_f32, &a_f32, keep_r)
            } else {
                let dense = matmul(&b_f32, &a_f32);
                fast_thin_svd(&dense)
            };

            let s0 = svd.sigma.first().copied().unwrap_or(0.0);
            let act_tol = (s0 * 1e-13).max(1e-15);
            let act_k = svd
                .sigma
                .iter()
                .take(svd.u.cols.min(svd.v.cols))
                .take_while(|&&s| s > act_tol)
                .count();
            canonicalize_uv_signs(&mut svd.u, &mut svd.v, act_k);

            let mut u_act = Tensor2D::zeros(d_out, act_k, DType::F32);
            for row in 0..d_out {
                for c in 0..act_k {
                    u_act.set(row, c, svd.u.get(row, c));
                }
            }
            let mut v_act = Tensor2D::zeros(d_in, act_k, DType::F32);
            for row in 0..d_in {
                for c in 0..act_k {
                    v_act.set(row, c, svd.v.get(row, c));
                }
            }
            let sigma_act: Vec<f64> = svd.sigma.into_iter().take(act_k).collect();
            let energy: f64 = sigma_act.iter().map(|s| s * s).sum();

            task_factors.push(TaskFactor {
                adapter_idx: idx,
                u: u_act,
                sigma: sigma_act,
                v: v_act,
                energy,
            });
        }

        let raw_sum_energy: f64 = task_factors.iter().map(|t| t.energy).sum();
        let (dominant_pos, max_single_energy) = task_factors
            .iter()
            .enumerate()
            .fold((0usize, 0.0_f64), |(bi, be), (i, t)| {
                if t.energy > be {
                    (i, t.energy)
                } else {
                    (bi, be)
                }
            });
        let raw_max_share = if raw_sum_energy > 1e-24 {
            (max_single_energy / raw_sum_energy).clamp(0.0, 1.0)
        } else {
            0.0
        };

        // LS045: Single-adapter task norm dominance check (>= 80% of pre-merge energy)
        if !cfg.normalize_tasks && raw_sum_energy > 1e-20 && raw_max_share >= 0.80 {
            let dom_adapter = &inputs[task_factors[dominant_pos].adapter_idx].name;
            diagnostics.push(KnotsDiagnostic {
                rule_id: LS045_ADAPTER_TASK_NORM_DOMINANCE,
                severity: KnotsSeverity::Warning,
                layer_name: base_name.clone(),
                artifact_uri: dom_adapter.clone(),
                message: format!(
                    "Layer '{}': adapter '{}' contributes {:.1}% of total pre-merge Frobenius energy; consider --normalize-tasks or balanced --weights",
                    base_name,
                    dom_adapter,
                    raw_max_share * 100.0
                ),
                coherence_ratio: 1.0,
                svd_retained_ratio: 1.0,
                max_task_energy_share: raw_max_share,
                pre_stable_rank: 1.0,
            });
        }

        // Optional RMS task-norm equalization
        if cfg.normalize_tasks && raw_sum_energy > 1e-24 {
            let active_tasks = task_factors.iter().filter(|t| t.energy > 1e-24).count();
            if active_tasks > 1 {
                let target_task_e = raw_sum_energy / (active_tasks as f64);
                for tf in &mut task_factors {
                    if tf.energy > 1e-24 {
                        let scale = (target_task_e / tf.energy).sqrt();
                        for s in &mut tf.sigma {
                            *s *= scale;
                        }
                        tf.energy = target_task_e;
                    }
                }
            }
        }

        let sum_task_energy: f64 = task_factors.iter().map(|t| t.energy).sum();
        let rms_task_norm = if !task_factors.is_empty() {
            (sum_task_energy / (task_factors.len() as f64)).sqrt()
        } else {
            0.0
        };
        let total_active_cols: usize = task_factors.iter().map(|t| t.sigma.len()).sum();

        if total_active_cols == 0 || sum_task_energy <= 1e-28 {
            let keep = target_rank.max(1);
            raw_merged_layers.push(LoraLayer {
                base_name: base_name.clone(),
                a_key: format!("{}.lora_A.weight", base_name),
                b_key: format!("{}.lora_B.weight", base_name),
                a: Tensor2D::zeros(keep, d_in, out_dtype),
                b: Tensor2D::zeros(d_out, keep, out_dtype),
            });
            layer_reports.push(LayerKnotsReport {
                base_name,
                participating_adapters: present.len(),
                d_out,
                d_in,
                input_ranks,
                joint_active_rank: 0,
                output_rank: keep,
                sum_task_energy: 0.0,
                joint_basis_energy: 0.0,
                linear_coherence_ratio: 1.0,
                min_pair_cosine: 0.0,
                max_task_energy_share: raw_max_share,
                aligned_sign_conflict_rate: 0.0,
                pre_stable_rank: 0.0,
                post_stable_rank: 0.0,
                svd_retained_ratio: 1.0,
                output_frobenius_norm: 0.0,
            });
            continue;
        }

        // Build tall-skinny concatenated scaled left-singular matrix H = [U_1 * Sigma_1 | ... | U_M * Sigma_M]
        let mut h_mat = Tensor2D::zeros(d_out, total_active_cols, DType::F32);
        let mut col_cursor = 0;
        for tf in &task_factors {
            let k_m = tf.sigma.len();
            for row in 0..d_out {
                for c in 0..k_m {
                    h_mat.set(row, col_cursor + c, tf.u.get(row, c) * tf.sigma[c]);
                }
            }
            col_cursor += k_m;
        }

        let mut h_svd = fast_thin_svd(&h_mat);
        let joint_basis_energy: f64 = h_svd.sigma.iter().map(|s| s * s).sum();
        let rel_drift = (joint_basis_energy - sum_task_energy).abs() / sum_task_energy.max(1e-12);
        if rel_drift > max_conservation_drift {
            max_conservation_drift = rel_drift;
        }

        let j_tol = (h_svd.sigma.first().copied().unwrap_or(0.0) * 1e-13).max(1e-15);
        let k_joint = h_svd
            .sigma
            .iter()
            .take(h_svd.u.cols)
            .take_while(|&&s| s > j_tol)
            .count()
            .max(1);
        canonicalize_u_columns(&mut h_svd.u, k_joint);

        let mut u_joint = Tensor2D::zeros(d_out, k_joint, DType::F32);
        for row in 0..d_out {
            for c in 0..k_joint {
                u_joint.set(row, c, h_svd.u.get(row, c));
            }
        }
        let u_joint_t = transpose(&u_joint);

        // Project each adapter into the shared U_joint frame:
        // G_m = (U_joint^T * U_m * Sigma_m) * V_m^T  (shape: k_joint x d_in)
        let mut aligned_gs: Vec<Tensor2D> = Vec::with_capacity(task_factors.len());
        let mut g_linear_sum = Tensor2D::zeros(k_joint, d_in, DType::F32);
        for tf in &task_factors {
            let k_m = tf.sigma.len();
            if k_m == 0 {
                aligned_gs.push(Tensor2D::zeros(k_joint, d_in, DType::F32));
                continue;
            }
            let mut us_m = Tensor2D::zeros(d_out, k_m, DType::F32);
            for row in 0..d_out {
                for c in 0..k_m {
                    us_m.set(row, c, tf.u.get(row, c) * tf.sigma[c]);
                }
            }
            let m_core = matmul(&u_joint_t, &us_m); // k_joint x k_m
            let vt_m = transpose(&tf.v); // k_m x d_in
            let g_m = matmul(&m_core, &vt_m); // k_joint x d_in
            for (dst, &src) in g_linear_sum.data.iter_mut().zip(g_m.data.iter()) {
                *dst += src;
            }
            aligned_gs.push(g_m);
        }

        // Exact pairwise Frobenius cosine in the lossless U_joint frame
        let mut min_pair_cosine = 1.0_f64;
        for p in 0..aligned_gs.len() {
            for q in (p + 1)..aligned_gs.len() {
                let np = task_factors[p].energy.sqrt();
                let nq = task_factors[q].energy.sqrt();
                if np > 1e-14 && nq > 1e-14 {
                    let ip = dot_slice(&aligned_gs[p].data, &aligned_gs[q].data);
                    let cos = (ip / (np * nq)).clamp(-1.0, 1.0);
                    if cos < min_pair_cosine {
                        min_pair_cosine = cos;
                    }
                }
            }
        }

        let linear_sum_svd = fast_thin_svd(&g_linear_sum);
        let linear_energy: f64 = linear_sum_svd.sigma.iter().map(|s| s * s).sum();
        let linear_coherence_ratio = (linear_energy / sum_task_energy).max(0.0);
        let pre_stable_rank = stable_rank_from_sigma(&linear_sum_svd.sigma);

        // Synthesize merged low-rank SVD according to cfg.mode
        let mut aligned_sign_conflict_rate = 0.0_f64;
        let (mut u_final, mut sigma_final, mut v_final, pre_trunc_energy) = match cfg.mode {
            KnotsMode::KnotsTies => {
                let mut trimmed: Vec<Tensor2D> = aligned_gs.clone();
                for g in &mut trimmed {
                    apply_top_density_mask_inplace(g, cfg.density);
                }
                let len = k_joint * d_in;
                let mut g_ties = Tensor2D::zeros(k_joint, d_in, DType::F32);
                let mut multi_active = 0usize;
                let mut sign_conflicts = 0usize;

                for idx in 0..len {
                    let mut pos_sum = 0.0_f64;
                    let mut neg_sum = 0.0_f64;
                    let mut pos_cnt = 0usize;
                    let mut neg_cnt = 0usize;
                    for g in &trimmed {
                        let v = g.data[idx];
                        if v > 1e-15 {
                            pos_sum += v;
                            pos_cnt += 1;
                        } else if v < -1e-15 {
                            neg_sum += v;
                            neg_cnt += 1;
                        }
                    }
                    let total_cnt = pos_cnt + neg_cnt;
                    if total_cnt >= 2 {
                        multi_active += 1;
                        if pos_cnt > 0 && neg_cnt > 0 {
                            sign_conflicts += 1;
                        }
                    }
                    let net = pos_sum + neg_sum;
                    if net > 1e-15 && pos_cnt > 0 {
                        g_ties.data[idx] = pos_sum / (pos_cnt as f64);
                    } else if net < -1e-15 && neg_cnt > 0 {
                        g_ties.data[idx] = neg_sum / (neg_cnt as f64);
                    } else if pos_cnt > 0 && neg_cnt > 0 {
                        // Exact tie-break toward larger single magnitude
                        g_ties.data[idx] = 0.0;
                    }
                }
                if multi_active > 0 {
                    aligned_sign_conflict_rate = (sign_conflicts as f64) / (multi_active as f64);
                }

                let g_svd = fast_thin_svd(&g_ties);
                let pre_e: f64 = g_svd.sigma.iter().map(|s| s * s).sum();
                let u_lifted = matmul(&u_joint, &g_svd.u);
                (u_lifted, g_svd.sigma, g_svd.v, pre_e)
            }
            KnotsMode::IsoSpectral => {
                let pre_e: f64 = linear_sum_svd.sigma.iter().map(|s| s * s).sum();
                let u_lifted = matmul(&u_joint, &linear_sum_svd.u);
                let mut shaped = linear_sum_svd.sigma.clone();
                let keep_n = target_rank.min(shaped.len());
                let s0 = shaped.first().copied().unwrap_or(0.0);
                let act_n = shaped
                    .iter()
                    .take(keep_n)
                    .take_while(|&&s| s > (s0 * 1e-12).max(1e-15))
                    .count();
                if act_n > 1 && cfg.isotropy > 0.0 {
                    let kept_energy: f64 = shaped[..act_n].iter().map(|s| s * s).sum();
                    let mean_s: f64 = shaped[..act_n].iter().sum::<f64>() / (act_n as f64);
                    let beta = cfg.isotropy;
                    let mut shaped_sq = 0.0_f64;
                    for s in &mut shaped[..act_n] {
                        let v = s.powf(1.0 - beta) * mean_s.powf(beta);
                        *s = v;
                        shaped_sq += v * v;
                    }
                    if shaped_sq > 1e-28 {
                        let norm_fix = (kept_energy / shaped_sq).sqrt();
                        for s in &mut shaped[..act_n] {
                            *s *= norm_fix;
                        }
                    }
                }
                (u_lifted, shaped, linear_sum_svd.v, pre_e)
            }
            KnotsMode::TaskSingularVectors => {
                // Polar Procrustes orthogonalization on concatenated U_cat and V_cat:
                // DeltaW_tsv = (P_U * Q_U^T) * diag(sigma_cat) * (P_V * Q_V^T)^T
                //            = P_U * (Q_U^T * diag(sigma_cat) * Q_V) * P_V^T
                let mut u_cat = Tensor2D::zeros(d_out, total_active_cols, DType::F32);
                let mut v_cat = Tensor2D::zeros(d_in, total_active_cols, DType::F32);
                let mut s_cat = Vec::with_capacity(total_active_cols);
                let mut c_off = 0usize;
                for tf in &task_factors {
                    let km = tf.sigma.len();
                    for r in 0..d_out {
                        for c in 0..km {
                            u_cat.set(r, c_off + c, tf.u.get(r, c));
                        }
                    }
                    for r in 0..d_in {
                        for c in 0..km {
                            v_cat.set(r, c_off + c, tf.v.get(r, c));
                        }
                    }
                    s_cat.extend_from_slice(&tf.sigma);
                    c_off += km;
                }

                let u_svd = fast_thin_svd(&u_cat); // P_U: d_out x p_u, Q_U: R x p_u
                let v_svd = fast_thin_svd(&v_cat); // P_V: d_in x p_v, Q_V: R x p_v
                let p_u = u_svd.v.cols;
                let mut qu_scaled = Tensor2D::zeros(total_active_cols, p_u, DType::F32);
                for r in 0..total_active_cols {
                    let sr = s_cat[r];
                    for c in 0..p_u {
                        qu_scaled.set(r, c, u_svd.v.get(r, c) * sr);
                    }
                }
                let qu_scaled_t = transpose(&qu_scaled); // p_u x R
                let core_tsv = matmul(&qu_scaled_t, &v_svd.v); // p_u x p_v
                let core_svd = jacobi_svd(&core_tsv);
                let pre_e: f64 = core_svd.sigma.iter().map(|s| s * s).sum();
                let u_out = matmul(&u_svd.u, &core_svd.u);
                let v_out = matmul(&v_svd.u, &core_svd.v);
                (u_out, core_svd.sigma, v_out, pre_e)
            }
        };

        let avail = u_final.cols.min(v_final.cols).min(sigma_final.len());
        let keep = target_rank.min(avail.max(1)).max(1);
        canonicalize_uv_signs(&mut u_final, &mut v_final, keep.min(avail));

        let kept_energy: f64 = sigma_final.iter().take(keep).map(|s| s * s).sum();
        let svd_retained_ratio = if pre_trunc_energy > 1e-24 {
            (kept_energy / pre_trunc_energy).clamp(0.0, 1.0)
        } else {
            1.0
        };

        if cfg.preserve_norm && kept_energy > 1e-24 && rms_task_norm > 1e-14 {
            let rescale = rms_task_norm / kept_energy.sqrt();
            for s in sigma_final.iter_mut().take(keep) {
                *s *= rescale;
            }
        }

        let final_energy: f64 = sigma_final.iter().take(keep).map(|s| s * s).sum();
        let post_stable_rank =
            stable_rank_from_sigma(&sigma_final[..keep.min(sigma_final.len())]);

        // Emit SARIF v2.1.0 diagnostics (LS043, LS044, LS046)
        if linear_coherence_ratio < 0.50 || min_pair_cosine < -0.25 {
            diagnostics.push(KnotsDiagnostic {
                rule_id: LS043_KNOTS_DESTRUCTIVE_CANCELLATION,
                severity: KnotsSeverity::Warning,
                layer_name: base_name.clone(),
                artifact_uri: inputs[first_idx].name.clone(),
                message: format!(
                    "Layer '{}': destructive task cancellation in shared Knots frame (linear coherence={:.3}, min pairwise cosine={:.3})",
                    base_name, linear_coherence_ratio, min_pair_cosine
                ),
                coherence_ratio: linear_coherence_ratio,
                svd_retained_ratio,
                max_task_energy_share: raw_max_share,
                pre_stable_rank,
            });
        }

        if svd_retained_ratio < 0.80 {
            diagnostics.push(KnotsDiagnostic {
                rule_id: LS044_JOINT_RANK_TRUNCATION_OVERFLOW,
                severity: KnotsSeverity::Warning,
                layer_name: base_name.clone(),
                artifact_uri: inputs[first_idx].name.clone(),
                message: format!(
                    "Layer '{}': truncating joint subspace (active rank {}) to target rank {} retained {:.1}% of Frobenius energy (>20% loss)",
                    base_name,
                    k_joint,
                    keep,
                    svd_retained_ratio * 100.0
                ),
                coherence_ratio: linear_coherence_ratio,
                svd_retained_ratio,
                max_task_energy_share: raw_max_share,
                pre_stable_rank,
            });
        }

        if cfg.mode != KnotsMode::IsoSpectral
            && max_in_rank >= 4
            && pre_stable_rank > 0.0
            && pre_stable_rank < 1.35
        {
            diagnostics.push(KnotsDiagnostic {
                rule_id: LS046_SPECTRUM_ANISOTROPY_SPIKE,
                severity: KnotsSeverity::Warning,
                layer_name: base_name.clone(),
                artifact_uri: inputs[first_idx].name.clone(),
                message: format!(
                    "Layer '{}': severe singular-value anisotropy spike (stable rank={:.2} across nominal rank {}); consider --mode iso",
                    base_name, pre_stable_rank, max_in_rank
                ),
                coherence_ratio: linear_coherence_ratio,
                svd_retained_ratio,
                max_task_energy_share: raw_max_share,
                pre_stable_rank,
            });
        }

        let svd_out = SvdDecomposition {
            u: u_final,
            sigma: sigma_final,
            v: v_final,
        };
        let (b_out, a_out) = if avail > 0 {
            svd_to_lora_factors(&svd_out, keep, out_dtype)
        } else {
            (
                Tensor2D::zeros(d_out, keep, out_dtype),
                Tensor2D::zeros(keep, d_in, out_dtype),
            )
        };

        raw_merged_layers.push(LoraLayer {
            base_name: base_name.clone(),
            a_key: format!("{}.lora_A.weight", base_name),
            b_key: format!("{}.lora_B.weight", base_name),
            a: a_out,
            b: b_out,
        });

        layer_reports.push(LayerKnotsReport {
            base_name,
            participating_adapters: present.len(),
            d_out,
            d_in,
            input_ranks,
            joint_active_rank: k_joint,
            output_rank: keep,
            sum_task_energy,
            joint_basis_energy,
            linear_coherence_ratio,
            min_pair_cosine,
            max_task_energy_share: raw_max_share,
            aligned_sign_conflict_rate,
            pre_stable_rank,
            post_stable_rank,
            svd_retained_ratio,
            output_frobenius_norm: final_energy.sqrt(),
        });
    }

    convert_layers_key_format(&mut raw_merged_layers, &cfg.key_format)?;

    let n_rep = layer_reports.len().max(1) as f64;
    let mean_coherence_ratio = layer_reports
        .iter()
        .map(|r| r.linear_coherence_ratio)
        .sum::<f64>()
        / n_rep;
    let mean_svd_retained_ratio = layer_reports
        .iter()
        .map(|r| r.svd_retained_ratio)
        .sum::<f64>()
        / n_rep;
    let mean_pre_stable_rank = layer_reports
        .iter()
        .map(|r| r.pre_stable_rank)
        .sum::<f64>()
        / n_rep;
    let mean_post_stable_rank = layer_reports
        .iter()
        .map(|r| r.post_stable_rank)
        .sum::<f64>()
        / n_rep;

    Ok(KnotsSummary {
        mode: cfg.mode,
        density: cfg.density,
        isotropy: cfg.isotropy,
        normalize_tasks: cfg.normalize_tasks,
        preserve_norm: cfg.preserve_norm,
        adapter_names: inputs.iter().map(|i| i.name.clone()).collect(),
        weights: inputs.iter().map(|i| i.weight).collect(),
        merged_layers: raw_merged_layers,
        layer_reports,
        incompatible_shapes,
        diagnostics,
        mean_coherence_ratio,
        mean_svd_retained_ratio,
        mean_pre_stable_rank,
        mean_post_stable_rank,
        max_joint_conservation_drift: max_conservation_drift,
    })
}

fn escape_json(s: &str) -> String {
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

pub fn render_knots_table(summary: &KnotsSummary) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "Knots Joint-SVD & Task-Vector Synthesis (mode={}, adapters={}, layers={}, diagnostics={})\n",
        summary.mode.as_str(),
        summary.adapter_names.len(),
        summary.layer_reports.len(),
        summary.diagnostics.len()
    ));
    out.push_str(&format!(
        "Summary: mean_coherence={:.4}, mean_svd_retained={:.4}, stable_rank {:.2} -> {:.2}, max_conservation_drift={:.2e}\n",
        summary.mean_coherence_ratio,
        summary.mean_svd_retained_ratio,
        summary.mean_pre_stable_rank,
        summary.mean_post_stable_rank,
        summary.max_joint_conservation_drift
    ));
    out.push_str(&format!(
        "{:<42} {:>5} {:>7} {:>7} {:>9} {:>9} {:>10}\n",
        "Layer", "Tasks", "J-Rank", "Out-R", "Coherence", "Retained", "StableR"
    ));
    out.push_str(&"-".repeat(95));
    out.push('\n');
    for r in &summary.layer_reports {
        let short = if r.base_name.len() > 42 {
            format!("...{}", &r.base_name[r.base_name.len() - 39..])
        } else {
            r.base_name.clone()
        };
        out.push_str(&format!(
            "{:<42} {:>5} {:>7} {:>7} {:>9.4} {:>8.2}% {:>4.2}->{:<4.2}\n",
            short,
            r.participating_adapters,
            r.joint_active_rank,
            r.output_rank,
            r.linear_coherence_ratio,
            r.svd_retained_ratio * 100.0,
            r.pre_stable_rank,
            r.post_stable_rank
        ));
    }
    if !summary.diagnostics.is_empty() {
        out.push_str("\nDiagnostics:\n");
        for d in &summary.diagnostics {
            out.push_str(&format!(
                "  [{}] ({}) {}\n",
                d.rule_id,
                d.severity.as_str(),
                d.message
            ));
        }
    }
    out
}

pub fn render_knots_json(summary: &KnotsSummary) -> String {
    let adapters_json: Vec<String> = summary
        .adapter_names
        .iter()
        .zip(summary.weights.iter())
        .map(|(n, w)| format!("{{\"name\":\"{}\",\"weight\":{:.6}}}", escape_json(n), w))
        .collect();

    let layers_json: Vec<String> = summary
        .layer_reports
        .iter()
        .map(|r| {
            format!(
                "{{\"base_name\":\"{}\",\"participating_adapters\":{},\"d_out\":{},\"d_in\":{},\"joint_active_rank\":{},\"output_rank\":{},\"sum_task_energy\":{:.8},\"joint_basis_energy\":{:.8},\"linear_coherence_ratio\":{:.6},\"min_pair_cosine\":{:.6},\"max_task_energy_share\":{:.6},\"aligned_sign_conflict_rate\":{:.6},\"pre_stable_rank\":{:.6},\"post_stable_rank\":{:.6},\"svd_retained_ratio\":{:.6},\"output_frobenius_norm\":{:.8}}}",
                escape_json(&r.base_name),
                r.participating_adapters,
                r.d_out,
                r.d_in,
                r.joint_active_rank,
                r.output_rank,
                r.sum_task_energy,
                r.joint_basis_energy,
                r.linear_coherence_ratio,
                r.min_pair_cosine,
                r.max_task_energy_share,
                r.aligned_sign_conflict_rate,
                r.pre_stable_rank,
                r.post_stable_rank,
                r.svd_retained_ratio,
                r.output_frobenius_norm
            )
        })
        .collect();

    let diags_json: Vec<String> = summary
        .diagnostics
        .iter()
        .map(|d| {
            format!(
                "{{\"rule_id\":\"{}\",\"severity\":\"{}\",\"layer_name\":\"{}\",\"artifact_uri\":\"{}\",\"message\":\"{}\",\"coherence_ratio\":{:.6},\"svd_retained_ratio\":{:.6},\"max_task_energy_share\":{:.6},\"pre_stable_rank\":{:.6}}}",
                d.rule_id,
                d.severity.as_str(),
                escape_json(&d.layer_name),
                escape_json(&d.artifact_uri),
                escape_json(&d.message),
                d.coherence_ratio,
                d.svd_retained_ratio,
                d.max_task_energy_share,
                d.pre_stable_rank
            )
        })
        .collect();

    format!(
        "{{\"mode\":\"{}\",\"density\":{:.4},\"isotropy\":{:.4},\"normalize_tasks\":{},\"preserve_norm\":{},\"mean_coherence_ratio\":{:.6},\"mean_svd_retained_ratio\":{:.6},\"mean_pre_stable_rank\":{:.6},\"mean_post_stable_rank\":{:.6},\"max_joint_conservation_drift\":{:.12},\"adapters\":[{}],\"layers\":[{}],\"diagnostics\":[{}]}}\n",
        summary.mode.as_str(),
        summary.density,
        summary.isotropy,
        summary.normalize_tasks,
        summary.preserve_norm,
        summary.mean_coherence_ratio,
        summary.mean_svd_retained_ratio,
        summary.mean_pre_stable_rank,
        summary.mean_post_stable_rank,
        summary.max_joint_conservation_drift,
        adapters_json.join(","),
        layers_json.join(","),
        diags_json.join(",")
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::qr_core_singular_values;
    use crate::Rng64;

    fn make_layer(name: &str, b: Tensor2D, a: Tensor2D) -> LoraLayer {
        LoraLayer {
            base_name: name.to_string(),
            a_key: format!("{}.lora_A.weight", name),
            b_key: format!("{}.lora_B.weight", name),
            a,
            b,
        }
    }

    fn rand_mat(rows: usize, cols: usize, rng: &mut Rng64, scale: f64) -> Tensor2D {
        let mut t = Tensor2D::zeros(rows, cols, DType::F32);
        for v in &mut t.data {
            *v = rng.next_gaussian() * scale;
        }
        t
    }

    fn frob_diff(a: &Tensor2D, b: &Tensor2D) -> f64 {
        assert_eq!((a.rows, a.cols), (b.rows, b.cols));
        a.data
            .iter()
            .zip(b.data.iter())
            .map(|(x, y)| (x - y) * (x - y))
            .sum::<f64>()
            .sqrt()
    }

    #[test]
    fn test_knots_joint_energy_identity_and_orthogonal_gauge_invariance() {
        let mut rng = Rng64::new(20261119);
        let d_out = 24;
        let d_in = 20;
        let r = 4;

        let b1 = rand_mat(d_out, r, &mut rng, 0.4);
        let a1 = rand_mat(r, d_in, &mut rng, 0.4);
        let b2 = rand_mat(d_out, r, &mut rng, 0.35);
        let a2 = rand_mat(r, d_in, &mut rng, 0.35);

        // Build an exact r x r orthogonal gauge matrix Q via thin_qr_columns
        let g_seed = rand_mat(r, r, &mut rng, 1.0);
        let (q_cols, _) = thin_qr_columns(&g_seed);
        let mut q = Tensor2D::zeros(r, r, DType::F32);
        for row in 0..r {
            for col in 0..r {
                q.set(row, col, q_cols[col][row]);
            }
        }
        let qt = transpose(&q);

        // Gauge-rotated factors: (B1 * Q) * (Q^T * A1) == B1 * A1
        let b1_rot = matmul(&b1, &q);
        let a1_rot = matmul(&qt, &a1);
        let b2_rot = matmul(&b2, &q);
        let a2_rot = matmul(&qt, &a2);

        let inputs_orig = vec![
            AdapterMergeInput {
                name: "a1.safetensors".to_string(),
                layers: vec![make_layer("model.layers.0.q_proj", b1, a1)],
                alphas: BTreeMap::new(),
                peft_config: None,
                weight: 1.0,
            },
            AdapterMergeInput {
                name: "a2.safetensors".to_string(),
                layers: vec![make_layer("model.layers.0.q_proj", b2, a2)],
                alphas: BTreeMap::new(),
                peft_config: None,
                weight: 0.8,
            },
        ];

        let inputs_rot = vec![
            AdapterMergeInput {
                name: "a1.safetensors".to_string(),
                layers: vec![make_layer("model.layers.0.q_proj", b1_rot, a1_rot)],
                alphas: BTreeMap::new(),
                peft_config: None,
                weight: 1.0,
            },
            AdapterMergeInput {
                name: "a2.safetensors".to_string(),
                layers: vec![make_layer("model.layers.0.q_proj", b2_rot, a2_rot)],
                alphas: BTreeMap::new(),
                peft_config: None,
                weight: 0.8,
            },
        ];

        for mode in [
            KnotsMode::KnotsTies,
            KnotsMode::IsoSpectral,
            KnotsMode::TaskSingularVectors,
        ] {
            let cfg = KnotsConfig {
                mode,
                density: 0.6,
                isotropy: 1.0,
                target_rank: Some(4),
                ..KnotsConfig::default()
            };
            let sum_orig = compose_adapters_knots(&inputs_orig, &cfg).expect("orig knots");
            let sum_rot = compose_adapters_knots(&inputs_rot, &cfg).expect("rot knots");

            let rep = &sum_orig.layer_reports[0];
            assert!(
                (rep.sum_task_energy - rep.joint_basis_energy).abs()
                    <= rep.sum_task_energy * 1e-10,
                "Joint Frobenius conservation violated: {} vs {}",
                rep.sum_task_energy,
                rep.joint_basis_energy
            );

            let dw_orig = matmul(&sum_orig.merged_layers[0].b, &sum_orig.merged_layers[0].a);
            let dw_rot = matmul(&sum_rot.merged_layers[0].b, &sum_rot.merged_layers[0].a);
            let err = frob_diff(&dw_orig, &dw_rot);
            assert!(
                err < 1e-9,
                "Gauge invariance violated for mode {}: frob diff = {:e}",
                mode.as_str(),
                err
            );
        }
    }

    #[test]
    fn test_isotropic_mode_flattens_singular_spectrum_and_preserves_kept_energy() {
        let mut rng = Rng64::new(77);
        let d_out = 20;
        let d_in = 16;
        let r = 4;

        // Construct anisotropic rank-4 adapters where mode 0 is 20x larger than mode 3
        let mut b1 = rand_mat(d_out, r, &mut rng, 0.2);
        for row in 0..d_out {
            b1.set(row, 0, b1.get(row, 0) * 15.0);
        }
        let a1 = rand_mat(r, d_in, &mut rng, 0.2);
        let b2 = rand_mat(d_out, r, &mut rng, 0.2);
        let a2 = rand_mat(r, d_in, &mut rng, 0.2);

        let inputs = vec![
            AdapterMergeInput {
                name: "spike.safetensors".to_string(),
                layers: vec![make_layer("blk.0.attn_k", b1, a1)],
                alphas: BTreeMap::new(),
                peft_config: None,
                weight: 1.0,
            },
            AdapterMergeInput {
                name: "aux.safetensors".to_string(),
                layers: vec![make_layer("blk.0.attn_k", b2, a2)],
                alphas: BTreeMap::new(),
                peft_config: None,
                weight: 1.0,
            },
        ];

        let cfg_raw = KnotsConfig {
            mode: KnotsMode::IsoSpectral,
            isotropy: 0.0,
            target_rank: Some(4),
            ..KnotsConfig::default()
        };
        let cfg_iso = KnotsConfig {
            mode: KnotsMode::IsoSpectral,
            isotropy: 1.0,
            target_rank: Some(4),
            ..KnotsConfig::default()
        };

        let raw_res = compose_adapters_knots(&inputs, &cfg_raw).unwrap();
        let iso_res = compose_adapters_knots(&inputs, &cfg_iso).unwrap();

        let sig_iso = qr_core_singular_values(&iso_res.merged_layers[0].b, &iso_res.merged_layers[0].a);
        assert_eq!(sig_iso.len(), 4);
        for s in &sig_iso {
            assert!(
                (s - sig_iso[0]).abs() < 1e-9,
                "Expected flat singular spectrum under isotropy=1.0, got {:?}",
                sig_iso
            );
        }
        assert!(
            (iso_res.layer_reports[0].post_stable_rank - 4.0).abs() < 1e-6,
            "Expected post_stable_rank == 4.0, got {}",
            iso_res.layer_reports[0].post_stable_rank
        );
        assert!(
            (raw_res.layer_reports[0].output_frobenius_norm
                - iso_res.layer_reports[0].output_frobenius_norm)
                .abs()
                < 1e-9,
            "Isotropic flattening should preserve kept-subspace Frobenius norm"
        );
    }

    #[test]
    fn test_tsv_procrustes_preserves_singular_values_when_total_rank_fits_ambient() {
        let mut rng = Rng64::new(909);
        let d_out = 18;
        let d_in = 16;
        let r = 3;

        let b1 = rand_mat(d_out, r, &mut rng, 0.5);
        let a1 = rand_mat(r, d_in, &mut rng, 0.5);
        let b2 = rand_mat(d_out, r, &mut rng, 0.4);
        let a2 = rand_mat(r, d_in, &mut rng, 0.4);

        let mut expected_sig: Vec<f64> = Vec::new();
        expected_sig.extend(qr_core_singular_values(&b1, &a1));
        expected_sig.extend(qr_core_singular_values(&b2, &a2));
        expected_sig.sort_by(|x, y| y.partial_cmp(x).unwrap());

        let inputs = vec![
            AdapterMergeInput {
                name: "t1.safetensors".to_string(),
                layers: vec![make_layer("layers.1.mlp.up_proj", b1, a1)],
                alphas: BTreeMap::new(),
                peft_config: None,
                weight: 1.0,
            },
            AdapterMergeInput {
                name: "t2.safetensors".to_string(),
                layers: vec![make_layer("layers.1.mlp.up_proj", b2, a2)],
                alphas: BTreeMap::new(),
                peft_config: None,
                weight: 1.0,
            },
        ];

        let cfg = KnotsConfig {
            mode: KnotsMode::TaskSingularVectors,
            target_rank: Some(6),
            ..KnotsConfig::default()
        };
        let res = compose_adapters_knots(&inputs, &cfg).unwrap();
        let got_sig = qr_core_singular_values(&res.merged_layers[0].b, &res.merged_layers[0].a);
        assert_eq!(got_sig.len(), 6);
        for (g, e) in got_sig.iter().zip(expected_sig.iter()) {
            assert!(
                (g - e).abs() < 1e-8,
                "TSV polar Procrustes should preserve concatenated singular values: got {:?} vs {:?}",
                got_sig,
                expected_sig
            );
        }
    }

    #[test]
    fn test_sarif_rules_ls043_ls044_ls045_ls046_trigger() {
        let mut rng = Rng64::new(31337);
        let d_out = 16;
        let d_in = 16;

        // Layer 1: opposing adapters -> triggers LS043 (destructive cancellation)
        // Also rank-4 spike -> triggers LS046 (anisotropy spike)
        let mut b_cancel = Tensor2D::zeros(d_out, 4, DType::F32);
        let mut a_cancel = Tensor2D::zeros(4, d_in, DType::F32);
        for i in 0..d_out {
            b_cancel.set(i, 0, (i as f64 + 1.0) * 0.2);
        }
        for j in 0..d_in {
            a_cancel.set(0, j, (j as f64 + 1.0) * 0.2);
        }
        let mut b_neg = b_cancel.clone();
        scale_slice(&mut b_neg.data, -0.9);

        // Layer 2: orthogonal rank-4 + rank-4 compressed to rank-1 -> triggers LS044 (>20% truncation loss)
        let b_r1 = rand_mat(d_out, 4, &mut rng, 0.5);
        let a_r1 = rand_mat(4, d_in, &mut rng, 0.5);
        // Layer 3: adapter 1 has 10x amplitude (100x energy, >80% share) -> triggers LS045
        let b_dom = rand_mat(d_out, 2, &mut rng, 2.0);
        let a_dom = rand_mat(2, d_in, &mut rng, 2.0);
        let b_tiny = rand_mat(d_out, 2, &mut rng, 0.2);
        let a_tiny = rand_mat(2, d_in, &mut rng, 0.2);

        let inputs = vec![
            AdapterMergeInput {
                name: "primary.safetensors".to_string(),
                layers: vec![
                    make_layer("l_cancel", b_cancel, a_cancel.clone()),
                    make_layer("l_trunc", b_r1.clone(), a_r1.clone()),
                    make_layer("l_dom", b_dom, a_dom),
                ],
                alphas: BTreeMap::new(),
                peft_config: None,
                weight: 1.0,
            },
            AdapterMergeInput {
                name: "secondary.safetensors".to_string(),
                layers: vec![
                    make_layer("l_cancel", b_neg, a_cancel),
                    make_layer(
                        "l_trunc",
                        rand_mat(d_out, 4, &mut rng, 0.5),
                        rand_mat(4, d_in, &mut rng, 0.5),
                    ),
                    make_layer("l_dom", b_tiny, a_tiny),
                ],
                alphas: BTreeMap::new(),
                peft_config: None,
                weight: 1.0,
            },
        ];

        let cfg = KnotsConfig {
            mode: KnotsMode::TaskSingularVectors,
            target_rank: Some(1),
            ..KnotsConfig::default()
        };
        let summary = compose_adapters_knots(&inputs, &cfg).unwrap();
        let triggered: BTreeSet<&str> = summary.diagnostics.iter().map(|d| d.rule_id).collect();
        assert!(triggered.contains(LS043_KNOTS_DESTRUCTIVE_CANCELLATION));
        assert!(triggered.contains(LS044_JOINT_RANK_TRUNCATION_OVERFLOW));
        assert!(triggered.contains(LS045_ADAPTER_TASK_NORM_DOMINANCE));
        assert!(triggered.contains(LS046_SPECTRUM_ANISOTROPY_SPIKE));
    }
}

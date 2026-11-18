//! LoRA-GA: Low-Rank Adaptation with Gradient Approximation (Wang et al., NeurIPS 2024)
//! spectral initialization optimizer and SARIF diagnostics (`LS039`, `LS040`).
//!
//! Standard LoRA initializes `A_0 ~ Kaiming` and `B_0 = 0`, which yields `nabla_A L = 0` on
//! step 1 and restricts the initial weight update `Delta(W_0 + s * B * A)` to a random `r`-dim
//! projection of the full-parameter gradient `G_0 = nabla_W L(W_0)`.
//!
//! LoRA-GA decomposes `G_0 = U * Sigma * V^T` and assigns disjoint singular subspaces:
//!   - `B_0` (`d_out x r`) from top left singular vectors `U[:, 0..r]`
//!   - `A_0` (`r x d_in`) from disjoint right singular vectors `V[:, r..2r]^T`
//! Because `U[:, 0..r]^T * G_0 * V[:, r..2r] = 0` identically, the step-1 tangent update
//! `B_0 * B_0^T * G_0 + G_0 * A_0^T * A_0` spans the exact top-`2r` singular components
//! `sum_{i=1}^{2r} sigma_i * u_i * v_i^T` without double-counting, doubling first-step
//! gradient subspace capture from rank `r` to rank `2r`.
//! To preserve exact base-model outputs at `t = 0`, `W_res = W_0 - s * B_0 * A_0` (or companion
//! `.pissa_init` reference weights) compensates the non-zero initial product `B_0 * A_0`.

use std::collections::BTreeMap;

use crate::linalg::{
    axpy_slice, dot_slice, jacobi_svd, matmul, multiply_q_cols_by_small, qr_core_svd_truncated,
    scale_slice, thin_qr_columns, transpose, SvdDecomposition,
};
use crate::merge_check::resolve_effective_layer_scale;
use crate::safetensors::{
    convert_layers_key_format, DType, LoraLayer, PeftAdapterConfig, Tensor2D,
};
use crate::Rng64;

/// SARIF v2.1.0 rule IDs for LoRA-GA spectral initialization.
pub const LS039_INSUFFICIENT_GRAD_RANK: &str = "LS039";
pub const LS040_UNCOMPENSATED_BASE_DRIFT: &str = "LS040";

/// Scaling schedule for LoRA-GA factor initialization (Wang et al., NeurIPS 2024 Sec. 3.3).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoraGaScaleMode {
    /// Variance-stabilized scaling (`A_0 = d_out^{1/4} / sqrt(gamma) * V_{r..2r}^T`,
    /// `B_0 = d_in^{-1/4} / sqrt(gamma) * U_{0..r}`). Default in Wang et al. (2024).
    Stable,
    /// Unit-orthonormal directional basis (`A_0 = V_{r..2r}^T`, `B_0 = U_{0..r}`).
    Direction,
    /// Gradient-descent norm-matched basis scaling `eta_scale = sqrt(||G_{1..2r}||_F / sqrt(2r))`.
    Gd,
    /// Singular-value weighted basis (`B_0[:, j] = sqrt(sigma_j) u_j`, `A_0[j, :] = sqrt(sigma_{r+j}) v_{r+j}^T`).
    Svd,
}

impl LoraGaScaleMode {
    pub fn as_str(&self) -> &'static str {
        match self {
            LoraGaScaleMode::Stable => "stable",
            LoraGaScaleMode::Direction => "direction",
            LoraGaScaleMode::Gd => "gd",
            LoraGaScaleMode::Svd => "svd",
        }
    }

    pub fn from_name(s: &str) -> Result<Self, String> {
        match s.trim().to_ascii_lowercase().as_str() {
            "stable" | "lora-ga-stable" => Ok(LoraGaScaleMode::Stable),
            "direction" | "unit" | "orthonormal" => Ok(LoraGaScaleMode::Direction),
            "gd" | "grad" | "gradient" => Ok(LoraGaScaleMode::Gd),
            "svd" | "spectral" => Ok(LoraGaScaleMode::Svd),
            other => Err(format!(
                "Unsupported LoRA-GA scale mode '{}': expected stable, direction, gd, or svd",
                other
            )),
        }
    }
}

/// Configuration for `optimize_lora_ga_init`.
#[derive(Debug, Clone)]
pub struct LoraGaConfig {
    /// Target LoRA rank `r >= 1` (captures top `2r` gradient singular modes on step 1).
    pub target_rank: usize,
    /// Factor scaling schedule.
    pub scale_mode: LoraGaScaleMode,
    /// Stability constant `gamma > 0` for `LoraGaScaleMode::Stable` (default `16.0`).
    pub stable_gamma: f64,
    /// Nominal PEFT `lora_alpha` for forward scaling `s = alpha / r` (or `alpha / sqrt(r)` if `use_rslora`).
    pub lora_alpha: f64,
    /// Whether `use_rslora` (`alpha / sqrt(r)`) is active when computing forward perturbation `s * B_0 * A_0`.
    pub use_rslora: bool,
    /// Output key naming convention (`peft`, `kohya`, or `diffusers`).
    pub key_format: String,
    /// Output storage dtype (`F32`, `F16`, or `BF16`).
    pub output_dtype: Option<DType>,
    /// Emit companion `.pissa_init` reference layers so downstream tools can verify/subtract `B_0 * A_0`.
    pub emit_init_reference: bool,
    /// True if base-weight compensation (`--emit-init` or `--out-base`) is active for this run.
    pub base_compensated: bool,
    /// Deterministic seed for randomized subspace-iteration SVD on large dense gradients.
    pub seed: u64,
}

impl Default for LoraGaConfig {
    fn default() -> Self {
        Self {
            target_rank: 8,
            scale_mode: LoraGaScaleMode::Stable,
            stable_gamma: 16.0,
            lora_alpha: 16.0,
            use_rslora: false,
            key_format: "peft".to_string(),
            output_dtype: None,
            emit_init_reference: false,
            base_compensated: false,
            seed: 42,
        }
    }
}

/// Source gradient representation for LoRA-GA initialization.
pub enum GradientSource<'a> {
    /// Warm-up LoRA checkpoint or low-rank gradient accumulator (`G_l = s_l * B_l * A_l`).
    LowRank {
        layers: &'a [LoraLayer],
        alphas: &'a BTreeMap<String, f64>,
        peft_config: Option<&'a PeftAdapterConfig>,
    },
    /// Dense 2D gradient tensors (`*.grad`, `*.weight`, or dense checkpoint difference).
    Dense {
        tensors: &'a BTreeMap<String, Tensor2D>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoraGaSeverity {
    Error,
    Warning,
}

impl LoraGaSeverity {
    pub fn as_str(&self) -> &'static str {
        match self {
            LoraGaSeverity::Error => "error",
            LoraGaSeverity::Warning => "warning",
        }
    }
}

#[derive(Debug, Clone)]
pub struct LoraGaDiagnostic {
    pub rule_id: &'static str,
    pub severity: LoraGaSeverity,
    pub layer_name: String,
    pub artifact_uri: String,
    pub message: String,
    pub nonzero_grad_modes: usize,
    pub required_2r_modes: usize,
    pub initial_perturbation_norm: f64,
}

#[derive(Debug, Clone)]
pub struct LoraGaLayerReport {
    pub base_name: String,
    pub d_out: usize,
    pub d_in: usize,
    pub target_rank: usize,
    pub grad_frobenius_norm: f64,
    pub grad_spectral_norm: f64,
    pub grad_effective_rank: f64,
    pub nonzero_grad_modes: usize,
    /// Fraction of `||G||_F^2` in top `r` singular modes (standard rank-`r` static ceiling).
    pub top_r_energy_ratio: f64,
    /// Fraction of `||G||_F^2` captured on step 1 by LoRA-GA disjoint `[0..r]` + `[r..2r]` basis (`sum_{i=1..2r} sigma_i^2 / ||G||_F^2`).
    pub lora_ga_2r_alignment_ratio: f64,
    /// Expected step-1 gradient energy fraction of random Kaiming-A / Zero-B LoRA (`r / min(d_out, d_in)`).
    pub kaiming_expected_ratio: f64,
    /// Multiplier gain of LoRA-GA step-1 captured gradient energy over random Kaiming init.
    pub alignment_gain_vs_kaiming: f64,
    /// Frobenius norm `||s * B_0 * A_0||_F` of the initial forward perturbation prior to base compensation.
    pub initial_perturbation_norm: f64,
    /// Verified orthogonality residual `||U_{0..r}^T * G * V_{r..2r}||_F / ||G||_F` (0 to machine precision).
    pub cross_term_leakage: f64,
}

#[derive(Debug, Clone)]
pub struct LoraGaSummary {
    pub source_uri: String,
    pub scale_mode: LoraGaScaleMode,
    pub target_rank: usize,
    pub effective_forward_scale: f64,
    pub base_compensated: bool,
    pub layers: Vec<LoraLayer>,
    pub init_reference_layers: Vec<LoraLayer>,
    pub reports: Vec<LoraGaLayerReport>,
    pub diagnostics: Vec<LoraGaDiagnostic>,
    pub mean_top_r_energy_ratio: f64,
    pub mean_lora_ga_2r_alignment_ratio: f64,
    pub mean_kaiming_expected_ratio: f64,
    pub mean_alignment_gain_vs_kaiming: f64,
    pub total_initial_perturbation_norm: f64,
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

fn strip_grad_suffix(key: &str) -> String {
    for suffix in [".weight.grad", ".grad", ".weight"] {
        if let Some(base) = key.strip_suffix(suffix) {
            return base.to_string();
        }
    }
    key.to_string()
}

/// Orthonormal completion vector in `R^dim` orthogonal to all columns in `existing`.
fn find_orthogonal_unit_vector(dim: usize, existing: &[Vec<f64>], start_hint: usize) -> Vec<f64> {
    let mut cand = vec![0.0; dim];
    for offset in 0..dim {
        let idx = (start_hint + offset) % dim;
        cand.fill(0.0);
        cand[idx] = 1.0;
        for prev in existing {
            let proj = dot_slice(prev, &cand);
            axpy_slice(&mut cand, -proj, prev);
        }
        for prev in existing {
            let corr = dot_slice(prev, &cand);
            if corr.abs() > 1e-16 {
                axpy_slice(&mut cand, -corr, prev);
            }
        }
        let n = dot_slice(&cand, &cand).sqrt();
        if n > 1e-5 {
            scale_slice(&mut cand, 1.0 / n);
            return cand;
        }
    }
    cand.fill(0.0);
    cand
}

/// 3-pass QR-orthonormalized randomized subspace-iteration SVD returning top `keep` singular modes
/// plus total Frobenius norm squared `||M||_F^2` of dense matrix `m` (`rows x cols`).
fn dense_truncated_svd(m: &Tensor2D, keep: usize, seed: u64) -> (SvdDecomposition, f64) {
    let total_frob_sq = m.frobenius_norm_sq();
    let min_dim = m.rows.min(m.cols);
    let target_k = keep.min(min_dim).max(1);

    if min_dim <= 128 || target_k + 12 >= min_dim {
        return (jacobi_svd(m), total_frob_sq);
    }

    let oversampled = (target_k + 8).min(min_dim);
    let mut rng = Rng64::new(seed);
    let mt = transpose(m);

    // Sample Gaussian probe Omega (cols x oversampled)
    let mut omega = Tensor2D::zeros(m.cols, oversampled, DType::F32);
    let inv_sqrt = 1.0 / (oversampled as f64).sqrt();
    for v in &mut omega.data {
        *v = rng.next_gaussian() * inv_sqrt;
    }

    // Y = M * Omega, orthonormalize Q
    let mut y = matmul(m, &omega);
    let (mut q_cols, _) = thin_qr_columns(&y);

    // 2 power-iteration passes with QR stabilization for sharp singular gap convergence
    for _ in 0..2 {
        let mut q_mat = Tensor2D::zeros(m.rows, oversampled, DType::F32);
        for r in 0..m.rows {
            for c in 0..oversampled {
                q_mat.set(r, c, q_cols[c][r]);
            }
        }
        let z = matmul(&mt, &q_mat);
        let (qz_cols, _) = thin_qr_columns(&z);
        let mut qz_mat = Tensor2D::zeros(m.cols, oversampled, DType::F32);
        for r in 0..m.cols {
            for c in 0..oversampled {
                qz_mat.set(r, c, qz_cols[c][r]);
            }
        }
        y = matmul(m, &qz_mat);
        let (new_q, _) = thin_qr_columns(&y);
        q_cols = new_q;
    }

    // Project B_small = Q^T * M (oversampled x cols), then thin SVD on B_small^T (cols x oversampled)
    let mut qt = Tensor2D::zeros(oversampled, m.rows, DType::F32);
    for c in 0..oversampled {
        for r in 0..m.rows {
            qt.set(c, r, q_cols[c][r]);
        }
    }
    let b_small = matmul(&qt, m);
    let b_small_svd = jacobi_svd(&b_small);

    let k_out = target_k.min(b_small_svd.u.cols);
    let u_full = multiply_q_cols_by_small(&q_cols, m.rows, &b_small_svd.u, k_out, m.dtype);
    let mut v_full = Tensor2D::zeros(m.cols, k_out, m.dtype);
    for r in 0..m.cols {
        for c in 0..k_out {
            v_full.set(r, c, b_small_svd.v.get(r, c));
        }
    }
    let mut sigma_out = b_small_svd.sigma;
    sigma_out.truncate(k_out);

    (
        SvdDecomposition {
            u: u_full,
            sigma: sigma_out,
            v: v_full,
        },
        total_frob_sq,
    )
}

fn von_neumann_effective_rank(sigma: &[f64], total_energy: f64) -> f64 {
    if total_energy <= 1e-24 || sigma.is_empty() {
        return 0.0;
    }
    let mut entropy = 0.0;
    for &s in sigma {
        let e = s * s;
        if e > 1e-28 {
            let p = (e / total_energy).clamp(0.0, 1.0);
            if p > 0.0 {
                entropy -= p * p.ln();
            }
        }
    }
    entropy.exp()
}

/// Optimize LoRA-GA spectral initializations across all layers in `source`.
pub fn optimize_lora_ga_init(
    source_uri: &str,
    source: GradientSource<'_>,
    cfg: &LoraGaConfig,
) -> Result<LoraGaSummary, String> {
    if cfg.target_rank == 0 {
        return Err("target_rank must be >= 1".to_string());
    }
    if !cfg.stable_gamma.is_finite() || cfg.stable_gamma <= 0.0 {
        return Err("stable_gamma must be finite and > 0".to_string());
    }
    if !cfg.lora_alpha.is_finite() || cfg.lora_alpha <= 0.0 {
        return Err("lora_alpha must be finite and > 0".to_string());
    }

    let r = cfg.target_rank;
    let denom = if cfg.use_rslora {
        (r as f64).sqrt()
    } else {
        r as f64
    };
    let eff_scale = cfg.lora_alpha / denom;

    struct PreparedGrad {
        base_name: String,
        d_out: usize,
        d_in: usize,
        dtype: DType,
        svd: SvdDecomposition,
        full_sigma: Vec<f64>,
        total_energy: f64,
        has_non_finite: bool,
    }

    let mut prepared: Vec<PreparedGrad> = Vec::new();

    match source {
        GradientSource::LowRank {
            layers,
            alphas,
            peft_config,
        } => {
            if layers.is_empty() {
                return Err("Gradient source contains zero LoRA layers".to_string());
            }
            for l in layers {
                let s_in = resolve_effective_layer_scale(l, alphas, peft_config).abs();
                let has_non_finite = l.a.data.iter().any(|x| !x.is_finite())
                    || l.b.data.iter().any(|x| !x.is_finite());
                if has_non_finite {
                    prepared.push(PreparedGrad {
                        base_name: l.base_name.clone(),
                        d_out: l.d_out(),
                        d_in: l.d_in(),
                        dtype: cfg.output_dtype.unwrap_or(l.a.dtype),
                        svd: SvdDecomposition {
                            u: Tensor2D::zeros(l.d_out(), 1, DType::F32),
                            sigma: vec![0.0],
                            v: Tensor2D::zeros(l.d_in(), 1, DType::F32),
                        },
                        full_sigma: vec![0.0],
                        total_energy: 0.0,
                        has_non_finite: true,
                    });
                    continue;
                }
                let need_k = (2 * r).min(l.rank()).max(1);
                let mut svd = qr_core_svd_truncated(&l.b, &l.a, need_k);
                for s in &mut svd.sigma {
                    *s *= s_in;
                }
                let full_sigma = svd.sigma.clone();
                let total_energy: f64 = full_sigma.iter().map(|&s| s * s).sum();
                prepared.push(PreparedGrad {
                    base_name: l.base_name.clone(),
                    d_out: l.d_out(),
                    d_in: l.d_in(),
                    dtype: cfg.output_dtype.unwrap_or(l.a.dtype),
                    svd,
                    full_sigma,
                    total_energy,
                    has_non_finite: false,
                });
            }
        }
        GradientSource::Dense { tensors } => {
            for (idx, (key, mat)) in tensors.iter().enumerate() {
                if mat.rows < 2 || mat.cols < 2 {
                    continue;
                }
                let base_name = strip_grad_suffix(key);
                let has_non_finite = mat.data.iter().any(|x| !x.is_finite());
                let out_dtype = cfg.output_dtype.unwrap_or(match mat.dtype {
                    DType::I8 | DType::U8 => DType::F16,
                    other => other,
                });
                if has_non_finite {
                    prepared.push(PreparedGrad {
                        base_name,
                        d_out: mat.rows,
                        d_in: mat.cols,
                        dtype: out_dtype,
                        svd: SvdDecomposition {
                            u: Tensor2D::zeros(mat.rows, 1, DType::F32),
                            sigma: vec![0.0],
                            v: Tensor2D::zeros(mat.cols, 1, DType::F32),
                        },
                        full_sigma: vec![0.0],
                        total_energy: 0.0,
                        has_non_finite: true,
                    });
                    continue;
                }
                let need_k = (2 * r).min(mat.rows.min(mat.cols)).max(1);
                let (svd, total_energy) =
                    dense_truncated_svd(mat, need_k, cfg.seed.wrapping_add(idx as u64));
                let full_sigma = svd.sigma.clone();
                prepared.push(PreparedGrad {
                    base_name,
                    d_out: mat.rows,
                    d_in: mat.cols,
                    dtype: out_dtype,
                    svd,
                    full_sigma,
                    total_energy,
                    has_non_finite: false,
                });
            }
            if prepared.is_empty() {
                return Err("No 2D gradient tensors (rows >= 2, cols >= 2) found in input".to_string());
            }
        }
    }

    let mut raw_layers = Vec::with_capacity(prepared.len());
    let mut reports = Vec::with_capacity(prepared.len());
    let mut diagnostics = Vec::new();
    let mut total_perturbation_sq = 0.0;

    for item in prepared {
        let d_out = item.d_out;
        let d_in = item.d_in;
        let min_dim = d_out.min(d_in);
        if r > min_dim {
            return Err(format!(
                "Layer '{}' has min(d_out, d_in) = {} < target_rank = {}",
                item.base_name, min_dim, r
            ));
        }

        if item.has_non_finite {
            diagnostics.push(LoraGaDiagnostic {
                rule_id: LS039_INSUFFICIENT_GRAD_RANK,
                severity: LoraGaSeverity::Error,
                layer_name: item.base_name.clone(),
                artifact_uri: source_uri.to_string(),
                message: format!(
                    "Layer '{}' contains NaN or Inf gradient entries; initialized zeroed fallback factors.",
                    item.base_name
                ),
                nonzero_grad_modes: 0,
                required_2r_modes: 2 * r,
                initial_perturbation_norm: 0.0,
            });
            let zero_layer = LoraLayer {
                base_name: item.base_name.clone(),
                a_key: format!("{}.lora_A.weight", item.base_name),
                b_key: format!("{}.lora_B.weight", item.base_name),
                a: Tensor2D::zeros(r, d_in, item.dtype),
                b: Tensor2D::zeros(d_out, r, item.dtype),
            };
            raw_layers.push(zero_layer);
            reports.push(LoraGaLayerReport {
                base_name: item.base_name,
                d_out,
                d_in,
                target_rank: r,
                grad_frobenius_norm: 0.0,
                grad_spectral_norm: 0.0,
                grad_effective_rank: 0.0,
                nonzero_grad_modes: 0,
                top_r_energy_ratio: 0.0,
                lora_ga_2r_alignment_ratio: 0.0,
                kaiming_expected_ratio: (r as f64) / (min_dim as f64),
                alignment_gain_vs_kaiming: 0.0,
                initial_perturbation_norm: 0.0,
                cross_term_leakage: 0.0,
            });
            continue;
        }

        let sigma_max = item.full_sigma.first().copied().unwrap_or(0.0);
        let rank_tol = (sigma_max * 1e-12).max(1e-15);
        let nonzero_modes = item.full_sigma.iter().filter(|&&s| s > rank_tol).count();
        let grad_frob = item.total_energy.max(0.0).sqrt();
        let eff_rank = von_neumann_effective_rank(&item.full_sigma, item.total_energy);

        // Build orthonormal U_B (r columns in R^{d_out}) from singular modes [0..r]
        let mut u_b_cols: Vec<Vec<f64>> = Vec::with_capacity(r);
        for j in 0..r {
            if j < item.svd.u.cols && j < nonzero_modes {
                let mut col = vec![0.0; d_out];
                for row in 0..d_out {
                    col[row] = item.svd.u.get(row, j);
                }
                for prev in &u_b_cols {
                    let proj = dot_slice(prev, &col);
                    axpy_slice(&mut col, -proj, prev);
                }
                let n = dot_slice(&col, &col).sqrt();
                if n > 1e-12 {
                    scale_slice(&mut col, 1.0 / n);
                    u_b_cols.push(col);
                    continue;
                }
            }
            let fallback = find_orthogonal_unit_vector(d_out, &u_b_cols, j);
            u_b_cols.push(fallback);
        }

        // Build orthonormal V_A (r columns in R^{d_in}) from disjoint singular modes [r..2r]
        // while keeping V_A strictly orthogonal to the leading V[:, 0..min(r, v.cols)] modes
        // so U_{0..r}^T * G * V_A == 0 holds even when completing rank-deficient tails.
        let mut v_leading_and_a: Vec<Vec<f64>> = Vec::new();
        let lead_count = r.min(item.svd.v.cols).min(nonzero_modes);
        for j in 0..lead_count {
            let mut col = vec![0.0; d_in];
            for row in 0..d_in {
                col[row] = item.svd.v.get(row, j);
            }
            let n = dot_slice(&col, &col).sqrt();
            if n > 1e-12 {
                scale_slice(&mut col, 1.0 / n);
                v_leading_and_a.push(col);
            }
        }

        let mut v_a_cols: Vec<Vec<f64>> = Vec::with_capacity(r);
        for j in 0..r {
            let src_idx = r + j;
            if src_idx < item.svd.v.cols && src_idx < nonzero_modes {
                let mut col = vec![0.0; d_in];
                for row in 0..d_in {
                    col[row] = item.svd.v.get(row, src_idx);
                }
                for prev in &v_leading_and_a {
                    let proj = dot_slice(prev, &col);
                    axpy_slice(&mut col, -proj, prev);
                }
                let n = dot_slice(&col, &col).sqrt();
                if n > 1e-12 {
                    scale_slice(&mut col, 1.0 / n);
                    v_leading_and_a.push(col.clone());
                    v_a_cols.push(col);
                    continue;
                }
            }
            let fallback = find_orthogonal_unit_vector(d_in, &v_leading_and_a, r + j);
            v_leading_and_a.push(fallback.clone());
            v_a_cols.push(fallback);
        }

        // Verify cross-term leakage ||U_B^T * G * V_A||_F / ||G||_F in O(r * k) spectral coordinates
        let active_k = item.svd.u.cols.min(item.svd.v.cols).min(item.full_sigma.len());
        let mut cross_sq = 0.0;
        if item.total_energy > 1e-24 && active_k > 0 {
            // C = (U_B^T * U_active) * diag(sigma) * (V_active^T * V_A)
            let mut left_proj = vec![vec![0.0; active_k]; r];
            for bi in 0..r {
                for k_idx in 0..active_k {
                    let mut acc = 0.0;
                    for row in 0..d_out {
                        acc += u_b_cols[bi][row] * item.svd.u.get(row, k_idx);
                    }
                    left_proj[bi][k_idx] = acc * item.full_sigma[k_idx];
                }
            }
            let mut right_proj = vec![vec![0.0; r]; active_k];
            for k_idx in 0..active_k {
                for aj in 0..r {
                    let mut acc = 0.0;
                    for row in 0..d_in {
                        acc += item.svd.v.get(row, k_idx) * v_a_cols[aj][row];
                    }
                    right_proj[k_idx][aj] = acc;
                }
            }
            for bi in 0..r {
                for aj in 0..r {
                    let mut entry = 0.0;
                    for k_idx in 0..active_k {
                        entry += left_proj[bi][k_idx] * right_proj[k_idx][aj];
                    }
                    cross_sq += entry * entry;
                }
            }
        }
        let cross_term_leakage = if grad_frob > 1e-15 {
            cross_sq.sqrt() / grad_frob
        } else {
            0.0
        };

        // Compute captured step-1 energy ratios
        let top_r_e: f64 = item.full_sigma.iter().take(r).map(|&s| s * s).sum();
        let top_2r_e: f64 = item
            .full_sigma
            .iter()
            .take((2 * r).min(nonzero_modes))
            .map(|&s| s * s)
            .sum();
        let top_r_ratio = if item.total_energy > 1e-24 {
            (top_r_e / item.total_energy).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let lora_ga_2r_ratio = if item.total_energy > 1e-24 {
            (top_2r_e / item.total_energy).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let kaiming_ratio = ((r as f64) / (min_dim as f64)).clamp(0.0, 1.0);
        let gain_vs_kaiming = if kaiming_ratio > 1e-15 {
            lora_ga_2r_ratio / kaiming_ratio
        } else {
            0.0
        };

        // Apply factor scaling schedule to construct B_0 (d_out x r) and A_0 (r x d_in)
        let mut b_mat = Tensor2D::zeros(d_out, r, item.dtype);
        let mut a_mat = Tensor2D::zeros(r, d_in, item.dtype);

        let gd_scale = if top_2r_e > 1e-24 {
            (top_2r_e.sqrt() / ((2 * r) as f64).sqrt()).sqrt()
        } else {
            1.0
        };
        let stable_inv_gamma = 1.0 / cfg.stable_gamma.sqrt();
        let scale_a_stable = (d_out as f64).powf(0.25) * stable_inv_gamma;
        let scale_b_stable = (d_in as f64).powf(-0.25) * stable_inv_gamma;

        for j in 0..r {
            let (sb, sa) = match cfg.scale_mode {
                LoraGaScaleMode::Stable => (scale_b_stable, scale_a_stable),
                LoraGaScaleMode::Direction => (1.0, 1.0),
                LoraGaScaleMode::Gd => (gd_scale, gd_scale),
                LoraGaScaleMode::Svd => {
                    let s_b = item
                        .full_sigma
                        .get(j)
                        .copied()
                        .unwrap_or(0.0)
                        .max(0.0)
                        .sqrt();
                    let s_a = item
                        .full_sigma
                        .get(r + j)
                        .copied()
                        .unwrap_or_else(|| s_b * 1e-2)
                        .max(0.0)
                        .sqrt();
                    (s_b, s_a)
                }
            };
            for row in 0..d_out {
                b_mat.set(row, j, sb * u_b_cols[j][row]);
            }
            for col in 0..d_in {
                a_mat.set(j, col, sa * v_a_cols[j][col]);
            }
        }

        // Exact Frobenius norm ||s * B_0 * A_0||_F:
        // Since columns of U_B and V_A are orthonormal, B_0 = U_B * diag(sb), A_0 = diag(sa) * V_A^T,
        // so ||s * B_0 * A_0||_F^2 = s^2 * sum_{j=0..r-1} (sb_j * sa_j)^2.
        let mut ba_frob_sq = 0.0;
        for j in 0..r {
            let mut b_col_sq = 0.0;
            for row in 0..d_out {
                let v = b_mat.get(row, j);
                b_col_sq += v * v;
            }
            let mut a_row_sq = 0.0;
            for col in 0..d_in {
                let v = a_mat.get(j, col);
                a_row_sq += v * v;
            }
            ba_frob_sq += b_col_sq * a_row_sq;
        }
        let init_perturb_norm = eff_scale * ba_frob_sq.sqrt();
        total_perturbation_sq += init_perturb_norm * init_perturb_norm;

        // Emit SARIF LS039 if gradient rank < 2r
        if nonzero_modes == 0 {
            diagnostics.push(LoraGaDiagnostic {
                rule_id: LS039_INSUFFICIENT_GRAD_RANK,
                severity: LoraGaSeverity::Error,
                layer_name: item.base_name.clone(),
                artifact_uri: source_uri.to_string(),
                message: format!(
                    "Layer '{}' has zero gradient Frobenius norm; LoRA-GA fell back to deterministic orthogonal basis with 0% gradient alignment.",
                    item.base_name
                ),
                nonzero_grad_modes: 0,
                required_2r_modes: 2 * r,
                initial_perturbation_norm: init_perturb_norm,
            });
        } else if nonzero_modes < 2 * r {
            diagnostics.push(LoraGaDiagnostic {
                rule_id: LS039_INSUFFICIENT_GRAD_RANK,
                severity: LoraGaSeverity::Warning,
                layer_name: item.base_name.clone(),
                artifact_uri: source_uri.to_string(),
                message: format!(
                    "Layer '{}' gradient has only {} non-zero singular modes (< 2*r = {}); {} disjoint A_0 row(s) lie in the gradient nullspace.",
                    item.base_name,
                    nonzero_modes,
                    2 * r,
                    (2 * r).saturating_sub(nonzero_modes).min(r)
                ),
                nonzero_grad_modes: nonzero_modes,
                required_2r_modes: 2 * r,
                initial_perturbation_norm: init_perturb_norm,
            });
        }

        // Emit SARIF LS040 if non-zero initial perturbation is not compensated
        if !cfg.base_compensated && !cfg.emit_init_reference && init_perturb_norm > 1e-6 {
            diagnostics.push(LoraGaDiagnostic {
                rule_id: LS040_UNCOMPENSATED_BASE_DRIFT,
                severity: LoraGaSeverity::Warning,
                layer_name: item.base_name.clone(),
                artifact_uri: source_uri.to_string(),
                message: format!(
                    "Layer '{}' has non-zero initial forward perturbation ||s*B_0*A_0||_F = {:.6} without base-weight residual subtraction (--emit-init or --base/--out-base).",
                    item.base_name, init_perturb_norm
                ),
                nonzero_grad_modes: nonzero_modes,
                required_2r_modes: 2 * r,
                initial_perturbation_norm: init_perturb_norm,
            });
        }

        raw_layers.push(LoraLayer {
            base_name: item.base_name.clone(),
            a_key: format!("{}.lora_A.weight", item.base_name),
            b_key: format!("{}.lora_B.weight", item.base_name),
            a: a_mat,
            b: b_mat,
        });

        reports.push(LoraGaLayerReport {
            base_name: item.base_name,
            d_out,
            d_in,
            target_rank: r,
            grad_frobenius_norm: grad_frob,
            grad_spectral_norm: sigma_max,
            grad_effective_rank: eff_rank,
            nonzero_grad_modes: nonzero_modes,
            top_r_energy_ratio: top_r_ratio,
            lora_ga_2r_alignment_ratio: lora_ga_2r_ratio,
            kaiming_expected_ratio: kaiming_ratio,
            alignment_gain_vs_kaiming: gain_vs_kaiming,
            initial_perturbation_norm: init_perturb_norm,
            cross_term_leakage,
        });
    }

    let init_reference_layers = if cfg.emit_init_reference {
        raw_layers
            .iter()
            .map(|l| LoraLayer {
                base_name: l.base_name.clone(),
                a_key: format!("base_model.model.{}.lora_A.pissa_init.weight", l.base_name),
                b_key: format!("base_model.model.{}.lora_B.pissa_init.weight", l.base_name),
                a: l.a.clone(),
                b: l.b.clone(),
            })
            .collect()
    } else {
        Vec::new()
    };
    convert_layers_key_format(&mut raw_layers, &cfg.key_format)?;

    let n = reports.len().max(1) as f64;
    let mean_top_r = reports.iter().map(|x| x.top_r_energy_ratio).sum::<f64>() / n;
    let mean_2r = reports
        .iter()
        .map(|x| x.lora_ga_2r_alignment_ratio)
        .sum::<f64>()
        / n;
    let mean_kaiming = reports
        .iter()
        .map(|x| x.kaiming_expected_ratio)
        .sum::<f64>()
        / n;
    let mean_gain = reports
        .iter()
        .map(|x| x.alignment_gain_vs_kaiming)
        .sum::<f64>()
        / n;

    Ok(LoraGaSummary {
        source_uri: source_uri.to_string(),
        scale_mode: cfg.scale_mode,
        target_rank: r,
        effective_forward_scale: eff_scale,
        base_compensated: cfg.base_compensated || cfg.emit_init_reference,
        layers: raw_layers,
        init_reference_layers,
        reports,
        diagnostics,
        mean_top_r_energy_ratio: mean_top_r,
        mean_lora_ga_2r_alignment_ratio: mean_2r,
        mean_kaiming_expected_ratio: mean_kaiming,
        mean_alignment_gain_vs_kaiming: mean_gain,
        total_initial_perturbation_norm: total_perturbation_sq.sqrt(),
    })
}

/// Subtract `effective_scale * B_0 * A_0` from matching 2D dense base weights `W_0` so that
/// `W_res + effective_scale * B_0 * A_0 == W_0` holds identically at fine-tuning step `t = 0`.
pub fn apply_lora_ga_base_compensation(
    base_tensors: &BTreeMap<String, Tensor2D>,
    init_layers: &[LoraLayer],
    effective_scale: f64,
) -> Result<(BTreeMap<String, Tensor2D>, usize), String> {
    let mut out = base_tensors.clone();
    let mut updated_count = 0usize;

    for layer in init_layers {
        let candidates = [
            format!("{}.weight", layer.base_name),
            format!("model.{}.weight", layer.base_name),
            format!("base_model.model.{}.weight", layer.base_name),
            layer.base_name.clone(),
        ];
        let matched_key = candidates.iter().find(|k| out.contains_key(k.as_str())).cloned();
        let Some(w_key) = matched_key else {
            continue;
        };
        let w_mat = out.get_mut(&w_key).unwrap();
        if w_mat.rows != layer.d_out() || w_mat.cols != layer.d_in() {
            return Err(format!(
                "Base weight '{}' shape ({}x{}) mismatches LoRA-GA layer '{}' ({}x{})",
                w_key,
                w_mat.rows,
                w_mat.cols,
                layer.base_name,
                layer.d_out(),
                layer.d_in()
            ));
        }
        let delta = matmul(&layer.b, &layer.a);
        for (dst, &dv) in w_mat.data.iter_mut().zip(delta.data.iter()) {
            *dst -= effective_scale * dv;
        }
        updated_count += 1;
    }

    Ok((out, updated_count))
}

pub fn render_lora_ga_table(summary: &LoraGaSummary) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "LoRA-GA Spectral Initialization Optimizer (mode={}, rank={}, scale={:.4}, compensated={})\n",
        summary.scale_mode.as_str(),
        summary.target_rank,
        summary.effective_forward_scale,
        summary.base_compensated
    ));
    out.push_str(&format!(
        "{:<38} {:>11} {:>7} {:>8} {:>9} {:>9} {:>8} {:>10}\n",
        "LAYER", "SHAPE", "MODES", "TOP_R%", "GA_2R%", "KAIMING%", "GAIN", "||s*BA||_F"
    ));
    out.push_str(&"-".repeat(108));
    out.push('\n');

    for r in &summary.reports {
        let short_name = if r.base_name.len() > 38 {
            format!("...{}", &r.base_name[r.base_name.len() - 35..])
        } else {
            r.base_name.clone()
        };
        out.push_str(&format!(
            "{:<38} {:>5}x{:<5} {:>7} {:>7.2}% {:>8.2}% {:>8.2}% {:>7.1}x {:>10.5}\n",
            short_name,
            r.d_out,
            r.d_in,
            r.nonzero_grad_modes,
            r.top_r_energy_ratio * 100.0,
            r.lora_ga_2r_alignment_ratio * 100.0,
            r.kaiming_expected_ratio * 100.0,
            r.alignment_gain_vs_kaiming,
            r.initial_perturbation_norm
        ));
    }

    out.push_str(&"-".repeat(108));
    out.push('\n');
    out.push_str(&format!(
        "SUMMARY ({} layers): mean top-r={:.2}%, LoRA-GA 2r alignment={:.2}%, Kaiming expected={:.2}% ({:.1}x gain), total ||s*B_0*A_0||_F={:.6}, diagnostics={}\n",
        summary.reports.len(),
        summary.mean_top_r_energy_ratio * 100.0,
        summary.mean_lora_ga_2r_alignment_ratio * 100.0,
        summary.mean_kaiming_expected_ratio * 100.0,
        summary.mean_alignment_gain_vs_kaiming,
        summary.total_initial_perturbation_norm,
        summary.diagnostics.len()
    ));
    out
}

pub fn render_lora_ga_json(summary: &LoraGaSummary) -> String {
    let layers_json: Vec<String> = summary
        .reports
        .iter()
        .map(|r| {
            format!(
                r#"    {{
      "base_name": "{}",
      "d_out": {},
      "d_in": {},
      "target_rank": {},
      "grad_frobenius_norm": {:.8},
      "grad_spectral_norm": {:.8},
      "grad_effective_rank": {:.6},
      "nonzero_grad_modes": {},
      "top_r_energy_ratio": {:.6},
      "lora_ga_2r_alignment_ratio": {:.6},
      "kaiming_expected_ratio": {:.6},
      "alignment_gain_vs_kaiming": {:.4},
      "initial_perturbation_norm": {:.8},
      "cross_term_leakage": {:.12}
    }}"#,
                json_escape(&r.base_name),
                r.d_out,
                r.d_in,
                r.target_rank,
                r.grad_frobenius_norm,
                r.grad_spectral_norm,
                r.grad_effective_rank,
                r.nonzero_grad_modes,
                r.top_r_energy_ratio,
                r.lora_ga_2r_alignment_ratio,
                r.kaiming_expected_ratio,
                r.alignment_gain_vs_kaiming,
                r.initial_perturbation_norm,
                r.cross_term_leakage
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
  "source_uri": "{}",
  "scale_mode": "{}",
  "target_rank": {},
  "effective_forward_scale": {:.6},
  "base_compensated": {},
  "mean_top_r_energy_ratio": {:.6},
  "mean_lora_ga_2r_alignment_ratio": {:.6},
  "mean_kaiming_expected_ratio": {:.6},
  "mean_alignment_gain_vs_kaiming": {:.4},
  "total_initial_perturbation_norm": {:.8},
  "layers": [
{}
  ],
  "diagnostics": [
{}
  ]
}}
"#,
        json_escape(&summary.source_uri),
        summary.scale_mode.as_str(),
        summary.target_rank,
        summary.effective_forward_scale,
        summary.base_compensated,
        summary.mean_top_r_energy_ratio,
        summary.mean_lora_ga_2r_alignment_ratio,
        summary.mean_kaiming_expected_ratio,
        summary.mean_alignment_gain_vs_kaiming,
        summary.total_initial_perturbation_norm,
        layers_json.join(",\n"),
        diags_json.join(",\n")
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_random_matrix(rows: usize, cols: usize, rng: &mut Rng64) -> Tensor2D {
        let mut t = Tensor2D::zeros(rows, cols, DType::F32);
        let s = 1.0 / (cols as f64).sqrt();
        for v in &mut t.data {
            *v = rng.next_gaussian() * s;
        }
        t
    }

    #[test]
    fn test_lora_ga_captures_exact_top_2r_singular_modes_without_cross_leakage() {
        let mut rng = Rng64::new(20261119);
        let d_out = 48;
        let d_in = 40;
        let r = 4;

        let g = make_random_matrix(d_out, d_in, &mut rng);
        let full_svd = jacobi_svd(&g);

        let mut dense_map = BTreeMap::new();
        dense_map.insert("model.layers.0.self_attn.q_proj.grad".to_string(), g.clone());

        let cfg = LoraGaConfig {
            target_rank: r,
            scale_mode: LoraGaScaleMode::Direction,
            lora_alpha: r as f64, // effective_scale = 1.0
            emit_init_reference: true,
            ..LoraGaConfig::default()
        };

        let summary = optimize_lora_ga_init(
            "calib_grads.safetensors",
            GradientSource::Dense {
                tensors: &dense_map,
            },
            &cfg,
        )
        .expect("lora-ga init");

        assert_eq!(summary.layers.len(), 1);
        assert!(summary.diagnostics.is_empty());
        let rep = &summary.reports[0];
        assert!(
            rep.cross_term_leakage < 1e-12,
            "Expected zero cross-term leakage, got {}",
            rep.cross_term_leakage
        );

        // Verify step-1 tangent projection B_0 B_0^T G + G A_0^T A_0 equals exact rank-2r truncated SVD of G!
        let l0 = &summary.layers[0];
        let b0 = &l0.b; // U_{0..r} (d_out x r)
        let a0 = &l0.a; // V_{r..2r}^T (r x d_in)
        let b0_t = transpose(b0);
        let a0_t = transpose(a0);

        let term_b = matmul(b0, &matmul(&b0_t, &g));
        let term_a = matmul(&g, &matmul(&a0_t, a0));

        // Exact top-2r reconstruction from full SVD
        let mut g_top_2r = Tensor2D::zeros(d_out, d_in, DType::F32);
        for k in 0..(2 * r) {
            let sk = full_svd.sigma[k];
            for row in 0..d_out {
                let u_rk = full_svd.u.get(row, k);
                for col in 0..d_in {
                    g_top_2r.set(
                        row,
                        col,
                        g_top_2r.get(row, col) + sk * u_rk * full_svd.v.get(col, k),
                    );
                }
            }
        }

        let mut max_err = 0.0_f64;
        for idx in 0..g.data.len() {
            let tangent_val = term_b.data[idx] + term_a.data[idx];
            let err = (tangent_val - g_top_2r.data[idx]).abs();
            if err > max_err {
                max_err = err;
            }
        }
        assert!(
            max_err < 1e-11,
            "Step-1 LoRA-GA tangent update mismatched exact top-2r SVD: max_err={}",
            max_err
        );

        // Verify 2r alignment strictly exceeds top-r ceiling and Kaiming expectation
        assert!(rep.lora_ga_2r_alignment_ratio > rep.top_r_energy_ratio);
        assert!(rep.lora_ga_2r_alignment_ratio > rep.kaiming_expected_ratio);
    }

    #[test]
    fn test_lora_ga_base_weight_compensation_preserves_zero_drift_at_t0() {
        let mut rng = Rng64::new(77);
        let d_out = 32;
        let d_in = 32;
        let r = 4;

        let w0 = make_random_matrix(d_out, d_in, &mut rng);
        let g = make_random_matrix(d_out, d_in, &mut rng);

        let mut base_map = BTreeMap::new();
        base_map.insert("model.layers.0.mlp.up_proj.weight".to_string(), w0.clone());

        let mut grad_map = BTreeMap::new();
        grad_map.insert("model.layers.0.mlp.up_proj.grad".to_string(), g);

        let cfg = LoraGaConfig {
            target_rank: r,
            scale_mode: LoraGaScaleMode::Stable,
            stable_gamma: 16.0,
            lora_alpha: 16.0,
            base_compensated: true,
            ..LoraGaConfig::default()
        };

        let summary = optimize_lora_ga_init(
            "grads.safetensors",
            GradientSource::Dense { tensors: &grad_map },
            &cfg,
        )
        .expect("lora-ga");

        let (compensated_map, count) = apply_lora_ga_base_compensation(
            &base_map,
            &summary.layers,
            summary.effective_forward_scale,
        )
        .expect("base compensation");
        assert_eq!(count, 1);

        let w_res = compensated_map
            .get("model.layers.0.mlp.up_proj.weight")
            .unwrap();
        let ba = matmul(&summary.layers[0].b, &summary.layers[0].a);

        for i in 0..w0.data.len() {
            let reconstructed = w_res.data[i] + summary.effective_forward_scale * ba.data[i];
            assert!(
                (reconstructed - w0.data[i]).abs() < 1e-13,
                "Forward preservation violated at {}: {} vs {}",
                i,
                reconstructed,
                w0.data[i]
            );
        }
    }

    #[test]
    fn test_lora_ga_diagnostics_ls039_and_ls040_on_rank_deficient_uncompensated_warmup() {
        let mut rng = Rng64::new(909);
        let d_out = 32;
        let d_in = 32;
        // Input warmup LoRA has rank 4, but target LoRA-GA rank is 4 (requiring 2*r = 8 modes)
        let warm_layer = LoraLayer {
            base_name: "model.layers.0.self_attn.v_proj".to_string(),
            a_key: "model.layers.0.self_attn.v_proj.lora_A.weight".to_string(),
            b_key: "model.layers.0.self_attn.v_proj.lora_B.weight".to_string(),
            a: make_random_matrix(4, d_in, &mut rng),
            b: make_random_matrix(d_out, 4, &mut rng),
        };
        let alphas = BTreeMap::new();
        let cfg = LoraGaConfig {
            target_rank: 4,
            scale_mode: LoraGaScaleMode::Stable,
            emit_init_reference: false,
            base_compensated: false,
            ..LoraGaConfig::default()
        };

        let summary = optimize_lora_ga_init(
            "warmup.safetensors",
            GradientSource::LowRank {
                layers: &[warm_layer],
                alphas: &alphas,
                peft_config: None,
            },
            &cfg,
        )
        .expect("warmup lora-ga");

        let rules: Vec<&str> = summary.diagnostics.iter().map(|d| d.rule_id).collect();
        assert!(rules.contains(&LS039_INSUFFICIENT_GRAD_RANK));
        assert!(rules.contains(&LS040_UNCOMPENSATED_BASE_DRIFT));
    }
}

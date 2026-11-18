//! TIES-Merging (Yadav et al., NeurIPS 2023) and DARE (Yu et al., ICML 2024)
//! multi-adapter composition with exact Frobenius energy accounting and SARIF diagnostics.
//!
//! Unlike naive factor-level pruning on `(B_t, A_t)` (which is not gauge-invariant under
//! `(B R)(R^{-1} A) = B A` and cannot resolve coordinate sign opposition across `N >= 3`
//! adapters jointly), this module evaluates adapter updates in parameter delta space
//! `tau_t = s_t * B_t * A_t`, performs deterministic magnitude trimming and/or stateless
//! SplitMix64 Bernoulli drop-and-rescale, elects consensus parameter signs across all `N`
//! adapters simultaneously, accounts for every unit of Frobenius energy (`||.||_F^2`),
//! and re-projects the composed delta into optimal rank-`k` LoRA factors via exact Jacobi
//! or orthonormalized subspace-iteration SVD.

use std::collections::{BTreeMap, BTreeSet};

use crate::linalg::{
    jacobi_svd, matmul, multiply_q_cols_by_small, qr_core_singular_values, svd_to_lora_factors,
    thin_qr_columns, transpose, SvdDecomposition,
};
use crate::merge_check::{resolve_effective_layer_scale, AdapterMergeInput, IncompatibleShapeFinding};
use crate::safetensors::{DType, LoraLayer, Tensor2D};
use crate::Rng64;

/// SARIF v2.1.0 rule IDs for TIES and DARE multi-adapter composition.
pub const LS035_TIES_SIGN_CONFLICT: &str = "LS035";
pub const LS036_DARE_SIGNAL_DISTORTION: &str = "LS036";
pub const LS037_MASK_RANK_SCATTERING_LOSS: &str = "LS037";
pub const LS038_EXCESSIVE_TRIM_LOSS: &str = "LS038";

/// Multi-adapter sparsification and sign-election composition mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TiesDareMode {
    /// TIES-Merging (Yadav et al., 2023): magnitude trim top `density` entries per adapter,
    /// elect sign `sgn(sum_t w_t * tau_hat_t)`, and disjoint-merge agreeing adapters.
    Ties,
    /// DARE-Linear (Yu et al., 2024): Bernoulli drop with probability `drop_rate` and
    /// unbiased `1 / (1 - drop_rate)` rescaling, followed by weighted linear summation.
    DareLinear,
    /// DARE-TIES: Bernoulli drop with `1 / (1 - drop_rate)` rescaling (plus optional
    /// magnitude trim if `density < 1.0`), followed by TIES sign election and disjoint merge.
    DareTies,
}

impl TiesDareMode {
    pub fn as_str(&self) -> &'static str {
        match self {
            TiesDareMode::Ties => "ties",
            TiesDareMode::DareLinear => "dare-linear",
            TiesDareMode::DareTies => "dare-ties",
        }
    }

    pub fn from_name(name: &str) -> Result<Self, String> {
        match name.trim().to_ascii_lowercase().as_str() {
            "ties" | "ties-merging" | "ties_merging" => Ok(TiesDareMode::Ties),
            "dare-linear" | "dare_linear" | "dare" => Ok(TiesDareMode::DareLinear),
            "dare-ties" | "dare_ties" => Ok(TiesDareMode::DareTies),
            other => Err(format!(
                "Unsupported TIES/DARE mode '{}': expected ties, dare-ties, or dare-linear",
                other
            )),
        }
    }

    pub fn uses_dare(&self) -> bool {
        matches!(self, TiesDareMode::DareLinear | TiesDareMode::DareTies)
    }

    pub fn uses_sign_election(&self) -> bool {
        matches!(self, TiesDareMode::Ties | TiesDareMode::DareTies)
    }
}

/// Aggregation rule applied across adapters agreeing with the elected coordinate sign.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DisjointAggregation {
    /// Divide summed agreeing updates by `|A_{ij}|` (count of agreeing non-zero adapters).
    /// Preserves full single-adapter magnitude on disjoint coordinates while preventing
    /// `N`-fold norm inflation on shared coordinates (Yadav et al., NeurIPS 2023 Eq. 1).
    DisjointMean,
    /// Divide summed agreeing updates by `sum_{t in A_{ij}} |w_t|`.
    WeightedMean,
    /// Direct sum `sum_{t in A_{ij}} w_t * tau_tilde_{t,ij}` (default for `dare-linear`).
    Sum,
}

impl DisjointAggregation {
    pub fn as_str(&self) -> &'static str {
        match self {
            DisjointAggregation::DisjointMean => "disjoint-mean",
            DisjointAggregation::WeightedMean => "weighted-mean",
            DisjointAggregation::Sum => "sum",
        }
    }

    pub fn from_name(name: &str) -> Result<Self, String> {
        match name.trim().to_ascii_lowercase().as_str() {
            "disjoint-mean" | "disjoint_mean" | "mean" => Ok(DisjointAggregation::DisjointMean),
            "weighted-mean" | "weighted_mean" => Ok(DisjointAggregation::WeightedMean),
            "sum" | "linear" => Ok(DisjointAggregation::Sum),
            other => Err(format!(
                "Unsupported aggregation '{}': expected disjoint-mean, weighted-mean, or sum",
                other
            )),
        }
    }
}

/// Configuration for `compose_adapters_ties_dare`.
#[derive(Debug, Clone)]
pub struct TiesDareConfig {
    pub mode: TiesDareMode,
    /// Fraction of largest-magnitude entries kept during TIES trimming (`0.0 < density <= 1.0`).
    pub density: f64,
    /// Bernoulli drop probability `p` in DARE (`0.0 <= drop_rate < 1.0`).
    pub drop_rate: f64,
    /// Deterministic SplitMix64 seed for DARE masks and subspace-iteration SVD.
    pub seed: u64,
    /// Target output LoRA rank `k >= 1`.
    pub target_rank: usize,
    /// Coordinate aggregation rule.
    pub aggregation: DisjointAggregation,
    /// Cap merged layer spectral norm `sigma_1` at `max_t sigma_1(source_t)`.
    pub preserve_spectral_norm: bool,
    /// Target output tensor dtype (`None` preserves first participating adapter's dtype).
    pub output_dtype: Option<DType>,
}

impl TiesDareConfig {
    pub fn new(mode: TiesDareMode, target_rank: usize) -> Self {
        let (density, drop_rate, aggregation) = match mode {
            TiesDareMode::Ties => (0.5, 0.0, DisjointAggregation::DisjointMean),
            TiesDareMode::DareLinear => (1.0, 0.5, DisjointAggregation::Sum),
            TiesDareMode::DareTies => (1.0, 0.5, DisjointAggregation::DisjointMean),
        };
        Self {
            mode,
            density,
            drop_rate,
            seed: 42,
            target_rank: target_rank.max(1),
            aggregation,
            preserve_spectral_norm: false,
            output_dtype: None,
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        if !self.density.is_finite() || self.density <= 0.0 || self.density > 1.0 {
            return Err(format!(
                "Invalid TIES --density {}: must be in (0.0, 1.0]",
                self.density
            ));
        }
        if !self.drop_rate.is_finite() || self.drop_rate < 0.0 || self.drop_rate >= 1.0 {
            return Err(format!(
                "Invalid DARE --drop-rate {}: must be in [0.0, 1.0)",
                self.drop_rate
            ));
        }
        if self.target_rank == 0 {
            return Err("Invalid --rank 0: target rank must be >= 1".to_string());
        }
        Ok(())
    }
}

/// Diagnostic severity for TIES/DARE composition findings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TiesDareSeverity {
    Error,
    Warning,
}

impl TiesDareSeverity {
    pub fn as_str(&self) -> &'static str {
        match self {
            TiesDareSeverity::Error => "error",
            TiesDareSeverity::Warning => "warning",
        }
    }
}

/// Structured SARIF-ready diagnostic emitted during TIES/DARE composition.
#[derive(Debug, Clone)]
pub struct TiesDareDiagnostic {
    pub rule_id: &'static str,
    pub severity: TiesDareSeverity,
    pub layer_name: String,
    pub artifact_uri: String,
    pub message: String,
    pub sign_conflict_rate: f64,
    pub discarded_conflict_ratio: f64,
    pub svd_retained_ratio: f64,
    pub trim_retained_ratio: f64,
    pub dare_rescale_ratio: f64,
}

/// Exact Frobenius energy ledger (`||.||_F^2`) and sign-election statistics for one layer.
#[derive(Debug, Clone)]
pub struct LayerTiesDareReport {
    pub base_name: String,
    pub d_out: usize,
    pub d_in: usize,
    pub output_rank: usize,
    pub participating_adapters: usize,
    /// `sum_t ||w_t * s_t * B_t * A_t||_F^2`
    pub input_energy_sum: f64,
    pub per_adapter_input_energy: Vec<f64>,
    /// `sum_t ||M_t^{(dare)}||_F^2` after optional DARE drop-and-rescale (equals `input_energy_sum` in pure TIES).
    pub post_dare_energy_sum: f64,
    /// `post_dare_energy_sum / input_energy_sum` (expected `1 / (1 - p)` under DARE).
    pub dare_rescale_energy_ratio: f64,
    /// `sum_t ||M_tilde_t||_F^2` after DARE and magnitude trimming.
    pub post_sparsify_energy_sum: f64,
    /// Exact Frobenius energy removed by magnitude trimming (`post_dare_energy_sum - post_sparsify_energy_sum`).
    pub pruned_trim_energy: f64,
    /// `post_sparsify_energy_sum / post_dare_energy_sum` in `[0, 1]`.
    pub trim_retained_ratio: f64,
    /// Exact Frobenius energy of entries agreeing with the elected coordinate sign.
    pub elected_energy_sum: f64,
    /// Exact Frobenius energy vetoed due to sign conflict (`post_sparsify_energy_sum - elected_energy_sum`).
    pub discarded_conflict_energy: f64,
    /// `discarded_conflict_energy / post_sparsify_energy_sum` in `[0, 1]`.
    pub discarded_conflict_ratio: f64,
    /// Coordinates with `>= 1` non-zero adapter after sparsification.
    pub active_coords: usize,
    /// Coordinates with `>= 2` non-zero adapters after sparsification.
    pub overlap_coords: usize,
    /// Overlapping coordinates where positive and negative updates collide.
    pub sign_conflict_coords: usize,
    /// `sign_conflict_coords / overlap_coords` in `[0, 1]`.
    pub sign_conflict_rate: f64,
    pub elected_positive_coords: usize,
    pub elected_negative_coords: usize,
    /// Exact Frobenius energy `||DeltaW_*||_F^2` of the dense composed delta prior to rank-`k` SVD.
    pub dense_merged_energy: f64,
    /// Exact Frobenius energy `sum_{i=1}^k sigma_i^2(DeltaW_*)` captured by top-`k` singular modes prior to `--preserve-norm`.
    pub pre_cap_low_rank_energy: f64,
    /// Exact Eckart-Young rank-`k` truncation loss `||DeltaW_* - DeltaW_{*,k}||_F^2 = dense_merged_energy - pre_cap_low_rank_energy`.
    pub svd_truncation_loss_energy: f64,
    /// `pre_cap_low_rank_energy / dense_merged_energy` in `[0, 1]`.
    pub svd_retained_energy_ratio: f64,
    /// Final output LoRA Frobenius energy `||B_k * A_k||_F^2` (after optional `--preserve-norm` cap).
    pub low_rank_output_energy: f64,
    /// Rank required to retain 99% of `DeltaW_*`'s spectral energy.
    pub dense_rank_99: usize,
    /// SARIF rule codes triggered on this layer.
    pub triggered_rules: Vec<&'static str>,
}

/// Full adapter composition result with merged `LoraLayer` tensors, energy ledger, and SARIF diagnostics.
#[derive(Debug, Clone)]
pub struct TiesDareComposeSummary {
    pub mode: TiesDareMode,
    pub density: f64,
    pub drop_rate: f64,
    pub seed: u64,
    pub target_rank: usize,
    pub aggregation: DisjointAggregation,
    pub preserve_spectral_norm: bool,
    pub adapter_names: Vec<String>,
    pub input_weights: Vec<f64>,
    pub merged_layers: Vec<LoraLayer>,
    pub layer_reports: Vec<LayerTiesDareReport>,
    pub incompatible_shapes: Vec<IncompatibleShapeFinding>,
    pub diagnostics: Vec<TiesDareDiagnostic>,
    pub total_input_energy: f64,
    pub total_post_dare_energy: f64,
    pub total_post_sparsify_energy: f64,
    pub total_pruned_trim_energy: f64,
    pub total_elected_energy: f64,
    pub total_discarded_conflict_energy: f64,
    pub total_dense_merged_energy: f64,
    pub total_pre_cap_low_rank_energy: f64,
    pub total_svd_truncation_loss: f64,
    pub total_low_rank_output_energy: f64,
    pub mean_sign_conflict_rate: f64,
    pub mean_svd_retained_ratio: f64,
}

impl TiesDareComposeSummary {
    pub fn has_errors(&self) -> bool {
        !self.incompatible_shapes.is_empty()
            || self
                .diagnostics
                .iter()
                .any(|d| d.severity == TiesDareSeverity::Error)
    }

    pub fn has_conflicts(&self) -> bool {
        self.has_errors()
            || self
                .diagnostics
                .iter()
                .any(|d| d.rule_id == LS035_TIES_SIGN_CONFLICT)
    }
}

/// Deterministic FNV-1a 64-bit hash for layer names so different layers draw independent DARE masks.
#[inline]
pub fn fnv1a_64(s: &str) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in s.as_bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// Stateless SplitMix64 uniform `[0, 1)` coordinate sampler for DARE Bernoulli masks.
///
/// Mixing `(seed, adapter_idx, layer_hash, row, col)` ensures each adapter `t` draws an
/// independent mask without materializing boolean mask tensors, reducing two-adapter
/// coordinate collision probability from `1` to `(1 - p)^2`.
#[inline]
pub fn dare_coordinate_u01(
    seed: u64,
    adapter_idx: usize,
    layer_hash: u64,
    row: usize,
    col: usize,
) -> f64 {
    let mut z = seed
        .wrapping_mul(0xa076_1d64_78bd_642f)
        .wrapping_add(0x9e37_79b9_7f4a_7c15)
        ^ (adapter_idx as u64).wrapping_mul(0xe703_7ed1_a0b4_28db)
        ^ layer_hash.wrapping_mul(0x8ebc_6af0_9c88_c6e3)
        ^ (row as u64).wrapping_mul(0x5899_65cc_7537_4cc3)
        ^ (col as u64).wrapping_mul(0x1d8e_4e27_c47d_124f);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^= z >> 31;
    ((z >> 11) as f64) * (1.0 / ((1u64 << 53) as f64))
}

/// Apply DARE Bernoulli drop (`drop_rate = p`) and unbiased `1 / (1 - p)` rescaling in place.
/// Returns `(pre_dare_energy, post_dare_energy)`.
pub fn apply_dare_drop_rescale_in_place(
    delta: &mut [f64],
    d_out: usize,
    d_in: usize,
    drop_rate: f64,
    seed: u64,
    adapter_idx: usize,
    layer_hash: u64,
) -> (f64, f64) {
    let mut pre_energy = 0.0_f64;
    for &v in delta.iter() {
        pre_energy += v * v;
    }
    if drop_rate <= 0.0 || delta.is_empty() {
        return (pre_energy, pre_energy);
    }
    let inv_keep = 1.0 / (1.0 - drop_rate);
    let mut post_energy = 0.0_f64;
    for r in 0..d_out {
        let row_off = r * d_in;
        for c in 0..d_in {
            let idx = row_off + c;
            let val = delta[idx];
            if val == 0.0 {
                continue;
            }
            if dare_coordinate_u01(seed, adapter_idx, layer_hash, r, c) < drop_rate {
                delta[idx] = 0.0;
            } else {
                let scaled = val * inv_keep;
                delta[idx] = scaled;
                post_energy += scaled * scaled;
            }
        }
    }
    (pre_energy, post_energy)
}

/// Apply exact top-`density` magnitude trimming in place via `O(L)` introselect,
/// breaking equal-magnitude ties deterministically by ascending coordinate index.
/// Returns `(pre_trim_energy, post_trim_energy, pruned_energy)` satisfying
/// `pre_trim_energy == post_trim_energy + pruned_energy`.
pub fn apply_magnitude_trim_in_place(delta: &mut [f64], density: f64) -> (f64, f64, f64) {
    let len = delta.len();
    let mut pre_energy = 0.0_f64;
    for &v in delta.iter() {
        pre_energy += v * v;
    }
    if len == 0 || density >= 1.0 - 1e-12 {
        return (pre_energy, pre_energy, 0.0);
    }

    let keep_count = ((len as f64) * density.clamp(0.0, 1.0)).ceil() as usize;
    let keep_count = keep_count.clamp(1, len);
    if keep_count >= len {
        return (pre_energy, pre_energy, 0.0);
    }

    let mut order: Vec<(f64, usize)> = delta
        .iter()
        .enumerate()
        .map(|(idx, &v)| (v.abs(), idx))
        .collect();

    order.select_nth_unstable_by(keep_count, |a, b| {
        b.0.total_cmp(&a.0).then_with(|| a.1.cmp(&b.1))
    });

    let mut post_energy = 0.0_f64;
    for &(_, idx) in &order[..keep_count] {
        let v = delta[idx];
        post_energy += v * v;
    }
    let mut pruned_energy = 0.0_f64;
    for &(_, idx) in &order[keep_count..] {
        let v = delta[idx];
        pruned_energy += v * v;
        delta[idx] = 0.0;
    }

    (pre_energy, post_energy, pruned_energy)
}

fn energy_rank_99_from_spectrum(sigma: &[f64], total_energy: f64) -> usize {
    if total_energy <= 1e-30 || sigma.is_empty() {
        return 1;
    }
    let mut cum = 0.0_f64;
    for (idx, &s) in sigma.iter().enumerate() {
        cum += s * s;
        if cum >= 0.99 * total_energy {
            return idx + 1;
        }
    }
    // If truncated spectrum did not yet reach 99% of total dense energy, lower-bound by len + 1
    (sigma.len() + 1).max(1)
}

/// Compute top-`k` SVD of a dense delta matrix `mat` (`d_out x d_in`).
///
/// Uses exact One-Sided Jacobi SVD when `min(d_out, d_in) <= 192` or `k + 12 >= min_dim`,
/// and 3-pass QR-orthonormalized Randomized Subspace Iteration (`l = k + 12`) for larger
/// matrices, reducing complexity from `O(d^3)` to `O(l * d_out * d_in + l^2 * (d_out + d_in))`.
pub fn truncated_dense_svd(mat: &Tensor2D, k: usize, seed: u64) -> SvdDecomposition {
    let d_out = mat.rows;
    let d_in = mat.cols;
    let min_dim = d_out.min(d_in);
    let keep = k.min(min_dim).max(1);

    if min_dim <= 192 || keep + 12 >= min_dim {
        let full = jacobi_svd(mat);
        let mut u_trunc = Tensor2D::zeros(d_out, keep, mat.dtype);
        for r in 0..d_out {
            for c in 0..keep {
                u_trunc.set(r, c, full.u.get(r, c));
            }
        }
        let mut v_trunc = Tensor2D::zeros(d_in, keep, mat.dtype);
        for r in 0..d_in {
            for c in 0..keep {
                v_trunc.set(r, c, full.v.get(r, c));
            }
        }
        return SvdDecomposition {
            u: u_trunc,
            sigma: full.sigma,
            v: v_trunc,
        };
    }

    // If d_out < d_in, factorize mat^T so the tall dimension is always on the left
    if d_out < d_in {
        let mt = transpose(mat);
        let svd_t = truncated_dense_svd(&mt, keep, seed ^ 0x6a09_e667_f3bc_c908);
        return SvdDecomposition {
            u: svd_t.v,
            sigma: svd_t.sigma,
            v: svd_t.u,
        };
    }

    let sub_l = (keep + 12).min(min_dim);
    let mut rng = Rng64::new(seed ^ 0x517c_c1b7_2722_0a95);
    let mut omega = Tensor2D::zeros(d_in, sub_l, DType::F32);
    let scale = 1.0 / (sub_l as f64).sqrt();
    for v in &mut omega.data {
        *v = rng.next_gaussian() * scale;
    }

    let mut y = matmul(mat, &omega);
    let mat_t = transpose(mat);

    // 3 orthonormalized subspace power iterations (Halko-Martinsson-Tropp Alg. 4.4)
    for _ in 0..3 {
        let (qy_cols, _) = thin_qr_columns(&y);
        let mut qy_mat = Tensor2D::zeros(d_out, sub_l, DType::F32);
        for r in 0..d_out {
            for c in 0..sub_l {
                qy_mat.set(r, c, qy_cols[c][r]);
            }
        }
        let z = matmul(&mat_t, &qy_mat);
        let (qz_cols, _) = thin_qr_columns(&z);
        let mut qz_mat = Tensor2D::zeros(d_in, sub_l, DType::F32);
        for r in 0..d_in {
            for c in 0..sub_l {
                qz_mat.set(r, c, qz_cols[c][r]);
            }
        }
        y = matmul(mat, &qz_mat);
    }

    let (q_cols, _) = thin_qr_columns(&y);
    let mut q_mat = Tensor2D::zeros(d_out, sub_l, DType::F32);
    for r in 0..d_out {
        for c in 0..sub_l {
            q_mat.set(r, c, q_cols[c][r]);
        }
    }

    // proj_t = mat^T * Q has shape (d_in, sub_l); thin-QR reduces Jacobi SVD to (sub_l x sub_l)
    let proj_t = matmul(&mat_t, &q_mat);
    let (qa_cols, r_small) = thin_qr_columns(&proj_t);
    let core = transpose(&r_small);
    let core_svd = jacobi_svd(&core);

    let u_trunc = multiply_q_cols_by_small(&q_cols, d_out, &core_svd.u, keep, mat.dtype);
    let v_trunc = multiply_q_cols_by_small(&qa_cols, d_in, &core_svd.v, keep, mat.dtype);

    SvdDecomposition {
        u: u_trunc,
        sigma: core_svd.sigma,
        v: v_trunc,
    }
}

fn materialize_scaled_delta(layer: &LoraLayer, scale: f64) -> Vec<f64> {
    let d_out = layer.d_out();
    let d_in = layer.d_in();
    let r = layer.rank();
    let mut out = vec![0.0_f64; d_out * d_in];
    if scale == 0.0 || r == 0 {
        return out;
    }
    for i in 0..d_out {
        let b_row = &layer.b.data[i * r..(i + 1) * r];
        let out_row = &mut out[i * d_in..(i + 1) * d_in];
        for p in 0..r {
            let b_ip = b_row[p] * scale;
            if b_ip == 0.0 {
                continue;
            }
            let a_row = &layer.a.data[p * d_in..(p + 1) * d_in];
            for j in 0..d_in {
                out_row[j] += b_ip * a_row[j];
            }
        }
    }
    out
}

/// Compose `N >= 2` LoRA adapters jointly using TIES-Merging, DARE-Linear, or DARE-TIES,
/// returning merged `LoraLayer` tensors, per-layer Frobenius energy ledgers, and SARIF diagnostics.
pub fn compose_adapters_ties_dare(
    inputs: &[AdapterMergeInput],
    config: &TiesDareConfig,
) -> Result<TiesDareComposeSummary, String> {
    config.validate()?;
    let n = inputs.len();
    if n < 2 {
        return Err("TIES/DARE composition requires at least 2 adapters".to_string());
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

    let mut merged_layers = Vec::with_capacity(all_layer_names.len());
    let mut layer_reports = Vec::with_capacity(all_layer_names.len());
    let mut incompatible_shapes = Vec::new();
    let mut diagnostics = Vec::new();

    for base_name in all_layer_names {
        let mut present: Vec<(usize, &LoraLayer, f64, f64)> = Vec::new();
        for (idx, m) in layer_maps.iter().enumerate() {
            if let Some(&layer) = m.get(base_name) {
                let alpha_scale = resolve_effective_layer_scale(
                    layer,
                    &inputs[idx].alphas,
                    inputs[idx].peft_config.as_ref(),
                );
                let user_w = inputs[idx].weight;
                present.push((idx, layer, alpha_scale, user_w));
            }
        }
        if present.is_empty() {
            continue;
        }

        let (first_idx, first_layer, _, _) = present[0];
        let d_out = first_layer.d_out();
        let d_in = first_layer.d_in();
        let mut shape_mismatch = false;
        for &(other_idx, other_layer, _, _) in &present[1..] {
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

        let numel = d_out * d_in;
        let m_count = present.len();
        let layer_hash = fnv1a_64(base_name);
        let out_dtype = config.output_dtype.unwrap_or(first_layer.a.dtype);

        let mut max_source_sigma = 0.0_f64;
        if config.preserve_spectral_norm {
            for &(_, layer, alpha_scale, _) in &present {
                let s1 = qr_core_singular_values(&layer.b, &layer.a)
                    .first()
                    .copied()
                    .unwrap_or(0.0)
                    * alpha_scale.abs();
                if s1 > max_source_sigma {
                    max_source_sigma = s1;
                }
            }
        }

        // Stage 1: Materialize weighted deltas M_t = (w_t * s_t) * B_t * A_t,
        // then apply DARE drop-and-rescale and/or TIES magnitude trimming.
        let mut deltas: Vec<Vec<f64>> = Vec::with_capacity(m_count);
        let mut per_adapter_input_energy = Vec::with_capacity(m_count);
        let mut input_energy_sum = 0.0_f64;
        let mut post_dare_energy_sum = 0.0_f64;
        let mut post_sparsify_energy_sum = 0.0_f64;
        let mut pruned_trim_energy = 0.0_f64;

        for &(adapter_idx, layer, alpha_scale, user_w) in &present {
            let mut delta = materialize_scaled_delta(layer, alpha_scale * user_w);
            let (e_in, e_after_dare) = if config.mode.uses_dare() && config.drop_rate > 0.0 {
                apply_dare_drop_rescale_in_place(
                    &mut delta,
                    d_out,
                    d_in,
                    config.drop_rate,
                    config.seed,
                    adapter_idx,
                    layer_hash,
                )
            } else {
                let mut e = 0.0_f64;
                for &v in &delta {
                    e += v * v;
                }
                (e, e)
            };

            let (_e_pre_trim, e_post_trim, e_pruned) = if config.density < 1.0 - 1e-12 {
                apply_magnitude_trim_in_place(&mut delta, config.density)
            } else {
                (e_after_dare, e_after_dare, 0.0)
            };

            per_adapter_input_energy.push(e_in);
            input_energy_sum += e_in;
            post_dare_energy_sum += e_after_dare;
            post_sparsify_energy_sum += e_post_trim;
            pruned_trim_energy += e_pruned;
            deltas.push(delta);
        }

        let dare_rescale_energy_ratio = if input_energy_sum > 1e-30 {
            post_dare_energy_sum / input_energy_sum
        } else {
            1.0
        };
        let trim_retained_ratio = if post_dare_energy_sum > 1e-30 {
            (post_sparsify_energy_sum / post_dare_energy_sum).clamp(0.0, 1.0)
        } else {
            1.0
        };

        // Stage 2: Joint N-adapter sign election & coordinate aggregation
        let mut dense_merged = Tensor2D::zeros(d_out, d_in, DType::F32);
        let mut elected_energy_sum = 0.0_f64;
        let mut discarded_conflict_energy = 0.0_f64;
        let mut active_coords = 0usize;
        let mut overlap_coords = 0usize;
        let mut sign_conflict_coords = 0usize;
        let mut elected_positive_coords = 0usize;
        let mut elected_negative_coords = 0usize;
        let mut dense_merged_energy = 0.0_f64;

        let use_election = config.mode.uses_sign_election();

        for idx in 0..numel {
            let mut sum_signed = 0.0_f64;
            let mut nz_pos = 0usize;
            let mut nz_neg = 0usize;
            let mut max_abs = 0.0_f64;
            let mut dominant_sign = 1.0_f64;

            for p in 0..m_count {
                let v = deltas[p][idx];
                if v > 0.0 {
                    nz_pos += 1;
                    sum_signed += v;
                    if v > max_abs {
                        max_abs = v;
                        dominant_sign = 1.0;
                    }
                } else if v < 0.0 {
                    nz_neg += 1;
                    sum_signed += v;
                    let av = -v;
                    if av > max_abs {
                        max_abs = av;
                        dominant_sign = -1.0;
                    }
                }
            }

            let nz_total = nz_pos + nz_neg;
            if nz_total == 0 {
                continue;
            }
            active_coords += 1;
            if nz_total >= 2 {
                overlap_coords += 1;
            }
            if nz_pos > 0 && nz_neg > 0 {
                sign_conflict_coords += 1;
            }

            let merged_val = if use_election {
                let elected_sign = if sum_signed > 0.0 {
                    1.0
                } else if sum_signed < 0.0 {
                    -1.0
                } else {
                    dominant_sign
                };
                if elected_sign > 0.0 {
                    elected_positive_coords += 1;
                } else {
                    elected_negative_coords += 1;
                }

                let mut agree_sum = 0.0_f64;
                let mut agree_count = 0usize;
                let mut agree_w_abs = 0.0_f64;

                for p in 0..m_count {
                    let v = deltas[p][idx];
                    if v == 0.0 {
                        continue;
                    }
                    if v * elected_sign > 0.0 {
                        elected_energy_sum += v * v;
                        agree_sum += v;
                        agree_count += 1;
                        agree_w_abs += present[p].3.abs();
                    } else {
                        discarded_conflict_energy += v * v;
                    }
                }

                match config.aggregation {
                    DisjointAggregation::DisjointMean => {
                        if agree_count > 0 {
                            agree_sum / (agree_count as f64)
                        } else {
                            0.0
                        }
                    }
                    DisjointAggregation::WeightedMean => {
                        if agree_w_abs > 1e-18 {
                            agree_sum / agree_w_abs
                        } else if agree_count > 0 {
                            agree_sum / (agree_count as f64)
                        } else {
                            0.0
                        }
                    }
                    DisjointAggregation::Sum => agree_sum,
                }
            } else {
                // DareLinear: no sign veto; all surviving updates participate
                if sum_signed >= 0.0 {
                    elected_positive_coords += 1;
                } else {
                    elected_negative_coords += 1;
                }
                let mut w_abs_sum = 0.0_f64;
                for p in 0..m_count {
                    let v = deltas[p][idx];
                    if v != 0.0 {
                        elected_energy_sum += v * v;
                        w_abs_sum += present[p].3.abs();
                    }
                }
                match config.aggregation {
                    DisjointAggregation::Sum => sum_signed,
                    DisjointAggregation::DisjointMean => sum_signed / (nz_total as f64),
                    DisjointAggregation::WeightedMean => {
                        if w_abs_sum > 1e-18 {
                            sum_signed / w_abs_sum
                        } else {
                            sum_signed / (nz_total as f64)
                        }
                    }
                }
            };

            dense_merged.data[idx] = merged_val;
            dense_merged_energy += merged_val * merged_val;
        }

        let sign_conflict_rate = if overlap_coords > 0 {
            (sign_conflict_coords as f64) / (overlap_coords as f64)
        } else {
            0.0
        };
        let discarded_conflict_ratio = if post_sparsify_energy_sum > 1e-30 {
            (discarded_conflict_energy / post_sparsify_energy_sum).clamp(0.0, 1.0)
        } else {
            0.0
        };

        // Stage 3: Optimal Eckart-Young rank-k re-factorization
        let keep = config.target_rank.min(d_out.min(d_in)).max(1);
        let svd = truncated_dense_svd(&dense_merged, keep, config.seed ^ layer_hash);
        let raw_top_k_energy: f64 = svd.sigma.iter().take(keep).map(|&s| s * s).sum();
        let pre_cap_low_rank_energy = raw_top_k_energy.min(dense_merged_energy);
        let svd_truncation_loss_energy = (dense_merged_energy - pre_cap_low_rank_energy).max(0.0);
        let svd_retained_energy_ratio = if dense_merged_energy > 1e-30 {
            (pre_cap_low_rank_energy / dense_merged_energy).clamp(0.0, 1.0)
        } else {
            1.0
        };
        let dense_rank_99 = energy_rank_99_from_spectrum(&svd.sigma, dense_merged_energy);

        let (mut b_merged, mut a_merged) = svd_to_lora_factors(&svd, keep, out_dtype);
        let mut low_rank_output_energy = pre_cap_low_rank_energy;

        if config.preserve_spectral_norm && max_source_sigma > 1e-12 {
            let s1_merged = svd.sigma.first().copied().unwrap_or(0.0);
            if s1_merged > max_source_sigma {
                let scale_factor = max_source_sigma / s1_merged;
                let sqrt_scale = scale_factor.sqrt();
                for v in &mut b_merged.data {
                    *v *= sqrt_scale;
                }
                for v in &mut a_merged.data {
                    *v *= sqrt_scale;
                }
                low_rank_output_energy *= scale_factor * scale_factor;
            }
        }

        // Stage 4: Evaluate SARIF diagnostics LS035..LS038
        let primary_uri = &inputs[first_idx].name;
        let mut triggered_rules = Vec::new();

        // LS035: Severe cross-adapter sign opposition (or linear cancellation in DareLinear)
        let dare_linear_cancelled = !use_election
            && m_count >= 2
            && sign_conflict_rate >= 0.35
            && post_sparsify_energy_sum > 1e-18
            && dense_merged_energy < 0.55 * post_sparsify_energy_sum;

        if m_count >= 2
            && (sign_conflict_rate >= 0.35
                || discarded_conflict_ratio >= 0.25
                || dare_linear_cancelled)
        {
            triggered_rules.push(LS035_TIES_SIGN_CONFLICT);
            let sev = if discarded_conflict_ratio >= 0.40 || dare_linear_cancelled {
                TiesDareSeverity::Error
            } else {
                TiesDareSeverity::Warning
            };
            let msg = if use_election {
                format!(
                    "Layer '{}' experienced {:.1}% sign conflict across {} overlapping coordinates, vetoing {:.1}% of post-trim Frobenius energy",
                    base_name,
                    sign_conflict_rate * 100.0,
                    overlap_coords,
                    discarded_conflict_ratio * 100.0
                )
            } else {
                format!(
                    "Layer '{}' suffered destructive sign cancellation under dare-linear ({:.1}% opposing overlaps, merged energy {:.4e} vs post-dare {:.4e}); switch to --mode dare-ties or ortho-svd",
                    base_name,
                    sign_conflict_rate * 100.0,
                    dense_merged_energy,
                    post_sparsify_energy_sum
                )
            };
            diagnostics.push(TiesDareDiagnostic {
                rule_id: LS035_TIES_SIGN_CONFLICT,
                severity: sev,
                layer_name: base_name.to_string(),
                artifact_uri: primary_uri.clone(),
                message: msg,
                sign_conflict_rate,
                discarded_conflict_ratio,
                svd_retained_ratio: svd_retained_energy_ratio,
                trim_retained_ratio,
                dare_rescale_ratio: dare_rescale_energy_ratio,
            });
        }

        // LS036: DARE drop wipeout or extreme heavy-tailed rescale variance blowup
        if config.mode.uses_dare() && config.drop_rate > 0.0 && input_energy_sum > 1e-18 {
            let expected_ratio = 1.0 / (1.0 - config.drop_rate);
            let wiped_out = dare_rescale_energy_ratio < 0.20;
            let variance_spike = dare_rescale_energy_ratio > 2.5 * expected_ratio;
            if wiped_out || variance_spike {
                triggered_rules.push(LS036_DARE_SIGNAL_DISTORTION);
                let (sev, msg) = if wiped_out {
                    (
                        TiesDareSeverity::Error,
                        format!(
                            "Layer '{}' lost {:.1}% of Frobenius energy after DARE drop_rate={:.2} + 1/(1-p) rescaling (concentrated sparse weights dropped); lower --drop-rate or use --mode ties",
                            base_name,
                            (1.0 - dare_rescale_energy_ratio).max(0.0) * 100.0,
                            config.drop_rate
                        ),
                    )
                } else {
                    (
                        TiesDareSeverity::Warning,
                        format!(
                            "Layer '{}' post-DARE energy jumped {:.2}x (expected {:.2}x at drop_rate={:.2}) due to outlier rescaling variance",
                            base_name,
                            dare_rescale_energy_ratio,
                            expected_ratio,
                            config.drop_rate
                        ),
                    )
                };
                diagnostics.push(TiesDareDiagnostic {
                    rule_id: LS036_DARE_SIGNAL_DISTORTION,
                    severity: sev,
                    layer_name: base_name.to_string(),
                    artifact_uri: primary_uri.clone(),
                    message: msg,
                    sign_conflict_rate,
                    discarded_conflict_ratio,
                    svd_retained_ratio: svd_retained_energy_ratio,
                    trim_retained_ratio,
                    dare_rescale_ratio: dare_rescale_energy_ratio,
                });
            }
        }

        // LS037: Non-linear mask rank scattering exceeds target_rank capacity
        if dense_merged_energy > 1e-18 && svd_retained_energy_ratio < 0.85 {
            triggered_rules.push(LS037_MASK_RANK_SCATTERING_LOSS);
            diagnostics.push(TiesDareDiagnostic {
                rule_id: LS037_MASK_RANK_SCATTERING_LOSS,
                severity: TiesDareSeverity::Warning,
                layer_name: base_name.to_string(),
                artifact_uri: primary_uri.clone(),
                message: format!(
                    "Layer '{}' retained only {:.1}% of composed Frobenius energy at target rank {} (mask rank scattering requires rank >= {} for 99% energy)",
                    base_name,
                    svd_retained_energy_ratio * 100.0,
                    keep,
                    dense_rank_99
                ),
                sign_conflict_rate,
                discarded_conflict_ratio,
                svd_retained_ratio: svd_retained_energy_ratio,
                trim_retained_ratio,
                dare_rescale_ratio: dare_rescale_energy_ratio,
            });
        }

        // LS038: Over-aggressive TIES magnitude trimming discarding > 50% energy
        if config.density < 1.0 - 1e-12
            && post_dare_energy_sum > 1e-18
            && trim_retained_ratio < 0.50
        {
            triggered_rules.push(LS038_EXCESSIVE_TRIM_LOSS);
            diagnostics.push(TiesDareDiagnostic {
                rule_id: LS038_EXCESSIVE_TRIM_LOSS,
                severity: TiesDareSeverity::Warning,
                layer_name: base_name.to_string(),
                artifact_uri: primary_uri.clone(),
                message: format!(
                    "Layer '{}' retained only {:.1}% of Frobenius energy after TIES magnitude trim at density={:.2}; increase --density to preserve dense task signal",
                    base_name,
                    trim_retained_ratio * 100.0,
                    config.density
                ),
                sign_conflict_rate,
                discarded_conflict_ratio,
                svd_retained_ratio: svd_retained_energy_ratio,
                trim_retained_ratio,
                dare_rescale_ratio: dare_rescale_energy_ratio,
            });
        }

        merged_layers.push(LoraLayer {
            base_name: base_name.to_string(),
            a_key: first_layer.a_key.clone(),
            b_key: first_layer.b_key.clone(),
            a: a_merged,
            b: b_merged,
        });

        layer_reports.push(LayerTiesDareReport {
            base_name: base_name.to_string(),
            d_out,
            d_in,
            output_rank: keep,
            participating_adapters: m_count,
            input_energy_sum,
            per_adapter_input_energy,
            post_dare_energy_sum,
            dare_rescale_energy_ratio,
            post_sparsify_energy_sum,
            pruned_trim_energy,
            trim_retained_ratio,
            elected_energy_sum,
            discarded_conflict_energy,
            discarded_conflict_ratio,
            active_coords,
            overlap_coords,
            sign_conflict_coords,
            sign_conflict_rate,
            elected_positive_coords,
            elected_negative_coords,
            dense_merged_energy,
            pre_cap_low_rank_energy,
            svd_truncation_loss_energy,
            svd_retained_energy_ratio,
            low_rank_output_energy,
            dense_rank_99,
            triggered_rules,
        });
    }

    let total_input_energy: f64 = layer_reports.iter().map(|r| r.input_energy_sum).sum();
    let total_post_dare_energy: f64 = layer_reports.iter().map(|r| r.post_dare_energy_sum).sum();
    let total_post_sparsify_energy: f64 = layer_reports
        .iter()
        .map(|r| r.post_sparsify_energy_sum)
        .sum();
    let total_pruned_trim_energy: f64 = layer_reports.iter().map(|r| r.pruned_trim_energy).sum();
    let total_elected_energy: f64 = layer_reports.iter().map(|r| r.elected_energy_sum).sum();
    let total_discarded_conflict_energy: f64 = layer_reports
        .iter()
        .map(|r| r.discarded_conflict_energy)
        .sum();
    let total_dense_merged_energy: f64 = layer_reports.iter().map(|r| r.dense_merged_energy).sum();
    let total_pre_cap_low_rank_energy: f64 = layer_reports
        .iter()
        .map(|r| r.pre_cap_low_rank_energy)
        .sum();
    let total_svd_truncation_loss: f64 = layer_reports
        .iter()
        .map(|r| r.svd_truncation_loss_energy)
        .sum();
    let total_low_rank_output_energy: f64 = layer_reports
        .iter()
        .map(|r| r.low_rank_output_energy)
        .sum();

    let multi_reports: Vec<&LayerTiesDareReport> = layer_reports
        .iter()
        .filter(|r| r.participating_adapters >= 2)
        .collect();
    let mean_sign_conflict_rate = if !multi_reports.is_empty() {
        multi_reports
            .iter()
            .map(|r| r.sign_conflict_rate)
            .sum::<f64>()
            / (multi_reports.len() as f64)
    } else {
        0.0
    };
    let mean_svd_retained_ratio = if !layer_reports.is_empty() {
        layer_reports
            .iter()
            .map(|r| r.svd_retained_energy_ratio)
            .sum::<f64>()
            / (layer_reports.len() as f64)
    } else {
        1.0
    };

    Ok(TiesDareComposeSummary {
        mode: config.mode,
        density: config.density,
        drop_rate: config.drop_rate,
        seed: config.seed,
        target_rank: config.target_rank,
        aggregation: config.aggregation,
        preserve_spectral_norm: config.preserve_spectral_norm,
        adapter_names,
        input_weights,
        merged_layers,
        layer_reports,
        incompatible_shapes,
        diagnostics,
        total_input_energy,
        total_post_dare_energy,
        total_post_sparsify_energy,
        total_pruned_trim_energy,
        total_elected_energy,
        total_discarded_conflict_energy,
        total_dense_merged_energy,
        total_pre_cap_low_rank_energy,
        total_svd_truncation_loss,
        total_low_rank_output_energy,
        mean_sign_conflict_rate,
        mean_svd_retained_ratio,
    })
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

/// Render human-readable CLI table for `loraspectra ties-dare`.
pub fn render_ties_dare_table(summary: &TiesDareComposeSummary) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "TIES/DARE Multi-Adapter Composition (mode={}, density={:.2}, drop_rate={:.2}, seed={}, rank={}, agg={})\n",
        summary.mode.as_str(),
        summary.density,
        summary.drop_rate,
        summary.seed,
        summary.target_rank,
        summary.aggregation.as_str()
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
        out.push_str("\nINCOMPATIBLE SHAPES:\n");
        for inc in &summary.incompatible_shapes {
            out.push_str(&format!(
                "  [LS018] {}: {} ({}x{}) vs {} ({}x{})\n",
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

    out.push('\n');
    out.push_str(&format!(
        "{:<40} {:>5} {:>8} {:>9} {:>9} {:>9} {:>9} {:<12}\n",
        "LAYER", "ADAPT", "TRIM_RET", "CONFLICT", "VETO_E%", "SVD_RET", "OUT_NORM", "RULES"
    ));
    out.push_str(&"-".repeat(108));
    out.push('\n');

    for r in &summary.layer_reports {
        let rules_str = if r.triggered_rules.is_empty() {
            "OK".to_string()
        } else {
            r.triggered_rules.join(",")
        };
        out.push_str(&format!(
            "{:<40} {:>5} {:>7.1}% {:>8.1}% {:>8.1}% {:>8.1}% {:>9.4} {:<12}\n",
            r.base_name,
            r.participating_adapters,
            r.trim_retained_ratio * 100.0,
            r.sign_conflict_rate * 100.0,
            r.discarded_conflict_ratio * 100.0,
            r.svd_retained_energy_ratio * 100.0,
            r.low_rank_output_energy.sqrt(),
            rules_str
        ));
    }

    out.push_str(&"-".repeat(108));
    out.push('\n');
    out.push_str(&format!(
        "Frobenius Energy Ledger (||.||_F^2): input={:.6e} -> post_dare={:.6e} -> post_trim={:.6e} (pruned={:.6e})\n",
        summary.total_input_energy,
        summary.total_post_dare_energy,
        summary.total_post_sparsify_energy,
        summary.total_pruned_trim_energy
    ));
    out.push_str(&format!(
        "Sign Election & SVD Projection:      elected={:.6e} + vetoed={:.6e} | dense_merged={:.6e} -> rank_{}={:.6e} (svd_loss={:.6e})\n",
        summary.total_elected_energy,
        summary.total_discarded_conflict_energy,
        summary.total_dense_merged_energy,
        summary.target_rank,
        summary.total_pre_cap_low_rank_energy,
        summary.total_svd_truncation_loss
    ));
    out.push_str(&format!(
        "Summary: {} layers | mean sign conflict {:.1}% | mean SVD energy retained {:.1}% | diagnostics: {}\n",
        summary.layer_reports.len(),
        summary.mean_sign_conflict_rate * 100.0,
        summary.mean_svd_retained_ratio * 100.0,
        summary.diagnostics.len()
    ));

    out
}

/// Render machine-readable JSON summary for `loraspectra ties-dare --json`.
pub fn render_ties_dare_json(summary: &TiesDareComposeSummary) -> String {
    let adapters_json: Vec<String> = summary
        .adapter_names
        .iter()
        .zip(summary.input_weights.iter())
        .map(|(n, w)| format!("{{\"name\": \"{}\", \"weight\": {:.6}}}", json_escape(n), w))
        .collect();

    let layers_json: Vec<String> = summary
        .layer_reports
        .iter()
        .map(|r| {
            let rules_j: Vec<String> = r
                .triggered_rules
                .iter()
                .map(|c| format!("\"{}\"", c))
                .collect();
            let per_in_j: Vec<String> = r
                .per_adapter_input_energy
                .iter()
                .map(|e| format!("{:.9e}", e))
                .collect();
            format!(
                r#"    {{
      "base_name": "{}",
      "d_out": {},
      "d_in": {},
      "output_rank": {},
      "participating_adapters": {},
      "input_energy_sum": {:.9e},
      "per_adapter_input_energy": [{}],
      "post_dare_energy_sum": {:.9e},
      "dare_rescale_energy_ratio": {:.6},
      "post_sparsify_energy_sum": {:.9e},
      "pruned_trim_energy": {:.9e},
      "trim_retained_ratio": {:.6},
      "elected_energy_sum": {:.9e},
      "discarded_conflict_energy": {:.9e},
      "discarded_conflict_ratio": {:.6},
      "active_coords": {},
      "overlap_coords": {},
      "sign_conflict_coords": {},
      "sign_conflict_rate": {:.6},
      "elected_positive_coords": {},
      "elected_negative_coords": {},
      "dense_merged_energy": {:.9e},
      "pre_cap_low_rank_energy": {:.9e},
      "svd_truncation_loss_energy": {:.9e},
      "svd_retained_energy_ratio": {:.6},
      "low_rank_output_energy": {:.9e},
      "dense_rank_99": {},
      "triggered_rules": [{}]
    }}"#,
                json_escape(&r.base_name),
                r.d_out,
                r.d_in,
                r.output_rank,
                r.participating_adapters,
                r.input_energy_sum,
                per_in_j.join(", "),
                r.post_dare_energy_sum,
                r.dare_rescale_energy_ratio,
                r.post_sparsify_energy_sum,
                r.pruned_trim_energy,
                r.trim_retained_ratio,
                r.elected_energy_sum,
                r.discarded_conflict_energy,
                r.discarded_conflict_ratio,
                r.active_coords,
                r.overlap_coords,
                r.sign_conflict_coords,
                r.sign_conflict_rate,
                r.elected_positive_coords,
                r.elected_negative_coords,
                r.dense_merged_energy,
                r.pre_cap_low_rank_energy,
                r.svd_truncation_loss_energy,
                r.svd_retained_energy_ratio,
                r.low_rank_output_energy,
                r.dense_rank_99,
                rules_j.join(", ")
            )
        })
        .collect();

    let diag_json: Vec<String> = summary
        .diagnostics
        .iter()
        .map(|d| {
            format!(
                r#"    {{
      "rule_id": "{}",
      "severity": "{}",
      "layer_name": "{}",
      "artifact_uri": "{}",
      "message": "{}",
      "sign_conflict_rate": {:.6},
      "discarded_conflict_ratio": {:.6},
      "svd_retained_ratio": {:.6},
      "trim_retained_ratio": {:.6},
      "dare_rescale_ratio": {:.6}
    }}"#,
                d.rule_id,
                d.severity.as_str(),
                json_escape(&d.layer_name),
                json_escape(&d.artifact_uri),
                json_escape(&d.message),
                d.sign_conflict_rate,
                d.discarded_conflict_ratio,
                d.svd_retained_ratio,
                d.trim_retained_ratio,
                d.dare_rescale_ratio
            )
        })
        .collect();

    format!(
        r#"{{
  "mode": "{}",
  "density": {:.6},
  "drop_rate": {:.6},
  "seed": {},
  "target_rank": {},
  "aggregation": "{}",
  "preserve_spectral_norm": {},
  "adapters": [{}],
  "energy_ledger": {{
    "total_input_energy": {:.9e},
    "total_post_dare_energy": {:.9e},
    "total_post_sparsify_energy": {:.9e},
    "total_pruned_trim_energy": {:.9e},
    "total_elected_energy": {:.9e},
    "total_discarded_conflict_energy": {:.9e},
    "total_dense_merged_energy": {:.9e},
    "total_pre_cap_low_rank_energy": {:.9e},
    "total_svd_truncation_loss": {:.9e},
    "total_low_rank_output_energy": {:.9e}
  }},
  "mean_sign_conflict_rate": {:.6},
  "mean_svd_retained_ratio": {:.6},
  "diagnostics": [
{}
  ],
  "layers": [
{}
  ]
}}"#,
        summary.mode.as_str(),
        summary.density,
        summary.drop_rate,
        summary.seed,
        summary.target_rank,
        summary.aggregation.as_str(),
        summary.preserve_spectral_norm,
        adapters_json.join(", "),
        summary.total_input_energy,
        summary.total_post_dare_energy,
        summary.total_post_sparsify_energy,
        summary.total_pruned_trim_energy,
        summary.total_elected_energy,
        summary.total_discarded_conflict_energy,
        summary.total_dense_merged_energy,
        summary.total_pre_cap_low_rank_energy,
        summary.total_svd_truncation_loss,
        summary.total_low_rank_output_energy,
        summary.mean_sign_conflict_rate,
        summary.mean_svd_retained_ratio,
        diag_json.join(",\n"),
        layers_json.join(",\n")
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sarif::build_ties_dare_sarif;

    fn make_rank1_layer(base_name: &str, col_b: &[f64], row_a: &[f64]) -> LoraLayer {
        let d_out = col_b.len();
        let d_in = row_a.len();
        let mut b = Tensor2D::zeros(d_out, 1, DType::F32);
        let mut a = Tensor2D::zeros(1, d_in, DType::F32);
        b.data.copy_from_slice(col_b);
        a.data.copy_from_slice(row_a);
        LoraLayer {
            base_name: base_name.to_string(),
            a_key: format!("{}.lora_A.weight", base_name),
            b_key: format!("{}.lora_B.weight", base_name),
            a,
            b,
        }
    }

    fn make_input(name: &str, layers: Vec<LoraLayer>, weight: f64) -> AdapterMergeInput {
        AdapterMergeInput {
            name: name.to_string(),
            layers,
            alphas: BTreeMap::new(),
            peft_config: None,
            weight,
        }
    }

    #[test]
    fn test_ties_three_way_joint_sign_election_beats_sequential_fold() {
        // Adapter 0 has -1.0 everywhere; Adapters 1 and 2 have +0.8 everywhere.
        // Sequential 2-way fold ((0, 1) -> 2) would pick (-1.0 + 0.8 = -0.2 < 0) -> -1.0,
        // and then (-1.0 + 0.8 = -0.2 < 0) -> -1.0 (wrong sign!).
        // True joint 3-way TIES elects sgn(-1.0 + 0.8 + 0.8 = +0.6 > 0) -> +1,
        // vetoing Adapter 0 and averaging Adapters 1 and 2 to +0.8!
        let d = 8;
        let ones = vec![1.0; d];
        let l0 = make_rank1_layer("model.layers.0.self_attn.q_proj", &vec![-1.0; d], &ones);
        let l1 = make_rank1_layer("model.layers.0.self_attn.q_proj", &vec![0.8; d], &ones);
        let l2 = make_rank1_layer("model.layers.0.self_attn.q_proj", &vec![0.8; d], &ones);

        let inputs = vec![
            make_input("a0.safetensors", vec![l0], 1.0),
            make_input("a1.safetensors", vec![l1], 1.0),
            make_input("a2.safetensors", vec![l2], 1.0),
        ];

        let mut cfg = TiesDareConfig::new(TiesDareMode::Ties, 4);
        cfg.density = 1.0;

        let summary = compose_adapters_ties_dare(&inputs, &cfg).expect("3-way ties");
        assert_eq!(summary.layer_reports.len(), 1);
        let rep = &summary.layer_reports[0];

        assert_eq!(rep.sign_conflict_coords, d * d);
        assert_eq!(rep.elected_positive_coords, d * d);
        assert_eq!(rep.elected_negative_coords, 0);
        assert!(rep.triggered_rules.contains(&LS035_TIES_SIGN_CONFLICT));

        // Reconstruct merged B*A and verify every entry equals +0.8
        let merged_dw = matmul(&summary.merged_layers[0].b, &summary.merged_layers[0].a);
        for &val in &merged_dw.data {
            assert!(
                (val - 0.8).abs() < 1e-10,
                "Expected +0.8 from joint 3-way TIES majority vote, got {}",
                val
            );
        }

        // Exact energy conservation checks
        let expected_vetoed = (d * d) as f64 * 1.0;
        let expected_elected = (d * d) as f64 * (0.64 + 0.64);
        assert!((rep.discarded_conflict_energy - expected_vetoed).abs() < 1e-10);
        assert!((rep.elected_energy_sum - expected_elected).abs() < 1e-10);
        assert!(
            (rep.post_sparsify_energy_sum - (rep.elected_energy_sum + rep.discarded_conflict_energy))
                .abs()
                < 1e-10
        );
    }

    #[test]
    fn test_ties_disjoint_support_preserves_full_single_task_magnitude() {
        // Adapter 1 updates rows 0..4 with +2.0; Adapter 2 updates rows 4..8 with -3.0.
        // DisjointMean divides by |A_{ij}| = 1 on every active entry, preserving +2.0 and -3.0
        // without the 0.5x dilution suffered by naive uniform averaging.
        let mut col1 = vec![0.0; 8];
        let mut col2 = vec![0.0; 8];
        for i in 0..4 {
            col1[i] = 2.0;
            col2[i + 4] = -3.0;
        }
        let row_ones = vec![1.0; 8];
        let l1 = make_rank1_layer("layers.0.mlp.up_proj", &col1, &row_ones);
        let l2 = make_rank1_layer("layers.0.mlp.up_proj", &col2, &row_ones);

        let inputs = vec![
            make_input("t1.safetensors", vec![l1], 1.0),
            make_input("t2.safetensors", vec![l2], 1.0),
        ];
        let mut cfg = TiesDareConfig::new(TiesDareMode::Ties, 4);
        cfg.density = 0.5; // Exactly half of entries in each adapter are non-zero!

        let summary = compose_adapters_ties_dare(&inputs, &cfg).expect("disjoint ties");
        let rep = &summary.layer_reports[0];
        assert_eq!(rep.overlap_coords, 0);
        assert_eq!(rep.sign_conflict_coords, 0);
        assert!((rep.trim_retained_ratio - 1.0).abs() < 1e-12);
        assert!(rep.discarded_conflict_energy.abs() < 1e-12);

        let merged_dw = matmul(&summary.merged_layers[0].b, &summary.merged_layers[0].a);
        for r in 0..4 {
            for c in 0..8 {
                assert!((merged_dw.get(r, c) - 2.0).abs() < 1e-10);
                assert!((merged_dw.get(r + 4, c) - (-3.0)).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_exact_frobenius_energy_conservation_across_all_stages() {
        let mut rng = Rng64::new(20261118);
        let d_out = 36;
        let d_in = 32;
        let make_rand_layer = |rng: &mut Rng64, r: usize| -> LoraLayer {
            let mut b = Tensor2D::zeros(d_out, r, DType::F32);
            let mut a = Tensor2D::zeros(r, d_in, DType::F32);
            for v in &mut b.data {
                *v = rng.next_gaussian() * 0.25;
            }
            for v in &mut a.data {
                *v = rng.next_gaussian() * 0.25;
            }
            LoraLayer {
                base_name: "model.layers.2.self_attn.v_proj".to_string(),
                a_key: "model.layers.2.self_attn.v_proj.lora_A.weight".to_string(),
                b_key: "model.layers.2.self_attn.v_proj.lora_B.weight".to_string(),
                a,
                b,
            }
        };

        let inputs = vec![
            make_input("a.safetensors", vec![make_rand_layer(&mut rng, 6)], 0.9),
            make_input("b.safetensors", vec![make_rand_layer(&mut rng, 8)], 1.1),
            make_input("c.safetensors", vec![make_rand_layer(&mut rng, 4)], 0.7),
        ];

        for mode in [
            TiesDareMode::Ties,
            TiesDareMode::DareTies,
            TiesDareMode::DareLinear,
        ] {
            let mut cfg = TiesDareConfig::new(mode, 6);
            cfg.density = 0.6;
            cfg.drop_rate = if mode.uses_dare() { 0.35 } else { 0.0 };
            cfg.seed = 99;

            let summary = compose_adapters_ties_dare(&inputs, &cfg).expect("compose");
            let rep = &summary.layer_reports[0];

            // Identity 1: Post-DARE energy == Post-Sparsify energy + Pruned-Trim energy
            assert!(
                (rep.post_dare_energy_sum - (rep.post_sparsify_energy_sum + rep.pruned_trim_energy))
                    .abs()
                    < 1e-10,
                "Trim conservation failed for {}",
                mode.as_str()
            );

            // Identity 2: Post-Sparsify energy == Elected energy + Discarded conflict energy
            assert!(
                (rep.post_sparsify_energy_sum
                    - (rep.elected_energy_sum + rep.discarded_conflict_energy))
                    .abs()
                    < 1e-10,
                "Sign-election conservation failed for {}",
                mode.as_str()
            );

            // Identity 3: Dense merged energy == Pre-cap rank-k energy + SVD truncation loss
            assert!(
                (rep.dense_merged_energy
                    - (rep.pre_cap_low_rank_energy + rep.svd_truncation_loss_energy))
                    .abs()
                    < 1e-10,
                "Eckart-Young SVD conservation failed for {}",
                mode.as_str()
            );

            // Verify actual output matrix Frobenius norm matches reported low_rank_output_energy
            let actual_dw = matmul(&summary.merged_layers[0].b, &summary.merged_layers[0].a);
            assert!(
                (actual_dw.frobenius_norm_sq() - rep.low_rank_output_energy).abs() < 1e-9,
                "Output factor energy mismatch for {}",
                mode.as_str()
            );
        }
    }

    #[test]
    fn test_dare_independent_masks_and_unbiased_expectation() {
        let d_out = 64;
        let d_in = 64;
        let numel = d_out * d_in;
        let drop_rate = 0.70; // keep prob = 0.30, pairwise joint survival = 0.09
        let layer_hash = fnv1a_64("layers.0.mlp.gate_proj");

        let mut surv_0 = 0usize;
        let mut surv_1 = 0usize;
        let mut surv_both = 0usize;
        for r in 0..d_out {
            for c in 0..d_in {
                let k0 = dare_coordinate_u01(42, 0, layer_hash, r, c) >= drop_rate;
                let k1 = dare_coordinate_u01(42, 1, layer_hash, r, c) >= drop_rate;
                if k0 {
                    surv_0 += 1;
                }
                if k1 {
                    surv_1 += 1;
                }
                if k0 && k1 {
                    surv_both += 1;
                }
            }
        }
        let p0 = (surv_0 as f64) / (numel as f64);
        let p1 = (surv_1 as f64) / (numel as f64);
        let p_both = (surv_both as f64) / (numel as f64);
        assert!((p0 - 0.30).abs() < 0.03, "p0={}", p0);
        assert!((p1 - 0.30).abs() < 0.03, "p1={}", p1);
        assert!(
            (p_both - 0.09).abs() < 0.025,
            "Pairwise collision should drop quadratically to (1-p)^2=0.09, got {}",
            p_both
        );

        // Monte Carlo check of unbiasedness E[tilde_delta] = delta across seeds
        let orig: Vec<f64> = (0..256).map(|i| ((i as f64) * 0.13).sin()).collect();
        let mut mean_acc = vec![0.0_f64; 256];
        let trials = 2500usize;
        for s in 0..trials {
            let mut buf = orig.clone();
            apply_dare_drop_rescale_in_place(&mut buf, 16, 16, 0.5, s as u64, 0, layer_hash);
            for i in 0..256 {
                mean_acc[i] += buf[i] / (trials as f64);
            }
        }
        let max_bias = orig
            .iter()
            .zip(mean_acc.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_bias < 0.09,
            "DARE rescaling should be unbiased, max_bias={}",
            max_bias
        );
    }

    #[test]
    fn test_sarif_diagnostics_ls036_ls037_ls038_and_valid_sarif_json() {
        // 1. LS036: Ultra-sparse spike tensor wiped out by high drop_rate DARE
        let mut spike_b = vec![0.0; 16];
        let mut spike_a = vec![0.0; 16];
        spike_b[0] = 10.0;
        spike_a[0] = 10.0;
        // Find a seed that drops (0, 0) for both adapters
        let lh = fnv1a_64("layers.0.attn.q_proj");
        let wipe_seed = (0u64..1000)
            .find(|&s| {
                dare_coordinate_u01(s, 0, lh, 0, 0) < 0.90
                    && dare_coordinate_u01(s, 1, lh, 0, 0) < 0.90
            })
            .expect("find seed dropping (0,0)");

        let spike_inputs = vec![
            make_input(
                "s1.safetensors",
                vec![make_rank1_layer("layers.0.attn.q_proj", &spike_b, &spike_a)],
                1.0,
            ),
            make_input(
                "s2.safetensors",
                vec![make_rank1_layer("layers.0.attn.q_proj", &spike_b, &spike_a)],
                1.0,
            ),
        ];
        let mut cfg_dare = TiesDareConfig::new(TiesDareMode::DareTies, 2);
        cfg_dare.drop_rate = 0.90;
        cfg_dare.seed = wipe_seed;
        let sum_ls036 = compose_adapters_ties_dare(&spike_inputs, &cfg_dare).unwrap();
        assert!(
            sum_ls036.layer_reports[0]
                .triggered_rules
                .contains(&LS036_DARE_SIGNAL_DISTORTION)
        );

        // 2. LS037 (rank scattering at k=1) & LS038 (aggressive trim density=0.10 on uniform matrix)
        let ones = vec![1.0; 16];
        let mut rng = Rng64::new(77);
        let make_iso = |rng: &mut Rng64| -> LoraLayer {
            let mut b = Tensor2D::zeros(16, 8, DType::F32);
            let mut a = Tensor2D::zeros(8, 16, DType::F32);
            for v in &mut b.data {
                *v = rng.next_gaussian();
            }
            for v in &mut a.data {
                *v = rng.next_gaussian();
            }
            LoraLayer {
                base_name: "layers.0.mlp.down_proj".to_string(),
                a_key: "layers.0.mlp.down_proj.lora_A.weight".to_string(),
                b_key: "layers.0.mlp.down_proj.lora_B.weight".to_string(),
                a,
                b,
            }
        };

        let dense_inputs = vec![
            make_input(
                "u1.safetensors",
                vec![
                    make_rank1_layer("layers.0.attn.k_proj", &ones, &ones),
                    make_iso(&mut rng),
                ],
                1.0,
            ),
            make_input(
                "u2.safetensors",
                vec![
                    make_rank1_layer("layers.0.attn.k_proj", &ones, &ones),
                    make_iso(&mut rng),
                ],
                1.0,
            ),
        ];

        let mut cfg_trim = TiesDareConfig::new(TiesDareMode::Ties, 1);
        cfg_trim.density = 0.15; // On uniform k_proj, 15% density retains only ~15% energy (< 50% -> LS038)
        let sum_trim = compose_adapters_ties_dare(&dense_inputs, &cfg_trim).unwrap();
        let all_rules: Vec<&str> = sum_trim
            .diagnostics
            .iter()
            .map(|d| d.rule_id)
            .collect();
        assert!(all_rules.contains(&LS038_EXCESSIVE_TRIM_LOSS));
        assert!(all_rules.contains(&LS037_MASK_RANK_SCATTERING_LOSS));

        let sarif = build_ties_dare_sarif(&sum_trim, "1.6.0");
        assert!(sarif.contains("\"version\": \"2.1.0\""));
        assert!(sarif.contains("\"id\": \"LS035\""));
        assert!(sarif.contains("\"id\": \"LS036\""));
        assert!(sarif.contains("\"id\": \"LS037\""));
        assert!(sarif.contains("\"id\": \"LS038\""));
    }

    #[test]
    fn test_subspace_iteration_truncated_dense_svd_matches_exact_jacobi() {
        // Dimensions > 192 trigger the randomized subspace iteration branch!
        let d_out = 220;
        let d_in = 200;
        let true_r = 10;
        let mut rng = Rng64::new(314159);
        let mut b = Tensor2D::zeros(d_out, true_r, DType::F32);
        let mut a = Tensor2D::zeros(true_r, d_in, DType::F32);
        for c in 0..true_r {
            let weight = 5.0 * (0.7_f64).powi(c as i32);
            for r in 0..d_out {
                b.set(r, c, rng.next_gaussian() * weight);
            }
            for col in 0..d_in {
                a.set(c, col, rng.next_gaussian());
            }
        }
        let dense = matmul(&b, &a);
        let exact_sigma = qr_core_singular_values(&b, &a);
        let fast_svd = truncated_dense_svd(&dense, 6, 42);

        for i in 0..6 {
            let rel_err = (fast_svd.sigma[i] - exact_sigma[i]).abs() / exact_sigma[0];
            assert!(
                rel_err < 1e-9,
                "Subspace SVD singular value {} mismatch: got {} vs exact {}",
                i,
                fast_svd.sigma[i],
                exact_sigma[i]
            );
        }
    }

    #[test]
    fn test_preserve_spectral_norm_json_table_and_merge_integration() {
        use crate::merge::merge_multiple_adapters;

        let ones = vec![2.0; 12];
        let l1 = make_rank1_layer("layers.0.attn.o_proj", &ones, &ones);
        let l2 = make_rank1_layer("layers.0.attn.o_proj", &ones, &ones);
        let s1_max = qr_core_singular_values(&l1.b, &l1.a)[0];

        let inputs = vec![
            make_input("a1.safetensors", vec![l1.clone()], 3.0),
            make_input("a2.safetensors", vec![l2.clone()], 3.0),
        ];
        let mut cfg = TiesDareConfig::new(TiesDareMode::DareTies, 2);
        cfg.drop_rate = 0.25;
        cfg.preserve_spectral_norm = true;

        let summary = compose_adapters_ties_dare(&inputs, &cfg).expect("compose preserve norm");
        let s_merged = qr_core_singular_values(
            &summary.merged_layers[0].b,
            &summary.merged_layers[0].a,
        )[0];
        assert!(
            s_merged <= s1_max + 1e-9,
            "preserve_spectral_norm failed: {} > {}",
            s_merged,
            s1_max
        );

        let table = render_ties_dare_table(&summary);
        assert!(table.contains("Frobenius Energy Ledger"));
        let json = render_ties_dare_json(&summary);
        assert!(json.contains("\"energy_ledger\""));
        assert!(json.contains("\"mode\": \"dare-ties\""));

        let multi = merge_multiple_adapters(
            &[vec![l1], vec![l2]],
            &[1.0, 1.0],
            "dare-ties",
            2,
            &[],
            true,
        )
        .expect("merge_multiple_adapters dare-ties");
        assert_eq!(multi.len(), 1);
    }
}

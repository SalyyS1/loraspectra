use std::collections::BTreeMap;

use crate::audit::{compute_truncation_bounds, IssueSeverity, IntegrityIssue, SvdTruncationBounds};
use crate::linalg::{qr_core_singular_values, qr_core_svd_truncated, svd_to_lora_factors};
use crate::safetensors::{DType, LoraLayer, Tensor2D};

/// Dettmers et al. (NeurIPS 2023, QLoRA) exact 4-bit NormalFloat (`NF4`) 16-level codebook,
/// constructed from equal-mass standard normal quantiles normalized to `[-1, 1]` with exact `0.0`.
pub const NF4_CODEBOOK: [f64; 16] = [
    -1.0,
    -0.6961928009986877,
    -0.5250730514526367,
    -0.39491748809814453,
    -0.28444138169288635,
    -0.18477343022823334,
    -0.09105003625154495,
    0.0,
    0.07958029955625534,
    0.16093020141124725,
    0.24611230194568634,
    0.33791524171829224,
    0.44070982933044434,
    0.5626170039176941,
    0.7229568362236023,
    1.0,
];

/// Canonical `bitsandbytes` 4-bit Floating-Point (`FP4`) 16-level codebook normalized to `[-1, 1]`.
/// Index layout matches `bitsandbytes.functional.create_fp4_map()` nibble encoding, sorted here for
/// nearest-neighbor lookup and indexed directly by nibble when unpacking `.safetensors` buffers.
pub const FP4_SORTED_LEVELS: [f64; 15] = [
    -1.0,
    -0.6666666865348816,
    -0.5,
    -0.3333333432674408,
    -0.25,
    -0.1666666716337204,
    -0.0625,
    0.0,
    0.0625,
    0.1666666716337204,
    0.25,
    0.3333333432674408,
    0.5,
    0.6666666865348816,
    1.0,
];

/// Direct 16-entry `bitsandbytes` nibble-indexed `FP4` lookup table (`create_fp4_map()`).
pub const FP4_BNB_NIBBLE_MAP: [f64; 16] = [
    0.0,
    0.0625,
    0.6666666865348816,
    1.0,
    0.3333333432674408,
    0.5,
    0.1666666716337204,
    0.25,
    0.0,
    -0.0625,
    -0.6666666865348816,
    -1.0,
    -0.3333333432674408,
    -0.5,
    -0.1666666716337204,
    -0.25,
];

/// Supported low-bit quantization formats for spectral reconstruction diagnostics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantScheme {
    Nf4,
    Fp4,
    Int4,
    Int8,
}

impl QuantScheme {
    pub fn as_str(&self) -> &'static str {
        match self {
            QuantScheme::Nf4 => "NF4",
            QuantScheme::Fp4 => "FP4",
            QuantScheme::Int4 => "INT4",
            QuantScheme::Int8 => "INT8",
        }
    }

    pub fn bits(&self) -> usize {
        match self {
            QuantScheme::Nf4 | QuantScheme::Fp4 | QuantScheme::Int4 => 4,
            QuantScheme::Int8 => 8,
        }
    }

    pub fn from_name(name: &str) -> Result<Self, String> {
        match name.trim().to_ascii_lowercase().as_str() {
            "nf4" | "bnb_nf4" | "qlora" => Ok(QuantScheme::Nf4),
            "fp4" | "e2m1" | "bnb_fp4" => Ok(QuantScheme::Fp4),
            "int4" | "i4" | "q4" => Ok(QuantScheme::Int4),
            "int8" | "i8" | "q8" | "llm.int8" => Ok(QuantScheme::Int8),
            other => Err(format!(
                "Unsupported quantization scheme '{}': expected nf4, fp4, int4, or int8",
                other
            )),
        }
    }
}

#[inline]
fn nearest_codebook_value(u: f64, levels: &[f64]) -> f64 {
    let mut best = levels[0];
    let mut best_err = (u - best).abs();
    for &lvl in &levels[1..] {
        let err = (u - lvl).abs();
        if err < best_err {
            best_err = err;
            best = lvl;
        }
    }
    best
}

/// Simulate blockwise quantization and dequantization on a 2D weight matrix (`block_size` defaults
/// to `64` in QLoRA / `bitsandbytes`).
pub fn quantize_dequantize_tensor(
    tensor: &Tensor2D,
    scheme: QuantScheme,
    block_size: usize,
) -> Tensor2D {
    let blk = block_size.max(1);
    let mut out = Vec::with_capacity(tensor.data.len());

    for chunk in tensor.data.chunks(blk) {
        let mut absmax = 0.0_f64;
        for &v in chunk {
            let av = v.abs();
            if av.is_finite() && av > absmax {
                absmax = av;
            }
        }
        if absmax <= 1e-30 {
            out.extend(std::iter::repeat(0.0).take(chunk.len()));
            continue;
        }
        let inv = 1.0 / absmax;
        match scheme {
            QuantScheme::Nf4 => {
                for &v in chunk {
                    let u = (v * inv).clamp(-1.0, 1.0);
                    out.push(nearest_codebook_value(u, &NF4_CODEBOOK) * absmax);
                }
            }
            QuantScheme::Fp4 => {
                for &v in chunk {
                    let u = (v * inv).clamp(-1.0, 1.0);
                    out.push(nearest_codebook_value(u, &FP4_SORTED_LEVELS) * absmax);
                }
            }
            QuantScheme::Int4 => {
                let step = absmax / 7.0;
                for &v in chunk {
                    let q = (v / step).round().clamp(-7.0, 7.0);
                    out.push(q * step);
                }
            }
            QuantScheme::Int8 => {
                let step = absmax / 127.0;
                for &v in chunk {
                    let q = (v / step).round().clamp(-127.0, 127.0);
                    out.push(q * step);
                }
            }
        }
    }

    Tensor2D {
        rows: tensor.rows,
        cols: tensor.cols,
        dtype: tensor.dtype,
        data: out,
    }
}

fn crest_factor(data: &[f64]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    let mut max_abs = 0.0_f64;
    let mut sq_sum = 0.0_f64;
    for &v in data {
        if v.is_finite() {
            max_abs = max_abs.max(v.abs());
            sq_sum += v * v;
        }
    }
    let rms = (sq_sum / (data.len() as f64)).sqrt();
    if rms <= 1e-30 {
        0.0
    } else {
        max_abs / rms
    }
}

/// Exact spectral and Frobenius error of `s * (B_1 * A_1 - B_2 * A_2)` computed via rank-`(r_1 + r_2)`
/// stacked QR-Core SVD in `O((r_1 + r_2)^2 * (d_out + d_in) + (r_1 + r_2)^3)` time without allocating
/// a dense `d_out x d_in` matrix. Returns `(spectral_error, frobenius_error)`.
pub fn exact_low_rank_difference_norms(
    b1: &Tensor2D,
    a1: &Tensor2D,
    scale1: f64,
    b2: &Tensor2D,
    a2: &Tensor2D,
    scale2: f64,
) -> (f64, f64) {
    assert_eq!(b1.rows, b2.rows, "d_out mismatch in low-rank difference");
    assert_eq!(a1.cols, a2.cols, "d_in mismatch in low-rank difference");
    assert_eq!(b1.cols, a1.rows, "rank mismatch in pair 1");
    assert_eq!(b2.cols, a2.rows, "rank mismatch in pair 2");

    let d_out = b1.rows;
    let d_in = a1.cols;
    let r1 = b1.cols;
    let r2 = b2.cols;
    let r_sum = r1 + r2;
    if r_sum == 0 || d_out == 0 || d_in == 0 {
        return (0.0, 0.0);
    }

    // Stacked B_diff: (d_out x (r1 + r2)) = [B1, -B2]
    let mut b_diff = Tensor2D::zeros(d_out, r_sum, DType::F32);
    for row in 0..d_out {
        let dst = &mut b_diff.data[row * r_sum..(row + 1) * r_sum];
        let src1 = &b1.data[row * r1..(row + 1) * r1];
        let src2 = &b2.data[row * r2..(row + 1) * r2];
        dst[..r1].copy_from_slice(src1);
        for j in 0..r2 {
            dst[r1 + j] = -src2[j];
        }
    }

    // Stacked A_diff: ((r1 + r2) x d_in) = [scale1 * A1; scale2 * A2]
    let mut a_diff = Tensor2D::zeros(r_sum, d_in, DType::F32);
    for row in 0..r1 {
        let dst = &mut a_diff.data[row * d_in..(row + 1) * d_in];
        let src = &a1.data[row * d_in..(row + 1) * d_in];
        for col in 0..d_in {
            dst[col] = scale1 * src[col];
        }
    }
    for row in 0..r2 {
        let dst = &mut a_diff.data[(r1 + row) * d_in..(r1 + row + 1) * d_in];
        let src = &a2.data[row * d_in..(row + 1) * d_in];
        for col in 0..d_in {
            dst[col] = scale2 * src[col];
        }
    }

    let err_sigma = qr_core_singular_values(&b_diff, &a_diff);
    let spec_err = err_sigma.first().copied().unwrap_or(0.0);
    let frob_err = err_sigma.iter().map(|&s| s * s).sum::<f64>().sqrt();
    (spec_err, frob_err)
}

/// Per-layer low-bit quantization (`NF4` / `FP4` / `INT4` / `INT8`) spectral reconstruction report.
#[derive(Debug, Clone)]
pub struct LayerQuantDiagnostics {
    pub base_name: String,
    pub scheme: QuantScheme,
    pub block_size: usize,
    pub rank: usize,
    pub signal_spectral_norm: f64,
    pub signal_frobenius_norm: f64,
    pub spectral_error: f64,
    pub relative_spectral_error: f64,
    pub frobenius_error: f64,
    pub relative_frobenius_error: f64,
    pub sqnr_db: f64,
    pub balanced_relative_frobenius_error: f64,
    pub max_crest_factor: f64,
    pub factor_norm_imbalance: f64,
}

/// Aggregate adapter quantization reconstruction summary across all LoRA layers.
#[derive(Debug, Clone)]
pub struct AdapterQuantSummary {
    pub scheme: QuantScheme,
    pub block_size: usize,
    pub layers: Vec<LayerQuantDiagnostics>,
    pub mean_sqnr_db: f64,
    pub min_sqnr_db: f64,
    pub mean_relative_frobenius_error: f64,
    pub max_relative_frobenius_error: f64,
    pub max_relative_spectral_error: f64,
    pub mean_balanced_relative_frobenius_error: f64,
}

/// Combined rank-`k` SVD truncation + low-bit factor quantization error certificate, providing
/// Weyl lower bounds, exact low-rank errors, and triangle-inequality upper bounds.
#[derive(Debug, Clone, Copy)]
pub struct CombinedTruncationQuantCertificate {
    pub truncation: SvdTruncationBounds,
    pub quant_spectral_error: f64,
    pub quant_frobenius_error: f64,
    pub exact_total_spectral_error: f64,
    pub exact_total_frobenius_error: f64,
    pub weyl_spectral_lower_bound: f64,
    pub triangle_spectral_upper_bound: f64,
    pub triangle_frobenius_upper_bound: f64,
}

/// Compute exact spectral reconstruction metrics for a single `LoraLayer` under `scheme` (`NF4`,
/// `FP4`, `INT4`, or `INT8`) with block size `block_size`, alongside SVD-rebalanced factor error.
pub fn analyze_layer_quantization(
    layer: &LoraLayer,
    effective_scale: f64,
    scheme: QuantScheme,
    block_size: usize,
) -> LayerQuantDiagnostics {
    let blk = block_size.max(1);
    let scale = if effective_scale.is_finite() && effective_scale != 0.0 {
        effective_scale
    } else {
        1.0
    };
    let abs_scale = scale.abs();

    let raw_sigma = qr_core_singular_values(&layer.b, &layer.a);
    let signal_spectral_norm = raw_sigma.first().copied().unwrap_or(0.0) * abs_scale;
    let signal_frobenius_norm =
        raw_sigma.iter().map(|&s| s * s).sum::<f64>().sqrt() * abs_scale;

    let a_q = quantize_dequantize_tensor(&layer.a, scheme, blk);
    let b_q = quantize_dequantize_tensor(&layer.b, scheme, blk);
    let (spectral_error, frobenius_error) =
        exact_low_rank_difference_norms(&layer.b, &layer.a, scale, &b_q, &a_q, scale);

    let relative_frobenius_error = if signal_frobenius_norm > 1e-18 {
        frobenius_error / signal_frobenius_norm
    } else {
        0.0
    };
    let relative_spectral_error = if signal_spectral_norm > 1e-18 {
        spectral_error / signal_spectral_norm
    } else {
        0.0
    };
    let sqnr_db = if signal_frobenius_norm <= 1e-18 {
        0.0
    } else if frobenius_error <= 1e-18 {
        120.0
    } else {
        (20.0 * (signal_frobenius_norm / frobenius_error).log10()).clamp(-60.0, 120.0)
    };

    // Compare against SVD-balanced factors (B_bal = U * sqrt(|s| * Sigma), A_bal = sign(s) * sqrt(|s| * Sigma) * V^T)
    let r = layer.rank().max(1);
    let balanced_relative_frobenius_error =
        if signal_frobenius_norm > 1e-18 && r <= layer.d_out() && r <= layer.d_in() {
            let mut svd = qr_core_svd_truncated(&layer.b, &layer.a, r);
            for s in &mut svd.sigma {
                *s *= abs_scale;
            }
            let (mut b_bal, a_bal) = svd_to_lora_factors(&svd, r, layer.a.dtype);
            if scale < 0.0 {
                for v in &mut b_bal.data {
                    *v = -*v;
                }
            }
            let b_bal_q = quantize_dequantize_tensor(&b_bal, scheme, blk);
            let a_bal_q = quantize_dequantize_tensor(&a_bal, scheme, blk);
            let (_, bal_frob_err) =
                exact_low_rank_difference_norms(&layer.b, &layer.a, scale, &b_bal_q, &a_bal_q, 1.0);
            bal_frob_err / signal_frobenius_norm
        } else {
            relative_frobenius_error
        };

    let crest_a = crest_factor(&layer.a.data);
    let crest_b = crest_factor(&layer.b.data);
    let max_crest_factor = crest_a.max(crest_b);

    let norm_a = layer.a.frobenius_norm_sq().sqrt();
    let norm_b = layer.b.frobenius_norm_sq().sqrt();
    let factor_norm_imbalance = if norm_a > 1e-15 && norm_b > 1e-15 {
        (norm_b / norm_a).max(norm_a / norm_b)
    } else {
        1.0
    };

    LayerQuantDiagnostics {
        base_name: layer.base_name.clone(),
        scheme,
        block_size: blk,
        rank: layer.rank(),
        signal_spectral_norm,
        signal_frobenius_norm,
        spectral_error,
        relative_spectral_error,
        frobenius_error,
        relative_frobenius_error,
        sqnr_db,
        balanced_relative_frobenius_error,
        max_crest_factor,
        factor_norm_imbalance,
    }
}

/// Analyze quantization reconstruction across all LoRA layers and emit `LS023` integrity issues
/// when reconstruction distortion exceeds safe thresholds or factor imbalance degrades SQNR.
pub fn analyze_adapter_quantization(
    layers: &[LoraLayer],
    scales: &[f64],
    scheme: QuantScheme,
    block_size: usize,
) -> (AdapterQuantSummary, Vec<IntegrityIssue>) {
    let blk = block_size.max(1);
    let mut reports = Vec::with_capacity(layers.len());
    let mut issues = Vec::new();

    for (idx, layer) in layers.iter().enumerate() {
        let s = scales.get(idx).copied().unwrap_or(1.0);
        let rep = analyze_layer_quantization(layer, s, scheme, blk);
        reports.push(rep);
    }

    let mut high_err_layers: Vec<&LayerQuantDiagnostics> = Vec::new();
    let mut imbalanced_layers: Vec<&LayerQuantDiagnostics> = Vec::new();

    for r in &reports {
        if r.signal_frobenius_norm <= 1e-12 {
            continue;
        }
        if r.relative_frobenius_error > 0.25 || r.relative_spectral_error > 0.30 {
            high_err_layers.push(r);
        } else if r.relative_frobenius_error > 0.12
            && r.balanced_relative_frobenius_error > 1e-12
            && r.relative_frobenius_error > 1.75 * r.balanced_relative_frobenius_error
        {
            imbalanced_layers.push(r);
        }
    }

    if !high_err_layers.is_empty() {
        let worst = high_err_layers
            .iter()
            .copied()
            .max_by(|a, b| {
                a.relative_frobenius_error
                    .total_cmp(&b.relative_frobenius_error)
            })
            .unwrap();
        let sev = if worst.relative_frobenius_error > 0.50 {
            IssueSeverity::Critical
        } else {
            IssueSeverity::Warning
        };
        issues.push(IntegrityIssue {
            code: "QUANT_RECONSTRUCTION_HIGH_ERROR",
            severity: sev,
            subject: worst.base_name.clone(),
            message: format!(
                "{} layer(s) suffer severe {}/block-{} quantization distortion (worst {}: rel_frob={:.1}%, rel_spec={:.1}%, SQNR={:.1} dB, crest={:.2})",
                high_err_layers.len(),
                scheme.as_str(),
                blk,
                worst.base_name,
                worst.relative_frobenius_error * 100.0,
                worst.relative_spectral_error * 100.0,
                worst.sqnr_db,
                worst.max_crest_factor
            ),
        });
    }

    if !imbalanced_layers.is_empty() {
        let worst = imbalanced_layers
            .iter()
            .copied()
            .max_by(|a, b| {
                (a.relative_frobenius_error / a.balanced_relative_frobenius_error.max(1e-12))
                    .total_cmp(
                        &(b.relative_frobenius_error
                            / b.balanced_relative_frobenius_error.max(1e-12)),
                    )
            })
            .unwrap();
        issues.push(IntegrityIssue {
            code: "QUANT_IMBALANCED_FACTOR_LOSS",
            severity: IssueSeverity::Warning,
            subject: worst.base_name.clone(),
            message: format!(
                "{} layer(s) lose excess {}/block-{} precision from imbalanced (B, A) scaling or channel outliers (e.g. {}: raw rel_frob={:.1}% vs SVD-balanced {:.1}%, norm ratio {:.1}x); rebalance via `loraspectra compress --energy 1.0`",
                imbalanced_layers.len(),
                scheme.as_str(),
                blk,
                worst.base_name,
                worst.relative_frobenius_error * 100.0,
                worst.balanced_relative_frobenius_error * 100.0,
                worst.factor_norm_imbalance
            ),
        });
    }

    let n = reports.len().max(1) as f64;
    let mean_sqnr_db = reports.iter().map(|r| r.sqnr_db).sum::<f64>() / n;
    let min_sqnr_db = reports
        .iter()
        .map(|r| r.sqnr_db)
        .fold(f64::INFINITY, f64::min);
    let mean_relative_frobenius_error = reports
        .iter()
        .map(|r| r.relative_frobenius_error)
        .sum::<f64>()
        / n;
    let max_relative_frobenius_error = reports
        .iter()
        .map(|r| r.relative_frobenius_error)
        .fold(0.0_f64, f64::max);
    let max_relative_spectral_error = reports
        .iter()
        .map(|r| r.relative_spectral_error)
        .fold(0.0_f64, f64::max);
    let mean_balanced_relative_frobenius_error = reports
        .iter()
        .map(|r| r.balanced_relative_frobenius_error)
        .sum::<f64>()
        / n;

    (
        AdapterQuantSummary {
            scheme,
            block_size: blk,
            layers: reports,
            mean_sqnr_db,
            min_sqnr_db: if min_sqnr_db.is_finite() {
                min_sqnr_db
            } else {
                0.0
            },
            mean_relative_frobenius_error,
            max_relative_frobenius_error,
            max_relative_spectral_error,
            mean_balanced_relative_frobenius_error,
        },
        issues,
    )
}

/// Compute a joint certificate bounding the error of rank-`k` SVD truncation followed by blockwise
/// quantization `DeltaW -> \widehat{DeltaW_k}`, verifying Weyl's lower bound and the triangle
/// inequality upper bound without dense matrix allocations.
pub fn compute_combined_truncation_quant_certificate(
    layer: &LoraLayer,
    effective_scale: f64,
    rank_k: usize,
    scheme: QuantScheme,
    block_size: usize,
) -> CombinedTruncationQuantCertificate {
    let scale = if effective_scale.is_finite() && effective_scale != 0.0 {
        effective_scale
    } else {
        1.0
    };
    let abs_scale = scale.abs();
    let k = rank_k.max(1).min(layer.rank().max(1));
    let blk = block_size.max(1);

    let mut svd = qr_core_svd_truncated(&layer.b, &layer.a, k);
    for s in &mut svd.sigma {
        *s *= abs_scale;
    }
    let truncation = compute_truncation_bounds(&svd.sigma, k);

    let (mut b_k, a_k) = svd_to_lora_factors(&svd, k, layer.a.dtype);
    if scale < 0.0 {
        for v in &mut b_k.data {
            *v = -*v;
        }
    }
    let b_kq = quantize_dequantize_tensor(&b_k, scheme, blk);
    let a_kq = quantize_dequantize_tensor(&a_k, scheme, blk);

    let (quant_spectral_error, quant_frobenius_error) =
        exact_low_rank_difference_norms(&b_k, &a_k, 1.0, &b_kq, &a_kq, 1.0);
    let (exact_total_spectral_error, exact_total_frobenius_error) =
        exact_low_rank_difference_norms(&layer.b, &layer.a, scale, &b_kq, &a_kq, 1.0);

    let weyl_spectral_lower_bound = (truncation.spectral_error - quant_spectral_error).abs();
    let triangle_spectral_upper_bound = truncation.spectral_error + quant_spectral_error;
    let triangle_frobenius_upper_bound = truncation.frobenius_error + quant_frobenius_error;

    CombinedTruncationQuantCertificate {
        truncation,
        quant_spectral_error,
        quant_frobenius_error,
        exact_total_spectral_error,
        exact_total_frobenius_error,
        weyl_spectral_lower_bound,
        triangle_spectral_upper_bound,
        triangle_frobenius_upper_bound,
    }
}

#[derive(Debug, Clone)]
struct BnbQuantState {
    quant_type: QuantScheme,
    rows: usize,
    cols: usize,
    blocksize: usize,
}

fn parse_bnb_quant_state_bytes(raw_u8: &[f64]) -> Option<BnbQuantState> {
    let bytes: Vec<u8> = raw_u8
        .iter()
        .map(|&v| v.round().clamp(0.0, 255.0) as u8)
        .collect();
    let s = std::str::from_utf8(&bytes).ok()?;
    let lower = s.to_ascii_lowercase();
    let quant_type = if lower.contains("\"fp4\"") {
        QuantScheme::Fp4
    } else if lower.contains("\"int4\"") {
        QuantScheme::Int4
    } else {
        QuantScheme::Nf4
    };

    // Parse "shape":[rows,cols]
    let shape_pos = s.find("\"shape\"")?;
    let lb = s[shape_pos..].find('[')? + shape_pos + 1;
    let rb = s[lb..].find(']')? + lb;
    let dims: Vec<usize> = s[lb..rb]
        .split(',')
        .filter_map(|p| p.trim().parse::<usize>().ok())
        .collect();
    if dims.len() != 2 || dims[0] == 0 || dims[1] == 0 {
        return None;
    }

    let blocksize = if let Some(bp) = s.find("\"blocksize\"") {
        let colon = s[bp..].find(':')? + bp + 1;
        let num_str: String = s[colon..]
            .chars()
            .skip_while(|c| c.is_whitespace())
            .take_while(|c| c.is_ascii_digit())
            .collect();
        num_str.parse::<usize>().ok().filter(|&b| b > 0).unwrap_or(64)
    } else {
        64
    };

    Some(BnbQuantState {
        quant_type,
        rows: dims[0],
        cols: dims[1],
        blocksize,
    })
}

fn resolve_absmax_values(
    weight_key: &str,
    absmax_tensor: &Tensor2D,
    all_tensors: &BTreeMap<String, Tensor2D>,
) -> Vec<f64> {
    if absmax_tensor.dtype != DType::U8 {
        return absmax_tensor.data.clone();
    }
    // Double quantization: absmax is U8 indices into nested_quant_map (256 floats) scaled by nested_absmax + nested_offset
    let prefix = weight_key.strip_suffix(".weight").unwrap_or(weight_key);
    let n_absmax = all_tensors
        .get(&format!("{}.nested_absmax", weight_key))
        .or_else(|| all_tensors.get(&format!("{}.nested_absmax", prefix)));
    let n_qmap = all_tensors
        .get(&format!("{}.nested_quant_map", weight_key))
        .or_else(|| all_tensors.get(&format!("{}.nested_quant_map", prefix)));
    let n_offset = all_tensors
        .get(&format!("{}.nested_offset", weight_key))
        .or_else(|| all_tensors.get(&format!("{}.nested_offset", prefix)))
        .or_else(|| all_tensors.get(&format!("{}.quant_state.nested_offset", weight_key)))
        .and_then(|t| t.data.first().copied())
        .unwrap_or(0.0);

    if let (Some(na), Some(nqm)) = (n_absmax, n_qmap) {
        if nqm.data.len() == 256 && !na.data.is_empty() {
            let n_blk = (absmax_tensor.data.len() + na.data.len() - 1) / na.data.len();
            let mut out = Vec::with_capacity(absmax_tensor.data.len());
            for (i, &u8_val) in absmax_tensor.data.iter().enumerate() {
                let idx = (u8_val.round().clamp(0.0, 255.0)) as usize;
                let scale = na.data.get(i / n_blk.max(1)).copied().unwrap_or(1.0);
                out.push(nqm.data[idx] * scale + n_offset);
            }
            return out;
        }
    }
    absmax_tensor.data.clone()
}

/// Transparently dequantize a quantized LoRA factor (`I8` symmetric / `LLM.int8()` `SCB`, or
/// packed `U8` 4-bit `bitsandbytes` `NF4` / `FP4` with optional double quantization) into `f64`
/// `Tensor2D` when companion scale metadata is present in `all_tensors`.
pub fn try_dequantize_lora_tensor(
    key: &str,
    tensor: &Tensor2D,
    all_tensors: &BTreeMap<String, Tensor2D>,
) -> Result<Tensor2D, String> {
    let prefix = key.strip_suffix(".weight").unwrap_or(key);

    if tensor.dtype == DType::I8 {
        // Check row-wise LLM.int8() SCB or weight_scale or blockwise absmax
        let scb = all_tensors
            .get(&format!("{}.SCB", prefix))
            .or_else(|| all_tensors.get(&format!("{}.SCB", key)))
            .or_else(|| all_tensors.get(&format!("{}.weight_scale", prefix)))
            .or_else(|| all_tensors.get(&format!("{}.weight_scale", key)))
            .or_else(|| all_tensors.get(&format!("{}.scale", prefix)));

        let mut out = tensor.data.clone();
        if let Some(scale_t) = scb {
            if scale_t.data.len() == tensor.rows && tensor.cols > 0 {
                // LLM.int8() convention: row-wise absmax stored in SCB -> val = i8_val * scb[r] / 127.0
                for r in 0..tensor.rows {
                    let s = scale_t.data[r] / 127.0;
                    let row_slice = &mut out[r * tensor.cols..(r + 1) * tensor.cols];
                    for v in row_slice {
                        *v *= s;
                    }
                }
            } else if scale_t.data.len() == 1 {
                let s = scale_t.data[0] / 127.0;
                for v in &mut out {
                    *v *= s;
                }
            }
        } else if let Some(absmax_t) = all_tensors
            .get(&format!("{}.absmax", key))
            .or_else(|| all_tensors.get(&format!("{}.absmax", prefix)))
        {
            if !absmax_t.data.is_empty() {
                let blk = (out.len() + absmax_t.data.len() - 1) / absmax_t.data.len();
                for (i, v) in out.iter_mut().enumerate() {
                    let s = absmax_t.data.get(i / blk.max(1)).copied().unwrap_or(1.0) / 127.0;
                    *v *= s;
                }
            }
        }
        return Ok(Tensor2D {
            rows: tensor.rows,
            cols: tensor.cols,
            dtype: DType::I8,
            data: out,
        });
    }

    if tensor.dtype == DType::U8 {
        let absmax_opt = all_tensors
            .get(&format!("{}.absmax", key))
            .or_else(|| all_tensors.get(&format!("{}.absmax", prefix)));
        let state_nf4 = all_tensors
            .get(&format!("{}.quant_state.bitsandbytes__nf4", key))
            .or_else(|| all_tensors.get(&format!("{}.quant_state.bitsandbytes__nf4", prefix)));
        let state_fp4 = all_tensors
            .get(&format!("{}.quant_state.bitsandbytes__fp4", key))
            .or_else(|| all_tensors.get(&format!("{}.quant_state.bitsandbytes__fp4", prefix)));

        let state_tensor = state_nf4.or(state_fp4);
        if let Some(absmax_t) = absmax_opt {
            let parsed_state = state_tensor.and_then(|st| parse_bnb_quant_state_bytes(&st.data));
            let (rows, cols, blk, default_scheme) = if let Some(st) = parsed_state {
                (st.rows, st.cols, st.blocksize, st.quant_type)
            } else if state_tensor.is_some() {
                return Err(format!(
                    "Corrupt bitsandbytes quant_state JSON metadata for tensor {}",
                    key
                ));
            } else if tensor.cols > 1 {
                // Packed 2D [rows, cols_packed] -> [rows, cols_packed * 2]
                let sch = if state_fp4.is_some() {
                    QuantScheme::Fp4
                } else {
                    QuantScheme::Nf4
                };
                let numel = tensor.rows * tensor.cols * 2;
                let inferred_blk = if !absmax_t.data.is_empty() {
                    (numel + absmax_t.data.len() - 1) / absmax_t.data.len()
                } else {
                    64
                };
                (tensor.rows, tensor.cols * 2, inferred_blk.max(1), sch)
            } else {
                return Err(format!(
                    "Packed 4-bit U8 tensor {} lacks quant_state shape metadata",
                    key
                ));
            };

            let target_numel = rows * cols;
            if tensor.data.len() * 2 < target_numel {
                return Err(format!(
                    "Packed 4-bit U8 tensor {} has {} bytes ({} nibbles), fewer than target shape ({}x{} = {})",
                    key,
                    tensor.data.len(),
                    tensor.data.len() * 2,
                    rows,
                    cols,
                    target_numel
                ));
            }

            let absmax_vals = resolve_absmax_values(key, absmax_t, all_tensors);
            if absmax_vals.is_empty() {
                return Err(format!("Empty absmax scale vector for quantized tensor {}", key));
            }

            let qmap_opt = all_tensors
                .get(&format!("{}.quant_map", key))
                .or_else(|| all_tensors.get(&format!("{}.quant_map", prefix)));
            if let Some(qm) = qmap_opt {
                if qm.data.len() != 16 {
                    return Err(format!(
                        "Invalid quant_map length {} for {} (expected 16)",
                        qm.data.len(),
                        key
                    ));
                }
            }

            let codebook: [f64; 16] = if let Some(qm) = qmap_opt {
                let mut cb = [0.0; 16];
                cb.copy_from_slice(&qm.data[..16]);
                cb
            } else {
                match default_scheme {
                    QuantScheme::Fp4 => FP4_BNB_NIBBLE_MAP,
                    _ => NF4_CODEBOOK,
                }
            };

            let mut unpacked = Vec::with_capacity(target_numel);
            for &byte_f in &tensor.data {
                let byte = byte_f.round().clamp(0.0, 255.0) as usize;
                let hi = (byte >> 4) & 0x0f;
                let lo = byte & 0x0f;
                if unpacked.len() < target_numel {
                    let s = absmax_vals
                        .get(unpacked.len() / blk.max(1))
                        .copied()
                        .unwrap_or(0.0);
                    unpacked.push(codebook[hi] * s);
                }
                if unpacked.len() < target_numel {
                    let s = absmax_vals
                        .get(unpacked.len() / blk.max(1))
                        .copied()
                        .unwrap_or(0.0);
                    unpacked.push(codebook[lo] * s);
                }
                if unpacked.len() >= target_numel {
                    break;
                }
            }

            return Ok(Tensor2D {
                rows,
                cols,
                dtype: DType::U8,
                data: unpacked,
            });
        }
    }

    Ok(tensor.clone())
}

/// Audit raw quantized auxiliary tensors (`.absmax`, `.nested_absmax`, `.SCB`, `.weight_scale`,
/// `.quant_map`, `.quant_state.bitsandbytes__*`) inside a `.safetensors` map for NaN/zero scale
/// blocks or malformed 4-bit packing metadata (`LS023`).
pub fn audit_quantized_safetensors_metadata(
    all_tensors: &BTreeMap<String, Tensor2D>,
) -> Vec<IntegrityIssue> {
    let mut issues = Vec::new();

    for (key, t) in all_tensors {
        if key.ends_with(".absmax")
            || key.ends_with(".nested_absmax")
            || key.ends_with(".SCB")
            || key.ends_with(".weight_scale")
        {
            if t.dtype == DType::U8 && key.ends_with(".absmax") {
                // Double-quantized U8 indices into nested_quant_map; validate companion nested_absmax instead
                let prefix = key.strip_suffix(".absmax").unwrap_or(key);
                let has_nested = all_tensors.contains_key(&format!("{}.nested_absmax", prefix))
                    && all_tensors.contains_key(&format!("{}.nested_quant_map", prefix));
                if !has_nested {
                    issues.push(IntegrityIssue {
                        code: "QUANT_CORRUPT_METADATA",
                        severity: IssueSeverity::Critical,
                        subject: key.clone(),
                        message: format!(
                            "Double-quantized U8 scale tensor '{}' is missing companion nested_absmax or nested_quant_map",
                            key
                        ),
                    });
                }
                continue;
            }
            let non_finite = t.data.iter().filter(|v| !v.is_finite()).count();
            let non_pos = t.data.iter().filter(|&&v| v.is_finite() && v <= 0.0).count();
            if non_finite > 0 {
                issues.push(IntegrityIssue {
                    code: "QUANT_ZERO_OR_NAN_ABSMAX",
                    severity: IssueSeverity::Critical,
                    subject: key.clone(),
                    message: format!(
                        "Quantization scale tensor '{}' contains {} NaN/Inf scale block(s) out of {}",
                        key,
                        non_finite,
                        t.data.len()
                    ),
                });
            } else if !t.data.is_empty() && non_pos == t.data.len() {
                issues.push(IntegrityIssue {
                    code: "QUANT_ZERO_OR_NAN_ABSMAX",
                    severity: IssueSeverity::Critical,
                    subject: key.clone(),
                    message: format!(
                        "Quantization scale tensor '{}' has all {} block scale(s) <= 0.0, zeroing out dequantized weights",
                        key,
                        t.data.len()
                    ),
                });
            } else if non_pos > 0 {
                issues.push(IntegrityIssue {
                    code: "QUANT_ZERO_OR_NAN_ABSMAX",
                    severity: IssueSeverity::Warning,
                    subject: key.clone(),
                    message: format!(
                        "Quantization scale tensor '{}' contains {}/{} dead or non-positive scale block(s)",
                        key,
                        non_pos,
                        t.data.len()
                    ),
                });
            }
        } else if key.ends_with(".quant_map") && t.data.len() != 16 {
            issues.push(IntegrityIssue {
                code: "QUANT_CORRUPT_METADATA",
                severity: IssueSeverity::Critical,
                subject: key.clone(),
                message: format!(
                    "4-bit codebook '{}' has {} entries instead of 16",
                    key,
                    t.data.len()
                ),
            });
        } else if key.contains(".quant_state.bitsandbytes__") {
            if parse_bnb_quant_state_bytes(&t.data).is_none() {
                issues.push(IntegrityIssue {
                    code: "QUANT_CORRUPT_METADATA",
                    severity: IssueSeverity::Critical,
                    subject: key.clone(),
                    message: format!(
                        "Malformed bitsandbytes quant_state JSON payload in '{}'",
                        key
                    ),
                });
            }
        }
    }

    issues
}

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

/// Automated merge strategy recommendation synthesized from principal subspace angles,
/// Frobenius sign conflict, effective spectral rank, and `attn` vs `mlp` localization.
#[derive(Debug, Clone)]
pub struct MergeRecommendation {
    pub recommended_mode: &'static str,
    pub suggested_rank: usize,
    pub suggested_layer_weights: Option<String>,
    pub preserve_norm: bool,
    pub conflict_localization: &'static str,
    pub rationale: String,
    pub cli_command: String,
}

/// Full adapter-pair interference report with `attn` vs `mlp` breakdown and merge advisor output.
#[derive(Debug, Clone)]
pub struct InterferenceSummary {
    pub layers: Vec<LayerInterferenceReport>,
    pub max_principal_cos: f64,
    pub mean_principal_cos: f64,
    pub mean_frobenius_cosine: f64,
    pub min_frobenius_cosine: f64,
    pub high_conflict_count: usize,
    pub moderate_overlap_count: usize,
    pub orthogonal_safe_count: usize,
    pub attn_layer_count: usize,
    pub mlp_layer_count: usize,
    pub attn_mean_conflict: f64,
    pub mlp_mean_conflict: f64,
    pub attn_mean_frobenius_cos: f64,
    pub mlp_mean_frobenius_cos: f64,
    pub recommendation: MergeRecommendation,
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

fn analyze_layer_interference_with_ranks(
    layer1: &LoraLayer,
    layer2: &LoraLayer,
) -> (LayerInterferenceReport, usize, usize) {
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

    let k99_1 = energy_rank_99(&svd1.sigma);
    let k99_2 = energy_rank_99(&svd2.sigma);

    (
        LayerInterferenceReport {
            base_name: layer1.base_name.clone(),
            max_left_principal_cos,
            max_right_principal_cos,
            grassmann_left_overlap,
            grassmann_right_overlap,
            frobenius_cosine,
            conflict_severity,
        },
        k99_1,
        k99_2,
    )
}

/// Analyze principal subspace angles and Frobenius cosine similarity between two LoRA layers
/// in `O(r^2 * (d_out + d_in) + r^3)` time without forming `d_out x d_in` dense matrices.
pub fn analyze_layer_interference(
    layer1: &LoraLayer,
    layer2: &LoraLayer,
) -> LayerInterferenceReport {
    analyze_layer_interference_with_ranks(layer1, layer2).0
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LayerBlockClass {
    Attn,
    Mlp,
    Other,
}

fn classify_layer_block(base_name: &str) -> LayerBlockClass {
    let lower = base_name.to_ascii_lowercase();
    if lower.contains("mlp")
        || lower.contains("ffn")
        || lower.contains("feed_forward")
        || lower.ends_with("gate_proj")
        || lower.ends_with("up_proj")
        || lower.ends_with("down_proj")
        || lower.contains(".ff.")
        || lower.contains("_ff_")
        || lower.ends_with(".linear1")
        || lower.ends_with(".linear2")
        || lower.ends_with("_linear1")
        || lower.ends_with("_linear2")
        || lower.ends_with(".fc1")
        || lower.ends_with(".fc2")
        || lower.ends_with("_fc1")
        || lower.ends_with("_fc2")
    {
        LayerBlockClass::Mlp
    } else if lower.contains("attn")
        || lower.contains("attention")
        || lower.ends_with("q_proj")
        || lower.ends_with("k_proj")
        || lower.ends_with("v_proj")
        || lower.ends_with("o_proj")
        || lower.contains(".to_q")
        || lower.contains(".to_k")
        || lower.contains(".to_v")
        || lower.contains(".to_out")
        || lower.contains("_to_q")
        || lower.contains("_to_k")
        || lower.contains("_to_v")
        || lower.contains("_to_out")
    {
        LayerBlockClass::Attn
    } else {
        LayerBlockClass::Other
    }
}

fn layer_conflict_score(rep: &LayerInterferenceReport) -> f64 {
    let geom_cos = (rep.max_left_principal_cos * rep.max_right_principal_cos).sqrt();
    let mean_cos = 0.5 * (rep.max_left_principal_cos + rep.max_right_principal_cos);
    let sign_penalty = (-rep.frobenius_cosine).max(0.0);
    (0.5 * geom_cos + 0.5 * mean_cos + sign_penalty).clamp(0.0, 1.5)
}

fn snap_suggested_rank(raw_rank: usize, max_cap: usize) -> usize {
    let capped = raw_rank.clamp(1, max_cap.max(1));
    let buckets = [4usize, 8, 12, 16, 24, 32, 48, 64, 96, 128];
    for &b in &buckets {
        if b >= capped {
            return b.min(max_cap.max(1));
        }
    }
    capped
}

fn shell_quote_arg(s: &str) -> String {
    if s.chars().any(|c| c.is_whitespace() || c == '"' || c == '\'') {
        format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\""))
    } else {
        s.to_string()
    }
}

fn normalize_match_key(name: &str) -> &str {
    name.strip_prefix("base_model.model.").unwrap_or(name)
}

/// Analyze interference across all matched layers of two LoRA adapters and generate an
/// automated merge recommendation (`--mode`, `--rank`, `--layer-weights`, `--preserve-norm`).
pub fn analyze_adapter_interference(
    layers_a: &[LoraLayer],
    layers_b: &[LoraLayer],
    label_a: &str,
    label_b: &str,
) -> Result<InterferenceSummary, String> {
    let mut map_b_exact: BTreeMap<&str, &LoraLayer> = BTreeMap::new();
    let mut map_b_norm: BTreeMap<&str, &LoraLayer> = BTreeMap::new();
    for lb in layers_b {
        map_b_exact.insert(lb.base_name.as_str(), lb);
        map_b_norm.insert(normalize_match_key(&lb.base_name), lb);
    }

    let mut reports = Vec::new();
    let mut combined_energy_ranks = Vec::new();
    let mut max_rank_sum = 1usize;
    let mut min_dim_cap = usize::MAX;

    for la in layers_a {
        let matched = map_b_exact
            .get(la.base_name.as_str())
            .or_else(|| map_b_norm.get(normalize_match_key(&la.base_name)));
        if let Some(&lb) = matched {
            if la.d_out() != lb.d_out() || la.d_in() != lb.d_in() {
                return Err(format!(
                    "Incompatible layer dimensions for {}: ({}, {}) vs ({}, {})",
                    la.base_name,
                    la.d_out(),
                    la.d_in(),
                    lb.d_out(),
                    lb.d_in()
                ));
            }
            let (rep, k99_a, k99_b) = analyze_layer_interference_with_ranks(la, lb);
            let grassmann_avg = 0.5 * (rep.grassmann_left_overlap + rep.grassmann_right_overlap);
            let needed_k =
                k99_a + ((1.0 - 0.5 * grassmann_avg) * (k99_b as f64)).ceil() as usize;
            combined_energy_ranks.push(needed_k.max(1));
            max_rank_sum = max_rank_sum.max(la.rank() + lb.rank());
            min_dim_cap = min_dim_cap.min(la.d_out().min(la.d_in()));
            reports.push(rep);
        }
    }

    if reports.is_empty() {
        return Err("No matching LoRA layer pairs found between the two adapters".to_string());
    }

    let n = reports.len() as f64;
    let mut max_principal_cos = 0.0_f64;
    let mut sum_principal_cos = 0.0_f64;
    let mut sum_frob_cos = 0.0_f64;
    let mut min_frobenius_cosine = 1.0_f64;
    let mut high_conflict_count = 0usize;
    let mut moderate_overlap_count = 0usize;
    let mut orthogonal_safe_count = 0usize;

    let mut attn_conflicts = Vec::new();
    let mut attn_frobs = Vec::new();
    let mut attn_high = 0usize;
    let mut attn_names = Vec::new();

    let mut mlp_conflicts = Vec::new();
    let mut mlp_frobs = Vec::new();
    let mut mlp_high = 0usize;
    let mut mlp_names = Vec::new();

    for r in &reports {
        let pair_max = r.max_left_principal_cos.max(r.max_right_principal_cos);
        let pair_mean = 0.5 * (r.max_left_principal_cos + r.max_right_principal_cos);
        max_principal_cos = max_principal_cos.max(pair_max);
        sum_principal_cos += pair_mean;
        sum_frob_cos += r.frobenius_cosine;
        min_frobenius_cosine = min_frobenius_cosine.min(r.frobenius_cosine);

        match r.conflict_severity {
            "HIGH_INTERFERENCE" => high_conflict_count += 1,
            "MODERATE_OVERLAP" => moderate_overlap_count += 1,
            _ => orthogonal_safe_count += 1,
        }

        let c_score = layer_conflict_score(r);
        match classify_layer_block(&r.base_name) {
            LayerBlockClass::Attn => {
                attn_conflicts.push(c_score);
                attn_frobs.push(r.frobenius_cosine);
                if r.conflict_severity == "HIGH_INTERFERENCE" {
                    attn_high += 1;
                }
                attn_names.push(r.base_name.as_str());
            }
            LayerBlockClass::Mlp => {
                mlp_conflicts.push(c_score);
                mlp_frobs.push(r.frobenius_cosine);
                if r.conflict_severity == "HIGH_INTERFERENCE" {
                    mlp_high += 1;
                }
                mlp_names.push(r.base_name.as_str());
            }
            LayerBlockClass::Other => {}
        }
    }

    let mean_principal_cos = sum_principal_cos / n;
    let mean_frobenius_cosine = sum_frob_cos / n;

    let attn_layer_count = attn_conflicts.len();
    let mlp_layer_count = mlp_conflicts.len();
    let attn_mean_conflict = if attn_layer_count > 0 {
        attn_conflicts.iter().sum::<f64>() / (attn_layer_count as f64)
    } else {
        0.0
    };
    let mlp_mean_conflict = if mlp_layer_count > 0 {
        mlp_conflicts.iter().sum::<f64>() / (mlp_layer_count as f64)
    } else {
        0.0
    };
    let attn_mean_frobenius_cos = if attn_layer_count > 0 {
        attn_frobs.iter().sum::<f64>() / (attn_layer_count as f64)
    } else {
        0.0
    };
    let mlp_mean_frobenius_cos = if mlp_layer_count > 0 {
        mlp_frobs.iter().sum::<f64>() / (mlp_layer_count as f64)
    } else {
        0.0
    };

    // 1. Select merge mode
    let recommended_mode = if high_conflict_count > 0
        || min_frobenius_cosine < -0.12
        || max_principal_cos > 0.70
    {
        "ortho-svd"
    } else if moderate_overlap_count > 0
        && (mean_principal_cos >= 0.28
            || (moderate_overlap_count as f64) / n >= 0.25
            || min_frobenius_cosine < -0.03)
    {
        "ties"
    } else {
        "linear"
    };

    // 2. Estimate suggested target rank from 90th-percentile union 99%-energy rank
    combined_energy_ranks.sort_unstable();
    let idx_90 = (((combined_energy_ranks.len() as f64) * 0.90).ceil() as usize)
        .saturating_sub(1)
        .min(combined_energy_ranks.len() - 1);
    let raw_target_rank = combined_energy_ranks[idx_90];
    let rank_cap = max_rank_sum.min(min_dim_cap).min(128);
    let suggested_rank = snap_suggested_rank(raw_target_rank, rank_cap);

    // 3. Detect attn vs mlp localized conflict for --layer-weights
    let attn_pat = if !attn_names.is_empty() && attn_names.iter().all(|s| s.contains("attn")) {
        "attn"
    } else {
        "proj"
    };
    let mlp_pat = if !mlp_names.is_empty() && mlp_names.iter().all(|s| s.contains("mlp")) {
        "mlp"
    } else if !mlp_names.is_empty() && mlp_names.iter().all(|s| s.contains("ffn")) {
        "ffn"
    } else if !mlp_names.is_empty() && mlp_names.iter().all(|s| s.contains("ff")) {
        "ff"
    } else {
        "linear"
    };

    let (conflict_localization, suggested_layer_weights) =
        if attn_layer_count > 0 && mlp_layer_count > 0 {
            let diff = attn_mean_conflict - mlp_mean_conflict;
            if (diff >= 0.20 && (attn_mean_conflict >= 0.38 || attn_high > 0))
                || (attn_high > 0 && mlp_high == 0 && diff >= 0.12)
            {
                let lw = if recommended_mode == "ortho-svd" {
                    format!("{}=1.0:0.35,{}=1.0:1.0", attn_pat, mlp_pat)
                } else {
                    format!("{}=0.8:0.2,{}=0.5:0.5", attn_pat, mlp_pat)
                };
                ("LOCALIZED_ATTN", Some(lw))
            } else if (-diff >= 0.20 && (mlp_mean_conflict >= 0.38 || mlp_high > 0))
                || (mlp_high > 0 && attn_high == 0 && -diff >= 0.12)
            {
                let lw = if recommended_mode == "ortho-svd" {
                    format!("{}=1.0:1.0,{}=1.0:0.35", attn_pat, mlp_pat)
                } else {
                    format!("{}=0.5:0.5,{}=0.8:0.2", attn_pat, mlp_pat)
                };
                ("LOCALIZED_MLP", Some(lw))
            } else if high_conflict_count > 0 || moderate_overlap_count > 0 {
                ("GLOBAL", None)
            } else {
                ("NONE", None)
            }
        } else if high_conflict_count > 0 || moderate_overlap_count > 0 {
            ("GLOBAL", None)
        } else {
            ("NONE", None)
        };

    let preserve_norm = recommended_mode == "ortho-svd"
        && (max_principal_cos > 0.60 || mean_frobenius_cosine > 0.20);

    let mode_reason = match recommended_mode {
        "ortho-svd" => format!(
            "Detected {} high-interference layer(s) (max principal cos {:.3}, min Frobenius cos {:.3}); orthogonal subspace projection prevents destructive sign cancellation",
            high_conflict_count, max_principal_cos, min_frobenius_cosine
        ),
        "ties" => format!(
            "Detected {} moderate-overlap layer(s) (mean principal cos {:.3}) without severe sign opposition; TIES magnitude pruning trims diffuse parameter drift before QR-Core SVD",
            moderate_overlap_count, mean_principal_cos
        ),
        _ => format!(
            "Principal subspaces are nearly orthogonal (max cos {:.3}, mean cos {:.3}); linear low-rank concatenation + QR-Core SVD retains full fidelity",
            max_principal_cos, mean_principal_cos
        ),
    };

    let loc_reason = match conflict_localization {
        "LOCALIZED_ATTN" => format!(
            " Conflict is concentrated in attention blocks (attn {:.2} vs mlp {:.2}); attenuating adapter B in '{}' layers is recommended.",
            attn_mean_conflict, mlp_mean_conflict, attn_pat
        ),
        "LOCALIZED_MLP" => format!(
            " Conflict is concentrated in MLP blocks (mlp {:.2} vs attn {:.2}); attenuating adapter B in '{}' layers is recommended.",
            mlp_mean_conflict, attn_mean_conflict, mlp_pat
        ),
        _ => String::new(),
    };
    let rationale = format!("{}.{}", mode_reason, loc_reason);

    let mut cli_command = format!(
        "loraspectra merge {} {} merged.safetensors --mode {} --rank {}",
        shell_quote_arg(label_a),
        shell_quote_arg(label_b),
        recommended_mode,
        suggested_rank
    );
    if let Some(ref lw) = suggested_layer_weights {
        cli_command.push_str(&format!(" --layer-weights \"{}\"", lw));
    }
    if preserve_norm {
        cli_command.push_str(" --preserve-norm");
    }

    Ok(InterferenceSummary {
        layers: reports,
        max_principal_cos,
        mean_principal_cos,
        mean_frobenius_cosine,
        min_frobenius_cosine,
        high_conflict_count,
        moderate_overlap_count,
        orthogonal_safe_count,
        attn_layer_count,
        mlp_layer_count,
        attn_mean_conflict,
        mlp_mean_conflict,
        attn_mean_frobenius_cos,
        mlp_mean_frobenius_cos,
        recommendation: MergeRecommendation {
            recommended_mode,
            suggested_rank,
            suggested_layer_weights,
            preserve_norm,
            conflict_localization,
            rationale,
            cli_command,
        },
    })
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

use std::collections::BTreeMap;

use crate::audit::pearson_correlation;
use crate::linalg::{dot_slice, jacobi_svd, matmul, thin_qr_columns, transpose};
use crate::safetensors::{
    resolve_layer_scaling_params, subtract_pissa_residual_layer, DType, LoraLayer,
    PeftAdapterConfig, Tensor2D,
};

#[derive(Debug, Clone)]
pub struct LayerDiffReport {
    pub base_name: String,
    pub rank_a: usize,
    pub rank_b: usize,
    pub scale_a: f64,
    pub scale_b: f64,
    pub d_out: usize,
    pub d_in: usize,
    pub frobenius_norm_a: f64,
    pub frobenius_norm_b: f64,
    pub frobenius_delta: f64,
    pub relative_drift: f64,
    pub subspace_similarity_in: f64,
    pub subspace_similarity_out: f64,
    pub subspace_similarity: f64,
    pub rank_delta: i64,
    pub pissa_residual_subtracted: bool,
    pub dora_l2_drift: Option<f64>,
    pub dora_relative_drift: Option<f64>,
    pub dora_max_channel_delta: Option<f64>,
    pub dora_mag_dir_coupling: Option<f64>,
}

#[derive(Debug, Clone)]
pub struct AdapterDiffSummary {
    pub shared_layers: Vec<LayerDiffReport>,
    pub added_layers: Vec<String>,
    pub removed_layers: Vec<String>,
    pub mean_relative_drift: f64,
    pub mean_subspace_similarity: f64,
    pub max_drift_layer: Option<String>,
    pub max_relative_drift: f64,
    pub use_rslora_a: bool,
    pub use_rslora_b: bool,
    pub pissa_layers_normalized: usize,
    pub dora_shared_layers: usize,
    pub mean_dora_relative_drift: Option<f64>,
}

fn compute_layer_scaling(base_name: &str, rank: usize, alphas: &BTreeMap<String, f64>) -> f64 {
    let r = rank.max(1) as f64;
    if let Some(&alpha) = alphas.get(base_name) {
        alpha / r
    } else {
        1.0
    }
}

/// Compute layer-level difference between two LoRA adapter representations.
/// Calculates exact Frobenius norm drift and Grassmanian subspace overlap without
/// dense matrix allocations: O(r^2 * (d_in + d_out)).
pub fn diff_layer(
    layer_a: &LoraLayer,
    scale_a: f64,
    layer_b: &LoraLayer,
    scale_b: f64,
) -> Option<LayerDiffReport> {
    diff_layer_with_dora(layer_a, scale_a, None, layer_b, scale_b, None)
}

/// Compute layer-level directional LoRA drift plus optional DoRA magnitude drift and
/// per-channel magnitude-vs-directional coupling in `O(r^2 * (d_in + d_out))`.
pub fn diff_layer_with_dora(
    layer_a: &LoraLayer,
    scale_a: f64,
    dora_a: Option<&Tensor2D>,
    layer_b: &LoraLayer,
    scale_b: f64,
    dora_b: Option<&Tensor2D>,
) -> Option<LayerDiffReport> {
    if layer_a.d_out() != layer_b.d_out() || layer_a.d_in() != layer_b.d_in() {
        return None;
    }

    let d_out = layer_a.d_out();
    let d_in = layer_a.d_in();
    let r_a = layer_a.rank();
    let r_b = layer_b.rank();

    // 1. Gram matrix for A: M_A = A_A * A_A^T (r_A x r_A), N_A = B_A^T * B_A (r_A x r_A)
    let mut ma = vec![0.0; r_a * r_a];
    for i in 0..r_a {
        let row_i = &layer_a.a.data[i * d_in..(i + 1) * d_in];
        for j in i..r_a {
            let row_j = &layer_a.a.data[j * d_in..(j + 1) * d_in];
            let dot = dot_slice(row_i, row_j);
            ma[i * r_a + j] = dot;
            ma[j * r_a + i] = dot;
        }
    }

    let mut na = vec![0.0; r_a * r_a];
    for i in 0..r_a {
        for j in i..r_a {
            let mut dot = 0.0;
            for k in 0..d_out {
                dot += layer_a.b.data[k * r_a + i] * layer_a.b.data[k * r_a + j];
            }
            na[i * r_a + j] = dot;
            na[j * r_a + i] = dot;
        }
    }

    let mut fn_sq_a = 0.0;
    for i in 0..r_a {
        for j in 0..r_a {
            fn_sq_a += na[i * r_a + j] * ma[j * r_a + i];
        }
    }
    let norm_a = (scale_a * scale_a * fn_sq_a.max(0.0)).sqrt();

    // 2. Gram matrix for B: M_B = A_B * A_B^T (r_B x r_B), N_B = B_B^T * B_B (r_B x r_B)
    let mut mb = vec![0.0; r_b * r_b];
    for i in 0..r_b {
        let row_i = &layer_b.a.data[i * d_in..(i + 1) * d_in];
        for j in i..r_b {
            let row_j = &layer_b.a.data[j * d_in..(j + 1) * d_in];
            let dot = dot_slice(row_i, row_j);
            mb[i * r_b + j] = dot;
            mb[j * r_b + i] = dot;
        }
    }

    let mut nb = vec![0.0; r_b * r_b];
    for i in 0..r_b {
        for j in i..r_b {
            let mut dot = 0.0;
            for k in 0..d_out {
                dot += layer_b.b.data[k * r_b + i] * layer_b.b.data[k * r_b + j];
            }
            nb[i * r_b + j] = dot;
            nb[j * r_b + i] = dot;
        }
    }

    let mut fn_sq_b = 0.0;
    for i in 0..r_b {
        for j in 0..r_b {
            fn_sq_b += nb[i * r_b + j] * mb[j * r_b + i];
        }
    }
    let norm_b = (scale_b * scale_b * fn_sq_b.max(0.0)).sqrt();

    // 3. Cross term: P = A_A * A_B^T (r_A x r_B) (also reused for DoRA row deltas)
    let mut p = vec![0.0; r_a * r_b];
    for i in 0..r_a {
        let row_a = &layer_a.a.data[i * d_in..(i + 1) * d_in];
        for j in 0..r_b {
            let row_b = &layer_b.a.data[j * d_in..(j + 1) * d_in];
            p[i * r_b + j] = dot_slice(row_a, row_b);
        }
    }

    // Cancellation-free QR-Core Frobenius drift:
    // s_B * B_B * A_B - s_A * B_A * A_A = [s_B * B_B, -s_A * B_A] * [A_B; A_A].
    // Evaluating ||R_B * R_A^T||_F avoids the sqrt(eps) cancellation floor of
    // sqrt(||X||_F^2 + ||Y||_F^2 - 2<X, Y>_F) when two checkpoints are nearly identical.
    let r_sum = r_a + r_b;
    let frobenius_delta = if r_sum == 0 || d_out == 0 || d_in == 0 {
        0.0
    } else {
        let mut c_b = Tensor2D::zeros(d_out, r_sum, DType::F32);
        for row in 0..d_out {
            let bb_row = &layer_b.b.data[row * r_b..(row + 1) * r_b];
            let ba_row = &layer_a.b.data[row * r_a..(row + 1) * r_a];
            let dst = &mut c_b.data[row * r_sum..(row + 1) * r_sum];
            for c in 0..r_b {
                dst[c] = scale_b * bb_row[c];
            }
            for c in 0..r_a {
                dst[r_b + c] = -scale_a * ba_row[c];
            }
        }
        let mut c_at = Tensor2D::zeros(d_in, r_sum, DType::F32);
        for col in 0..d_in {
            let dst = &mut c_at.data[col * r_sum..(col + 1) * r_sum];
            for c in 0..r_b {
                dst[c] = layer_b.a.data[c * d_in + col];
            }
            for c in 0..r_a {
                dst[r_b + c] = layer_a.a.data[c * d_in + col];
            }
        }
        if r_sum <= d_out.min(d_in) {
            let (_qb, rb) = thin_qr_columns(&c_b);
            let (_qa, ra) = thin_qr_columns(&c_at);
            let core = matmul(&rb, &transpose(&ra));
            core.frobenius_norm_sq().sqrt()
        } else {
            let dense_diff = matmul(&c_b, &transpose(&c_at));
            dense_diff.frobenius_norm_sq().sqrt()
        }
    };
    let relative_drift = frobenius_delta / norm_a.max(1e-12);

    // 4. Subspace principal cosine similarity
    let min_r = r_a.min(r_b).max(1) as f64;

    // In subspace: QR of A^T
    let at_a = transpose(&layer_a.a);
    let at_b = transpose(&layer_b.a);
    let (qa_in, _) = thin_qr_columns(&at_a);
    let (qb_in, _) = thin_qr_columns(&at_b);

    let mut sim_in = 0.0;
    for c_a in &qa_in {
        for c_b in &qb_in {
            let dot = dot_slice(c_a, c_b);
            sim_in += dot * dot;
        }
    }
    let subspace_sim_in = (sim_in / min_r).clamp(0.0, 1.0);

    // Out subspace: QR of B
    let (qa_out, _) = thin_qr_columns(&layer_a.b);
    let (qb_out, _) = thin_qr_columns(&layer_b.b);

    let mut sim_out = 0.0;
    for c_a in &qa_out {
        for c_b in &qb_out {
            let dot = dot_slice(c_a, c_b);
            sim_out += dot * dot;
        }
    }
    let subspace_sim_out = (sim_out / min_r).clamp(0.0, 1.0);

    let subspace_similarity = (subspace_sim_in * subspace_sim_out).sqrt().clamp(0.0, 1.0);

    // 5. Optional DoRA magnitude drift & per-channel directional coupling
    let (dora_l2_drift, dora_relative_drift, dora_max_channel_delta, dora_mag_dir_coupling) =
        match (dora_a, dora_b) {
            (Some(ta), Some(tb)) if ta.data.len() == d_out && tb.data.len() == d_out && d_out > 0 => {
                let mut mag_a_sq = 0.0;
                let mut mag_diff_sq = 0.0;
                let mut max_delta = 0.0_f64;
                let mut abs_mag_deltas = Vec::with_capacity(d_out);
                let mut dir_row_deltas = Vec::with_capacity(d_out);

                for k in 0..d_out {
                    let ma_k = ta.data[k];
                    let mb_k = tb.data[k];
                    let dm = (mb_k - ma_k).abs();
                    mag_a_sq += ma_k * ma_k;
                    mag_diff_sq += dm * dm;
                    max_delta = max_delta.max(dm);
                    abs_mag_deltas.push(dm);

                    // Exact ||(s_B B_B A_B - s_A B_A A_A)_{k,:}||_2 via Gram matrices ma, mb, p
                    let ba_row = &layer_a.b.data[k * r_a..(k + 1) * r_a];
                    let bb_row = &layer_b.b.data[k * r_b..(k + 1) * r_b];
                    let mut ua_sq = 0.0;
                    for i in 0..r_a {
                        ua_sq += ba_row[i] * dot_slice(&ma[i * r_a..(i + 1) * r_a], ba_row);
                    }
                    let mut vb_sq = 0.0;
                    for i in 0..r_b {
                        vb_sq += bb_row[i] * dot_slice(&mb[i * r_b..(i + 1) * r_b], bb_row);
                    }
                    let mut cross_row = 0.0;
                    for i in 0..r_a {
                        cross_row += ba_row[i] * dot_slice(&p[i * r_b..(i + 1) * r_b], bb_row);
                    }
                    let row_diff_sq = (scale_a * scale_a * ua_sq
                        + scale_b * scale_b * vb_sq
                        - 2.0 * scale_a * scale_b * cross_row)
                        .max(0.0);
                    dir_row_deltas.push(row_diff_sq.sqrt());
                }

                let l2_d = mag_diff_sq.sqrt();
                let rel_d = l2_d / mag_a_sq.sqrt().max(1e-12);
                let coupling = pearson_correlation(&abs_mag_deltas, &dir_row_deltas);
                (Some(l2_d), Some(rel_d), Some(max_delta), Some(coupling))
            }
            _ => (None, None, None, None),
        };

    Some(LayerDiffReport {
        base_name: layer_a.base_name.clone(),
        rank_a: r_a,
        rank_b: r_b,
        scale_a,
        scale_b,
        d_out,
        d_in,
        frobenius_norm_a: norm_a,
        frobenius_norm_b: norm_b,
        frobenius_delta,
        relative_drift,
        subspace_similarity_in: subspace_sim_in,
        subspace_similarity_out: subspace_sim_out,
        subspace_similarity,
        rank_delta: (r_b as i64) - (r_a as i64),
        pissa_residual_subtracted: false,
        dora_l2_drift,
        dora_relative_drift,
        dora_max_channel_delta,
        dora_mag_dir_coupling,
    })
}

/// Diff two LoRA adapters across all layers.
pub fn diff_adapters(
    layers_a: &[LoraLayer],
    alphas_a: &BTreeMap<String, f64>,
    layers_b: &[LoraLayer],
    alphas_b: &BTreeMap<String, f64>,
) -> AdapterDiffSummary {
    let empty_dora = BTreeMap::new();
    diff_adapters_with_dora(
        layers_a,
        alphas_a,
        &empty_dora,
        layers_b,
        alphas_b,
        &empty_dora,
    )
}

/// Diff two LoRA/DoRA adapters across all layers, decomposing directional `B*A` Frobenius/subspace
/// drift from DoRA `lora_magnitude_vector` channel rescaling drift.
pub fn diff_adapters_with_dora(
    layers_a: &[LoraLayer],
    alphas_a: &BTreeMap<String, f64>,
    dora_a: &BTreeMap<String, (String, Tensor2D)>,
    layers_b: &[LoraLayer],
    alphas_b: &BTreeMap<String, f64>,
    dora_b: &BTreeMap<String, (String, Tensor2D)>,
) -> AdapterDiffSummary {
    diff_adapters_full(
        layers_a, alphas_a, dora_a, None, None, layers_b, alphas_b, dora_b, None, None,
    )
}

/// Full LoRA / rsLoRA / DoRA / PiSSA adapter comparison resolving per-layer `PeftAdapterConfig`
/// (`lora_alpha`, `alpha_pattern`, `use_rslora` `alpha / sqrt(r)`) and normalizing net base-model
/// task perturbations `\Delta W_net = scale * B * A - B_init * A_init` when `pissa_init` layers exist.
#[allow(clippy::too_many_arguments)]
pub fn diff_adapters_full(
    layers_a: &[LoraLayer],
    alphas_a: &BTreeMap<String, f64>,
    dora_a: &BTreeMap<String, (String, Tensor2D)>,
    config_a: Option<&PeftAdapterConfig>,
    pissa_a: Option<&BTreeMap<String, LoraLayer>>,
    layers_b: &[LoraLayer],
    alphas_b: &BTreeMap<String, f64>,
    dora_b: &BTreeMap<String, (String, Tensor2D)>,
    config_b: Option<&PeftAdapterConfig>,
    pissa_b: Option<&BTreeMap<String, LoraLayer>>,
) -> AdapterDiffSummary {
    let mut map_a: BTreeMap<&str, &LoraLayer> = BTreeMap::new();
    for l in layers_a {
        map_a.insert(l.base_name.as_str(), l);
    }

    let mut map_b: BTreeMap<&str, &LoraLayer> = BTreeMap::new();
    for l in layers_b {
        map_b.insert(l.base_name.as_str(), l);
    }

    let use_rslora_a = config_a.map_or(false, |c| c.use_rslora);
    let use_rslora_b = config_b.map_or(false, |c| c.use_rslora);
    let mut pissa_layers_normalized = 0usize;

    let mut shared_reports: Vec<LayerDiffReport> = Vec::new();
    let mut removed_layers: Vec<String> = Vec::new();
    let mut added_layers: Vec<String> = Vec::new();

    for (name, layer_a) in &map_a {
        if let Some(layer_b) = map_b.get(name) {
            let (_, scale_a, _) = if config_a.is_some() {
                resolve_layer_scaling_params(layer_a, alphas_a, None, config_a, false)
            } else {
                (0.0, compute_layer_scaling(name, layer_a.rank(), alphas_a), false)
            };
            let (_, scale_b, _) = if config_b.is_some() {
                resolve_layer_scaling_params(layer_b, alphas_b, None, config_b, false)
            } else {
                (0.0, compute_layer_scaling(name, layer_b.rank(), alphas_b), false)
            };

            let init_a = pissa_a.and_then(|m| m.get(*name));
            let init_b = pissa_b.and_then(|m| m.get(*name));

            let mut pissa_sub = false;
            let mut fallback_denom: Option<f64> = None;

            let (eff_layer_a, eff_scale_a) = if let Some(ia) = init_a {
                if let Some(net_a) = subtract_pissa_residual_layer(layer_a, scale_a, ia, 1.0) {
                    pissa_sub = true;
                    if let Some(ia_rep) = diff_layer(ia, 1.0, ia, 0.0) {
                        fallback_denom = Some(ia_rep.frobenius_norm_a);
                    }
                    (Some(net_a), 1.0)
                } else {
                    (None, scale_a)
                }
            } else {
                (None, scale_a)
            };

            let (eff_layer_b, eff_scale_b) = if let Some(ib) = init_b {
                if let Some(net_b) = subtract_pissa_residual_layer(layer_b, scale_b, ib, 1.0) {
                    pissa_sub = true;
                    (Some(net_b), 1.0)
                } else {
                    (None, scale_b)
                }
            } else {
                (None, scale_b)
            };

            let ref_a = eff_layer_a.as_ref().unwrap_or(layer_a);
            let ref_b = eff_layer_b.as_ref().unwrap_or(layer_b);
            let da = dora_a.get(*name).map(|(_, t)| t);
            let db = dora_b.get(*name).map(|(_, t)| t);

            if let Some(mut report) =
                diff_layer_with_dora(ref_a, eff_scale_a, da, ref_b, eff_scale_b, db)
            {
                // Preserve nominal checkpoint ranks and resolved runtime scales on the report
                report.rank_a = layer_a.rank();
                report.rank_b = layer_b.rank();
                report.rank_delta = (layer_b.rank() as i64) - (layer_a.rank() as i64);
                report.scale_a = scale_a;
                report.scale_b = scale_b;
                report.pissa_residual_subtracted = pissa_sub;
                if pissa_sub {
                    pissa_layers_normalized += 1;
                    if report.frobenius_norm_a <= 1e-9 {
                        if let Some(fb) = fallback_denom.filter(|&v| v > 1e-9) {
                            report.relative_drift = report.frobenius_delta / fb;
                        }
                    }
                }
                shared_reports.push(report);
            }
        } else {
            removed_layers.push((*name).to_string());
        }
    }

    for name in map_b.keys() {
        if !map_a.contains_key(name) {
            added_layers.push((*name).to_string());
        }
    }

    // Sort shared reports descending by relative drift
    shared_reports.sort_by(|a, b| {
        b.relative_drift
            .partial_cmp(&a.relative_drift)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let n = shared_reports.len().max(1) as f64;
    let mean_relative_drift = shared_reports.iter().map(|r| r.relative_drift).sum::<f64>() / n;
    let mean_subspace_similarity =
        shared_reports.iter().map(|r| r.subspace_similarity).sum::<f64>() / n;

    let (max_drift_layer, max_relative_drift) = if let Some(top) = shared_reports.first() {
        (Some(top.base_name.clone()), top.relative_drift)
    } else {
        (None, 0.0)
    };

    let dora_rel_vals: Vec<f64> = shared_reports
        .iter()
        .filter_map(|r| r.dora_relative_drift)
        .collect();
    let dora_shared_layers = dora_rel_vals.len();
    let mean_dora_relative_drift = if !dora_rel_vals.is_empty() {
        Some(dora_rel_vals.iter().sum::<f64>() / (dora_rel_vals.len() as f64))
    } else {
        None
    };

    AdapterDiffSummary {
        shared_layers: shared_reports,
        added_layers,
        removed_layers,
        mean_relative_drift,
        mean_subspace_similarity,
        max_drift_layer,
        max_relative_drift,
        use_rslora_a,
        use_rslora_b,
        pissa_layers_normalized,
        dora_shared_layers,
        mean_dora_relative_drift,
    }
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

fn opt_json_f64(val: Option<f64>) -> String {
    match val {
        Some(v) if v.is_finite() => format!("{:.6}", v),
        _ => "null".to_string(),
    }
}

pub fn render_diff_json(summary: &AdapterDiffSummary) -> String {
    let layers_json: Vec<String> = summary
        .shared_layers
        .iter()
        .map(|r| {
            format!(
                r#"{{
      "layer": "{}",
      "rank_a": {},
      "rank_b": {},
      "scale_a": {:.6},
      "scale_b": {:.6},
      "rank_delta": {},
      "d_out": {},
      "d_in": {},
      "frobenius_norm_a": {:.6},
      "frobenius_norm_b": {:.6},
      "frobenius_delta": {:.6},
      "relative_drift": {:.6},
      "subspace_similarity_in": {:.6},
      "subspace_similarity_out": {:.6},
      "subspace_similarity": {:.6},
      "pissa_residual_subtracted": {},
      "dora_l2_drift": {},
      "dora_relative_drift": {},
      "dora_max_channel_delta": {},
      "dora_mag_dir_coupling": {}
    }}"#,
                escape_json(&r.base_name),
                r.rank_a,
                r.rank_b,
                r.scale_a,
                r.scale_b,
                r.rank_delta,
                r.d_out,
                r.d_in,
                r.frobenius_norm_a,
                r.frobenius_norm_b,
                r.frobenius_delta,
                r.relative_drift,
                r.subspace_similarity_in,
                r.subspace_similarity_out,
                r.subspace_similarity,
                r.pissa_residual_subtracted,
                opt_json_f64(r.dora_l2_drift),
                opt_json_f64(r.dora_relative_drift),
                opt_json_f64(r.dora_max_channel_delta),
                opt_json_f64(r.dora_mag_dir_coupling),
            )
        })
        .collect();

    let added_json: Vec<String> = summary
        .added_layers
        .iter()
        .map(|s| format!("\"{}\"", escape_json(s)))
        .collect();

    let removed_json: Vec<String> = summary
        .removed_layers
        .iter()
        .map(|s| format!("\"{}\"", escape_json(s)))
        .collect();

    format!(
        r#"{{
  "total_shared_layers": {},
  "total_added_layers": {},
  "total_removed_layers": {},
  "mean_relative_drift": {:.6},
  "mean_subspace_similarity": {:.6},
  "max_drift_layer": {},
  "max_relative_drift": {:.6},
  "use_rslora_a": {},
  "use_rslora_b": {},
  "pissa_layers_normalized": {},
  "dora_shared_layers": {},
  "mean_dora_relative_drift": {},
  "layers": [
    {}
  ],
  "added_layers": [{}],
  "removed_layers": [{}]
}}
"#,
        summary.shared_layers.len(),
        summary.added_layers.len(),
        summary.removed_layers.len(),
        summary.mean_relative_drift,
        summary.mean_subspace_similarity,
        summary
            .max_drift_layer
            .as_ref()
            .map_or("null".to_string(), |s| format!("\"{}\"", escape_json(s))),
        summary.max_relative_drift,
        summary.use_rslora_a,
        summary.use_rslora_b,
        summary.pissa_layers_normalized,
        summary.dora_shared_layers,
        opt_json_f64(summary.mean_dora_relative_drift),
        layers_json.join(",\n    "),
        added_json.join(", "),
        removed_json.join(", ")
    )
}

pub fn render_diff_table(summary: &AdapterDiffSummary) -> String {
    let mut out = String::new();
    out.push_str("+------------------------------------------------------+-------+-------+-----------+-----------+----------+---------------+--------------+\n");
    out.push_str("| Layer                                                | r_A   | r_B   | ||W_A||_F | ||W_B||_F | RelDrift | SubspaceSim   | RankDelta    |\n");
    out.push_str("+------------------------------------------------------+-------+-------+-----------+-----------+----------+---------------+--------------+\n");

    for r in &summary.shared_layers {
        let name_trunc = if r.base_name.len() > 52 {
            let start = r.base_name.len() - 49;
            format!("...{}", &r.base_name[start..])
        } else {
            r.base_name.clone()
        };

        let rank_d_str = if r.rank_delta > 0 {
            format!("+{}", r.rank_delta)
        } else if r.rank_delta < 0 {
            format!("{}", r.rank_delta)
        } else {
            "=".to_string()
        };

        out.push_str(&format!(
            "| {:<52} | {:>5} | {:>5} | {:>9.4} | {:>9.4} | {:>8.4} | {:>13.4} | {:>12} |\n",
            name_trunc,
            r.rank_a,
            r.rank_b,
            r.frobenius_norm_a,
            r.frobenius_norm_b,
            r.relative_drift,
            r.subspace_similarity,
            rank_d_str
        ));
    }
    out.push_str("+------------------------------------------------------+-------+-------+-----------+-----------+----------+---------------+--------------+\n");

    out.push_str(&format!(
        "\nDifferential Summary:\n  Shared Layers: {} | Added: {} | Removed: {}\n  Mean Relative Drift: {:.4}\n  Mean Subspace Similarity: {:.4}\n",
        summary.shared_layers.len(),
        summary.added_layers.len(),
        summary.removed_layers.len(),
        summary.mean_relative_drift,
        summary.mean_subspace_similarity
    ));

    if let Some(ref max_l) = summary.max_drift_layer {
        out.push_str(&format!(
            "  Maximum Drifting Layer: {} (rel_drift: {:.4})\n",
            max_l, summary.max_relative_drift
        ));
    }
    if let Some(mean_dora) = summary.mean_dora_relative_drift {
        out.push_str(&format!(
            "  DoRA Magnitude Drift: {} shared DoRA layer(s), mean rel_drift {:.4}\n",
            summary.dora_shared_layers, mean_dora
        ));
    }
    if summary.use_rslora_a || summary.use_rslora_b {
        out.push_str(&format!(
            "  Rank-Stabilized Scaling (rsLoRA): adapter_a={}, adapter_b={}\n",
            summary.use_rslora_a, summary.use_rslora_b
        ));
    }
    if summary.pissa_layers_normalized > 0 {
        out.push_str(&format!(
            "  PiSSA Residual Normalization: {} layer(s) compared via net task delta (s*B*A - B_init*A_init)\n",
            summary.pissa_layers_normalized
        ));
    }

    if !summary.added_layers.is_empty() {
        out.push_str(&format!(
            "  Added Layers: {}\n",
            summary.added_layers.join(", ")
        ));
    }
    if !summary.removed_layers.is_empty() {
        out.push_str(&format!(
            "  Removed Layers: {}\n",
            summary.removed_layers.join(", ")
        ));
    }

    out
}

// ---- Subspace Drift Analysis ----------------------------------------------------------------

/// Classification of subspace rotation magnitude between two LoRA checkpoints.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubspaceDriftClass {
    /// Mean principal angle < 5 deg: subspace essentially unchanged (training stalled).
    Stagnant,
    /// Mean principal angle 5-30 deg: healthy gradient exploration.
    Healthy,
    /// Mean principal angle > 60 deg: catastrophic rotation, adapter overwritten.
    CatastrophicDrift,
    /// Mean principal angle 30-60 deg: heavy drift, possible destructive fine-tuning.
    HeavyDrift,
}

impl SubspaceDriftClass {
    pub fn as_str(&self) -> &'static str {
        match self {
            SubspaceDriftClass::Stagnant => "STAGNANT",
            SubspaceDriftClass::Healthy => "HEALTHY",
            SubspaceDriftClass::HeavyDrift => "HEAVY_DRIFT",
            SubspaceDriftClass::CatastrophicDrift => "CATASTROPHIC_DRIFT",
        }
    }
}

/// Per-layer subspace drift result.
#[derive(Debug, Clone)]
pub struct LayerSubspaceDrift {
    pub base_name: String,
    /// Rank of checkpoint A.
    pub rank_a: usize,
    /// Rank of checkpoint B.
    pub rank_b: usize,
    /// Number of principal angle pairs computed (= min(rank_a, rank_b)).
    pub k: usize,
    /// Principal angles between right singular spaces V_A and V_B (input subspace), radians.
    pub principal_angles_input: Vec<f64>,
    /// Mean principal angle (input), degrees.
    pub mean_angle_input_deg: f64,
    /// Principal angles between left singular spaces U_A and U_B (output subspace), radians.
    pub principal_angles_output: Vec<f64>,
    /// Mean principal angle (output), degrees.
    pub mean_angle_output_deg: f64,
    /// Average of input and output mean angles, degrees.
    pub mean_angle_deg: f64,
    /// Frobenius norm of checkpoint A's effective update (||s_A * B_A * A_A||_F).
    pub frob_a: f64,
    /// Frobenius norm of checkpoint B's effective update.
    pub frob_b: f64,
    /// Relative Frobenius drift: |frob_b - frob_a| / max(frob_a, 1e-12).
    pub relative_frob_drift: f64,
    /// Drift classification.
    pub classification: SubspaceDriftClass,
    /// SARIF rules triggered.
    pub issues: Vec<&'static str>,
}

/// Summary of subspace drift across all shared layers between two checkpoints.
#[derive(Debug, Clone)]
pub struct AdapterSubspaceDriftSummary {
    pub total_shared_layers: usize,
    pub stagnant_layers: usize,
    pub healthy_layers: usize,
    pub heavy_drift_layers: usize,
    pub catastrophic_drift_layers: usize,
    pub mean_angle_deg: f64,
    pub max_angle_deg: f64,
    pub max_drift_layer: Option<String>,
    pub layers: Vec<LayerSubspaceDrift>,
}

/// Compute principal angles between two column-orthonormal bases Q1 and Q2.
/// Q1 has shape (d, k1) stored as k1 column-vectors (q_cols format from thin_qr_columns).
/// Q2 has shape (d, k2).  Returns up to min(k1, k2) principal angles in radians.
///
/// Algorithm: sigma_i = SVD(Q1^T Q2) -- singular values are cosines of principal angles.
fn principal_angles_from_q_cols(
    q1_cols: &[Vec<f64>],
    q2_cols: &[Vec<f64>],
) -> Vec<f64> {
    let k1 = q1_cols.len();
    let k2 = q2_cols.len();
    if k1 == 0 || k2 == 0 {
        return Vec::new();
    }
    // Build cross-Gram matrix G = Q1^T Q2 of shape (k1, k2)
    let mut g = Tensor2D::zeros(k1, k2, DType::F32);
    for i in 0..k1 {
        for j in 0..k2 {
            let dot = dot_slice(&q1_cols[i], &q2_cols[j]);
            g.set(i, j, dot);
        }
    }
    // SVD of G: singular values are cosines of principal angles
    let svd = jacobi_svd(&g);
    let k = k1.min(k2);
    svd.sigma[..k]
        .iter()
        .map(|&cos_theta| {
            let cos = cos_theta.clamp(-1.0, 1.0).abs().min(1.0);
            cos.acos()
        })
        .collect()
}

/// Compute subspace drift between two matching LoRA layers.
pub fn compute_layer_subspace_drift(
    layer_a: &LoraLayer,
    scale_a: f64,
    layer_b: &LoraLayer,
    scale_b: f64,
) -> Option<LayerSubspaceDrift> {
    if layer_a.d_out() != layer_b.d_out() || layer_a.d_in() != layer_b.d_in() {
        return None;
    }
    if layer_a.rank() == 0 || layer_b.rank() == 0 {
        return None;
    }

    let d_out = layer_a.d_out();
    let d_in = layer_a.d_in();
    let r_a = layer_a.rank();
    let r_b = layer_b.rank();

    // Frobenius norms via Gram traces
    let frob_a = {
        let mut fn_sq = 0.0;
        for i in 0..r_a {
            for j in 0..r_a {
                let ma_ij: f64 = {
                    let ri = &layer_a.a.data[i * d_in..(i + 1) * d_in];
                    let rj = &layer_a.a.data[j * d_in..(j + 1) * d_in];
                    dot_slice(ri, rj)
                };
                let na_ij: f64 = (0..d_out)
                    .map(|k| layer_a.b.data[k * r_a + i] * layer_a.b.data[k * r_a + j])
                    .sum();
                fn_sq += na_ij * ma_ij;
            }
        }
        (scale_a * scale_a * fn_sq.max(0.0)).sqrt()
    };

    let frob_b = {
        let mut fn_sq = 0.0;
        for i in 0..r_b {
            for j in 0..r_b {
                let mb_ij: f64 = {
                    let ri = &layer_b.a.data[i * d_in..(i + 1) * d_in];
                    let rj = &layer_b.a.data[j * d_in..(j + 1) * d_in];
                    dot_slice(ri, rj)
                };
                let nb_ij: f64 = (0..d_out)
                    .map(|k| layer_b.b.data[k * r_b + i] * layer_b.b.data[k * r_b + j])
                    .sum();
                fn_sq += nb_ij * mb_ij;
            }
        }
        (scale_b * scale_b * fn_sq.max(0.0)).sqrt()
    };

    let relative_frob_drift = (frob_b - frob_a).abs() / frob_a.max(1e-12);

    // Build row-major A matrices so thin_qr_columns can orthonormalize columns of A^T
    // A_a: (r_a, d_in) -> we want V_a = right singular space = QR of A_a^T (d_in x r_a)
    let a_a_t = transpose(&layer_a.a);
    let (v_a_cols, _) = thin_qr_columns(&a_a_t);

    let a_b_t = transpose(&layer_b.a);
    let (v_b_cols, _) = thin_qr_columns(&a_b_t);

    // B matrices: (d_out, r) -> we want U = QR of B (d_out x r)
    let (u_a_cols, _) = thin_qr_columns(&layer_a.b);
    let (u_b_cols, _) = thin_qr_columns(&layer_b.b);

    let angles_input = principal_angles_from_q_cols(&v_a_cols, &v_b_cols);
    let angles_output = principal_angles_from_q_cols(&u_a_cols, &u_b_cols);

    let k = r_a.min(r_b);

    let mean_angle_input_deg = if angles_input.is_empty() {
        0.0
    } else {
        angles_input.iter().map(|a| a.to_degrees()).sum::<f64>() / angles_input.len() as f64
    };
    let mean_angle_output_deg = if angles_output.is_empty() {
        0.0
    } else {
        angles_output.iter().map(|a| a.to_degrees()).sum::<f64>() / angles_output.len() as f64
    };
    let mean_angle_deg = (mean_angle_input_deg + mean_angle_output_deg) / 2.0;

    let classification = if mean_angle_deg < 5.0 {
        SubspaceDriftClass::Stagnant
    } else if mean_angle_deg < 30.0 {
        SubspaceDriftClass::Healthy
    } else if mean_angle_deg < 60.0 {
        SubspaceDriftClass::HeavyDrift
    } else {
        SubspaceDriftClass::CatastrophicDrift
    };

    let mut issues: Vec<&'static str> = Vec::new();
    if classification == SubspaceDriftClass::Stagnant {
        issues.push("LS027");
    }
    if classification == SubspaceDriftClass::CatastrophicDrift {
        issues.push("LS028");
    }

    Some(LayerSubspaceDrift {
        base_name: layer_a.base_name.clone(),
        rank_a: r_a,
        rank_b: r_b,
        k,
        principal_angles_input: angles_input,
        mean_angle_input_deg,
        principal_angles_output: angles_output,
        mean_angle_output_deg,
        mean_angle_deg,
        frob_a,
        frob_b,
        relative_frob_drift,
        classification,
        issues,
    })
}

/// Compute subspace drift summary between two entire adapters.
pub fn analyze_subspace_drift(
    layers_a: &[LoraLayer],
    alphas_a: &BTreeMap<String, f64>,
    layers_b: &[LoraLayer],
    alphas_b: &BTreeMap<String, f64>,
) -> AdapterSubspaceDriftSummary {
    let map_a: BTreeMap<&str, &LoraLayer> =
        layers_a.iter().map(|l| (l.base_name.as_str(), l)).collect();
    let map_b: BTreeMap<&str, &LoraLayer> =
        layers_b.iter().map(|l| (l.base_name.as_str(), l)).collect();

    let mut drift_reports: Vec<LayerSubspaceDrift> = Vec::new();

    for (base_name, la) in &map_a {
        if let Some(lb) = map_b.get(base_name) {
            let scale_a = {
                let r = la.rank();
                if let Some(&alpha) = alphas_a.get(*base_name) {
                    alpha / r.max(1) as f64
                } else {
                    1.0
                }
            };
            let scale_b = {
                let r = lb.rank();
                if let Some(&alpha) = alphas_b.get(*base_name) {
                    alpha / r.max(1) as f64
                } else {
                    1.0
                }
            };
            if let Some(report) = compute_layer_subspace_drift(la, scale_a, lb, scale_b) {
                drift_reports.push(report);
            }
        }
    }

    drift_reports.sort_by(|a, b| b.mean_angle_deg.partial_cmp(&a.mean_angle_deg).unwrap_or(std::cmp::Ordering::Equal));

    let total = drift_reports.len();
    let stagnant = drift_reports.iter().filter(|r| r.classification == SubspaceDriftClass::Stagnant).count();
    let healthy = drift_reports.iter().filter(|r| r.classification == SubspaceDriftClass::Healthy).count();
    let heavy = drift_reports.iter().filter(|r| r.classification == SubspaceDriftClass::HeavyDrift).count();
    let catastrophic = drift_reports.iter().filter(|r| r.classification == SubspaceDriftClass::CatastrophicDrift).count();

    let mean_angle_deg = if total > 0 {
        drift_reports.iter().map(|r| r.mean_angle_deg).sum::<f64>() / total as f64
    } else {
        0.0
    };
    let max_angle_deg = drift_reports.iter().map(|r| r.mean_angle_deg).fold(0.0_f64, f64::max);
    let max_drift_layer = drift_reports.first().map(|r| r.base_name.clone());

    AdapterSubspaceDriftSummary {
        total_shared_layers: total,
        stagnant_layers: stagnant,
        healthy_layers: healthy,
        heavy_drift_layers: heavy,
        catastrophic_drift_layers: catastrophic,
        mean_angle_deg,
        max_angle_deg,
        max_drift_layer,
        layers: drift_reports,
    }
}

/// Render subspace drift summary as compact JSON.
pub fn render_subspace_drift_json(summary: &AdapterSubspaceDriftSummary) -> String {
    let mut out = String::new();
    out.push_str("{\n");
    out.push_str(&format!("  \"total_shared_layers\": {},\n", summary.total_shared_layers));
    out.push_str(&format!("  \"stagnant_layers\": {},\n", summary.stagnant_layers));
    out.push_str(&format!("  \"healthy_layers\": {},\n", summary.healthy_layers));
    out.push_str(&format!("  \"heavy_drift_layers\": {},\n", summary.heavy_drift_layers));
    out.push_str(&format!("  \"catastrophic_drift_layers\": {},\n", summary.catastrophic_drift_layers));
    out.push_str(&format!("  \"mean_angle_deg\": {:.4},\n", summary.mean_angle_deg));
    out.push_str(&format!("  \"max_angle_deg\": {:.4},\n", summary.max_angle_deg));
    if let Some(ref layer) = summary.max_drift_layer {
        out.push_str(&format!("  \"max_drift_layer\": \"{}\",\n", layer));
    } else {
        out.push_str("  \"max_drift_layer\": null,\n");
    }
    out.push_str("  \"layers\": [\n");
    for (i, layer) in summary.layers.iter().enumerate() {
        out.push_str("    {\n");
        out.push_str(&format!("      \"base_name\": \"{}\",\n", layer.base_name));
        out.push_str(&format!("      \"rank_a\": {},\n", layer.rank_a));
        out.push_str(&format!("      \"rank_b\": {},\n", layer.rank_b));
        out.push_str(&format!("      \"mean_angle_input_deg\": {:.4},\n", layer.mean_angle_input_deg));
        out.push_str(&format!("      \"mean_angle_output_deg\": {:.4},\n", layer.mean_angle_output_deg));
        out.push_str(&format!("      \"mean_angle_deg\": {:.4},\n", layer.mean_angle_deg));
        out.push_str(&format!("      \"frob_a\": {:.6},\n", layer.frob_a));
        out.push_str(&format!("      \"frob_b\": {:.6},\n", layer.frob_b));
        out.push_str(&format!("      \"relative_frob_drift\": {:.6},\n", layer.relative_frob_drift));
        out.push_str(&format!("      \"classification\": \"{}\",\n", layer.classification.as_str()));
        out.push_str(&format!(
            "      \"issues\": [{}]\n",
            layer.issues.iter().map(|s| format!("\"{}\"", s)).collect::<Vec<_>>().join(", ")
        ));
        out.push_str("    }");
        if i + 1 < summary.layers.len() {
            out.push(',');
        }
        out.push('\n');
    }
    out.push_str("  ]\n");
    out.push('}');
    out
}

/// Render subspace drift summary as a human-readable table.
pub fn render_subspace_drift_table(summary: &AdapterSubspaceDriftSummary) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "Subspace Drift Analysis: {}/{} healthy, {} stagnant, {} catastrophic drift\n",
        summary.healthy_layers, summary.total_shared_layers, summary.stagnant_layers, summary.catastrophic_drift_layers
    ));
    out.push_str(&format!(
        "Mean angle: {:.2} deg  Max angle: {:.2} deg\n\n",
        summary.mean_angle_deg, summary.max_angle_deg
    ));
    out.push_str(&format!(
        "{:<50} {:>5} {:>5} {:>12} {:>13} {:>12} {:<}\n",
        "Layer", "r_A", "r_B", "angle_in(deg)", "angle_out(deg)", "rel_frob", "class"
    ));
    out.push_str(&"-".repeat(115));
    out.push('\n');
    for layer in &summary.layers {
        let short = if layer.base_name.len() > 50 {
            format!("...{}", &layer.base_name[layer.base_name.len() - 47..])
        } else {
            layer.base_name.clone()
        };
        let issues_str = if layer.issues.is_empty() {
            String::new()
        } else {
            format!(" [{}]", layer.issues.join(", "))
        };
        out.push_str(&format!(
            "{:<50} {:>5} {:>5} {:>12.4} {:>13.4} {:>12.4} {}{}\n",
            short,
            layer.rank_a,
            layer.rank_b,
            layer.mean_angle_input_deg,
            layer.mean_angle_output_deg,
            layer.relative_frob_drift,
            layer.classification.as_str(),
            issues_str
        ));
    }
    out
}

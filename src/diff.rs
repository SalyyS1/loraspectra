use std::collections::BTreeMap;

use crate::audit::pearson_correlation;
use crate::linalg::{dot_slice, matmul, thin_qr_columns, transpose};
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

use std::collections::BTreeMap;

use crate::linalg::{dot_slice, thin_qr_columns, transpose};
use crate::safetensors::LoraLayer;

#[derive(Debug, Clone)]
pub struct LayerDiffReport {
    pub base_name: String,
    pub rank_a: usize,
    pub rank_b: usize,
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

    // 3. Cross term: P = A_A * A_B^T (r_A x r_B), Q = B_A^T * B_B (r_A x r_B)
    let mut p = vec![0.0; r_a * r_b];
    for i in 0..r_a {
        let row_a = &layer_a.a.data[i * d_in..(i + 1) * d_in];
        for j in 0..r_b {
            let row_b = &layer_b.a.data[j * d_in..(j + 1) * d_in];
            p[i * r_b + j] = dot_slice(row_a, row_b);
        }
    }

    let mut q = vec![0.0; r_a * r_b];
    for i in 0..r_a {
        for j in 0..r_b {
            let mut dot = 0.0;
            for k in 0..d_out {
                dot += layer_a.b.data[k * r_a + i] * layer_b.b.data[k * r_b + j];
            }
            q[i * r_b + j] = dot;
        }
    }

    let mut cross_trace = 0.0;
    for i in 0..r_a {
        for j in 0..r_b {
            cross_trace += q[i * r_b + j] * p[i * r_b + j];
        }
    }
    let scaled_cross = scale_a * scale_b * cross_trace;

    let diff_sq = (norm_a * norm_a + norm_b * norm_b - 2.0 * scaled_cross).max(0.0);
    let frobenius_delta = diff_sq.sqrt();
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

    Some(LayerDiffReport {
        base_name: layer_a.base_name.clone(),
        rank_a: r_a,
        rank_b: r_b,
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
    })
}

/// Diff two LoRA adapters across all layers.
pub fn diff_adapters(
    layers_a: &[LoraLayer],
    alphas_a: &BTreeMap<String, f64>,
    layers_b: &[LoraLayer],
    alphas_b: &BTreeMap<String, f64>,
) -> AdapterDiffSummary {
    let mut map_a: BTreeMap<&str, &LoraLayer> = BTreeMap::new();
    for l in layers_a {
        map_a.insert(l.base_name.as_str(), l);
    }

    let mut map_b: BTreeMap<&str, &LoraLayer> = BTreeMap::new();
    for l in layers_b {
        map_b.insert(l.base_name.as_str(), l);
    }

    let mut shared_reports: Vec<LayerDiffReport> = Vec::new();
    let mut removed_layers: Vec<String> = Vec::new();
    let mut added_layers: Vec<String> = Vec::new();

    for (name, layer_a) in &map_a {
        if let Some(layer_b) = map_b.get(name) {
            let scale_a = compute_layer_scaling(name, layer_a.rank(), alphas_a);
            let scale_b = compute_layer_scaling(name, layer_b.rank(), alphas_b);
            if let Some(report) = diff_layer(layer_a, scale_a, layer_b, scale_b) {
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

    AdapterDiffSummary {
        shared_layers: shared_reports,
        added_layers,
        removed_layers,
        mean_relative_drift,
        mean_subspace_similarity,
        max_drift_layer,
        max_relative_drift,
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
      "rank_delta": {},
      "d_out": {},
      "d_in": {},
      "frobenius_norm_a": {:.6},
      "frobenius_norm_b": {:.6},
      "frobenius_delta": {:.6},
      "relative_drift": {:.6},
      "subspace_similarity_in": {:.6},
      "subspace_similarity_out": {:.6},
      "subspace_similarity": {:.6}
    }}"#,
                escape_json(&r.base_name),
                r.rank_a,
                r.rank_b,
                r.rank_delta,
                r.d_out,
                r.d_in,
                r.frobenius_norm_a,
                r.frobenius_norm_b,
                r.frobenius_delta,
                r.relative_drift,
                r.subspace_similarity_in,
                r.subspace_similarity_out,
                r.subspace_similarity
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

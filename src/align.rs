/// Cross-layer subspace alignment matrix for adjacent Transformer LoRA adapter layers.
///
/// For a LoRA adapter with N layers this module computes, for every consecutive pair
/// (sorted by base_name), the Grassmannian distance between their input subspaces.
///
/// The input subspace of a LoRA layer is the row space of its A-factor
/// (shape rank x d_in): a rank-dimensional subspace of R^{d_in}.
/// We orthonormalize it via thin-QR of A^T (shape d_in x rank) to obtain Q,
/// then compute principal angles between Q_a and Q_b via SVD of Q_a^T Q_b.
///
/// The Grassmannian distance between two rank-r subspaces is:
///
///   d_G(P, Q) = ||[theta_1, ..., theta_k]||_2
///
/// where theta_i are principal angles in [0, pi/2] and k = min(rank_a, rank_b).
///
/// References:
///   Absil, Mahony, Sepulchre (2004). Riemannian Geometry of Grassmann Manifolds.
///   Bjorck, Golub (1973). Numerical methods for computing angles between linear subspaces.

use crate::linalg::{dot_slice, jacobi_svd, thin_qr_columns};
use crate::safetensors::{DType, LoraLayer, Tensor2D};

/// Health classification for a cross-layer alignment pair.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlignmentClass {
    /// Grassmannian distance < pi/12 (~0.262 rad, mean angle < 15 deg):
    /// highly aligned -- consecutive layers share similar update directions,
    /// suggesting redundant adaptation capacity.
    HighlyAligned,
    /// Grassmannian distance in [pi/12, pi/4] (~0.262-0.785 rad): typical healthy diversity.
    Diverse,
    /// Grassmannian distance in (pi/4, pi/2.5] (~0.785-1.257 rad): substantial orthogonality.
    Orthogonal,
    /// Grassmannian distance > pi/2.5 (~1.257 rad): layers are nearly maximally disjoint.
    MaximallyDisjoint,
}

impl AlignmentClass {
    pub fn as_str(&self) -> &'static str {
        match self {
            AlignmentClass::HighlyAligned => "HIGHLY_ALIGNED",
            AlignmentClass::Diverse => "DIVERSE",
            AlignmentClass::Orthogonal => "ORTHOGONAL",
            AlignmentClass::MaximallyDisjoint => "MAXIMALLY_DISJOINT",
        }
    }
}

/// SARIF rule IDs for cross-layer alignment diagnostics.
pub const LS029_REDUNDANT_ALIGNMENT: &str = "LS029";
pub const LS030_MAXIMAL_DISJOINT: &str = "LS030";

/// Per adjacent-pair cross-layer alignment result.
#[derive(Debug, Clone)]
pub struct LayerAlignmentPair {
    /// Base name of the first (lower-sorted) layer.
    pub layer_a: String,
    /// Base name of the adjacent layer.
    pub layer_b: String,
    /// k = min(rank_a, rank_b): number of principal angles.
    pub k: usize,
    /// Principal angles in radians, sorted ascending.
    pub principal_angles: Vec<f64>,
    /// Grassmannian distance: Euclidean norm of the principal angle vector.
    pub grassmannian_distance: f64,
    /// Mean principal angle in degrees.
    pub mean_angle_deg: f64,
    /// Maximum principal angle in degrees.
    pub max_angle_deg: f64,
    /// Alignment classification.
    pub classification: AlignmentClass,
    /// SARIF rule IDs triggered.
    pub issues: Vec<&'static str>,
}

/// Full cross-layer subspace alignment matrix for a LoRA adapter.
#[derive(Debug, Clone)]
pub struct AdapterAlignmentMatrix {
    /// All adjacent-pair alignment results, in sorted-by-name order.
    pub pairs: Vec<LayerAlignmentPair>,
    /// Count of HIGHLY_ALIGNED pairs (LS029).
    pub redundant_pairs: usize,
    /// Count of MAXIMALLY_DISJOINT pairs (LS030).
    pub disjoint_pairs: usize,
    /// Mean Grassmannian distance across all adjacent pairs.
    pub mean_grassmannian_distance: f64,
    /// Max Grassmannian distance across all adjacent pairs.
    pub max_grassmannian_distance: f64,
    /// (layer_a, layer_b) with the highest Grassmannian distance.
    pub most_disjoint_pair: Option<(String, String)>,
}

/// Compute principal angles between two sets of orthonormal column vectors.
///
/// q1_cols[i] is the i-th column of Q1 (a d-dimensional unit vector).
/// q2_cols[j] is the j-th column of Q2.
/// Returns up to min(len(q1_cols), len(q2_cols)) principal angles in radians,
/// sorted ascending.
fn principal_angles(q1_cols: &[Vec<f64>], q2_cols: &[Vec<f64>]) -> Vec<f64> {
    let k1 = q1_cols.len();
    let k2 = q2_cols.len();
    if k1 == 0 || k2 == 0 {
        return Vec::new();
    }
    // Build cross-Gram matrix G = Q1^T Q2 of shape (k1, k2)
    let mut g = Tensor2D::zeros(k1, k2, DType::F32);
    for i in 0..k1 {
        for j in 0..k2 {
            g.set(i, j, dot_slice(&q1_cols[i], &q2_cols[j]));
        }
    }
    // SVD of G: singular values are cosines of principal angles
    let svd = jacobi_svd(&g);
    let k = k1.min(k2);
    let mut angles: Vec<f64> = svd.sigma[..k]
        .iter()
        .map(|&cos_theta| cos_theta.clamp(-1.0, 1.0).abs().min(1.0).acos())
        .collect();
    angles.sort_by(|a, b| a.partial_cmp(b).unwrap());
    angles
}

/// Orthonormalize the row space of a LoRA A-factor (shape rank x d_in) via thin-QR.
///
/// A^T has shape (d_in x rank); thin_qr_columns(A^T) returns `rank` orthonormal
/// column vectors each of dimension `d_in`. We build A^T explicitly.
fn input_subspace_q_cols(a: &Tensor2D) -> Vec<Vec<f64>> {
    let rank = a.rows;
    let d_in = a.cols;
    if rank == 0 || d_in == 0 {
        return Vec::new();
    }
    // Build A^T of shape (d_in x rank) in row-major order
    let mut at = Tensor2D::zeros(d_in, rank, DType::F32);
    for r in 0..rank {
        for c in 0..d_in {
            at.set(c, r, a.data[r * d_in + c]);
        }
    }
    // thin_qr_columns(m: &Tensor2D) where m is (rows x cols) returns `cols` column-vectors of length `rows`.
    // Here m = A^T is (d_in x rank), so we get `rank` column-vectors of length `d_in`.
    let (q_cols, _r) = thin_qr_columns(&at);
    // Filter out zero columns (rank-deficient rows)
    q_cols
        .into_iter()
        .filter(|col| dot_slice(col, col) > 1e-14)
        .collect()
}

/// Compute the alignment pair result for two adjacent LoRA layers.
/// Returns `None` when either layer has an empty or rank-zero subspace.
fn compute_pair_alignment(la: &LoraLayer, lb: &LoraLayer) -> Option<LayerAlignmentPair> {
    let q_a = input_subspace_q_cols(&la.a);
    let q_b = input_subspace_q_cols(&lb.a);
    if q_a.is_empty() || q_b.is_empty() {
        return None;
    }

    let angles = principal_angles(&q_a, &q_b);
    if angles.is_empty() {
        return None;
    }

    let k = angles.len();
    let grass_dist = angles.iter().map(|&t| t * t).sum::<f64>().sqrt();
    let mean_angle_deg =
        angles.iter().sum::<f64>() / (k as f64) * (180.0 / std::f64::consts::PI);
    let max_angle_deg = angles
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max)
        * (180.0 / std::f64::consts::PI);

    // Thresholds in terms of Grassmannian distance (radians):
    //   pi/12 ~ 0.2618, pi/4 ~ 0.7854, pi/2.5 ~ 1.2566
    let classification = if grass_dist < std::f64::consts::PI / 12.0 {
        AlignmentClass::HighlyAligned
    } else if grass_dist <= std::f64::consts::PI / 4.0 {
        AlignmentClass::Diverse
    } else if grass_dist <= std::f64::consts::PI / 2.5 {
        AlignmentClass::Orthogonal
    } else {
        AlignmentClass::MaximallyDisjoint
    };

    let mut issues: Vec<&'static str> = Vec::new();
    if classification == AlignmentClass::HighlyAligned {
        issues.push(LS029_REDUNDANT_ALIGNMENT);
    }
    if classification == AlignmentClass::MaximallyDisjoint {
        issues.push(LS030_MAXIMAL_DISJOINT);
    }

    Some(LayerAlignmentPair {
        layer_a: la.base_name.clone(),
        layer_b: lb.base_name.clone(),
        k,
        principal_angles: angles,
        grassmannian_distance: grass_dist,
        mean_angle_deg,
        max_angle_deg,
        classification,
        issues,
    })
}

/// Compute cross-layer subspace alignment for all adjacent layer pairs in a LoRA adapter.
///
/// Layers are sorted by base_name before pairing, so numerically adjacent Transformer blocks
/// (e.g. layers.0 -> layers.1) are always consecutive.
pub fn analyze_cross_layer_alignment(layers: &[LoraLayer]) -> AdapterAlignmentMatrix {
    if layers.len() < 2 {
        return AdapterAlignmentMatrix {
            pairs: Vec::new(),
            redundant_pairs: 0,
            disjoint_pairs: 0,
            mean_grassmannian_distance: 0.0,
            max_grassmannian_distance: 0.0,
            most_disjoint_pair: None,
        };
    }

    let mut sorted: Vec<&LoraLayer> = layers.iter().collect();
    sorted.sort_by(|a, b| a.base_name.cmp(&b.base_name));

    let mut pairs: Vec<LayerAlignmentPair> = Vec::new();
    for window in sorted.windows(2) {
        if let Some(pair) = compute_pair_alignment(window[0], window[1]) {
            pairs.push(pair);
        }
    }

    if pairs.is_empty() {
        return AdapterAlignmentMatrix {
            pairs: Vec::new(),
            redundant_pairs: 0,
            disjoint_pairs: 0,
            mean_grassmannian_distance: 0.0,
            max_grassmannian_distance: 0.0,
            most_disjoint_pair: None,
        };
    }

    let redundant_pairs = pairs
        .iter()
        .filter(|p| p.classification == AlignmentClass::HighlyAligned)
        .count();
    let disjoint_pairs = pairs
        .iter()
        .filter(|p| p.classification == AlignmentClass::MaximallyDisjoint)
        .count();
    let mean_grassmannian_distance =
        pairs.iter().map(|p| p.grassmannian_distance).sum::<f64>() / pairs.len() as f64;
    let max_grassmannian_distance = pairs
        .iter()
        .map(|p| p.grassmannian_distance)
        .fold(f64::NEG_INFINITY, f64::max);
    let most_disjoint_pair = pairs
        .iter()
        .max_by(|a, b| {
            a.grassmannian_distance
                .partial_cmp(&b.grassmannian_distance)
                .unwrap()
        })
        .map(|p| (p.layer_a.clone(), p.layer_b.clone()));

    AdapterAlignmentMatrix {
        pairs,
        redundant_pairs,
        disjoint_pairs,
        mean_grassmannian_distance,
        max_grassmannian_distance,
        most_disjoint_pair,
    }
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

fn short_name(full: &str) -> String {
    let prefix = "base_model.model.";
    if full.starts_with(prefix) {
        full[prefix.len()..].to_string()
    } else {
        full.to_string()
    }
}

/// Render the alignment matrix as a compact JSON object.
pub fn render_alignment_json(matrix: &AdapterAlignmentMatrix) -> String {
    let mut out = String::new();
    out.push_str("{\n");
    out.push_str(&format!("  \"total_pairs\": {},\n", matrix.pairs.len()));
    out.push_str(&format!(
        "  \"redundant_pairs\": {},\n",
        matrix.redundant_pairs
    ));
    out.push_str(&format!(
        "  \"disjoint_pairs\": {},\n",
        matrix.disjoint_pairs
    ));
    out.push_str(&format!(
        "  \"mean_grassmannian_distance\": {:.6},\n",
        matrix.mean_grassmannian_distance
    ));
    out.push_str(&format!(
        "  \"max_grassmannian_distance\": {:.6},\n",
        matrix.max_grassmannian_distance
    ));
    if let Some((a, b)) = &matrix.most_disjoint_pair {
        out.push_str(&format!(
            "  \"most_disjoint_pair\": [\"{}\", \"{}\"],\n",
            json_escape(a),
            json_escape(b)
        ));
    } else {
        out.push_str("  \"most_disjoint_pair\": null,\n");
    }
    out.push_str("  \"pairs\": [\n");
    for (i, pair) in matrix.pairs.iter().enumerate() {
        let comma = if i + 1 < matrix.pairs.len() { "," } else { "" };
        out.push_str("    {\n");
        out.push_str(&format!(
            "      \"layer_a\": \"{}\",\n",
            json_escape(&pair.layer_a)
        ));
        out.push_str(&format!(
            "      \"layer_b\": \"{}\",\n",
            json_escape(&pair.layer_b)
        ));
        out.push_str(&format!("      \"k\": {},\n", pair.k));
        out.push_str(&format!(
            "      \"grassmannian_distance\": {:.6},\n",
            pair.grassmannian_distance
        ));
        out.push_str(&format!(
            "      \"mean_angle_deg\": {:.3},\n",
            pair.mean_angle_deg
        ));
        out.push_str(&format!(
            "      \"max_angle_deg\": {:.3},\n",
            pair.max_angle_deg
        ));
        out.push_str(&format!(
            "      \"classification\": \"{}\",\n",
            pair.classification.as_str()
        ));
        let issues_str = pair
            .issues
            .iter()
            .map(|s| format!("\"{}\"", s))
            .collect::<Vec<_>>()
            .join(", ");
        out.push_str(&format!("      \"issues\": [{}]\n", issues_str));
        out.push_str(&format!("    }}{}\n", comma));
    }
    out.push_str("  ]\n");
    out.push('}');
    out
}

/// Render the alignment matrix as a human-readable text table.
pub fn render_alignment_table(matrix: &AdapterAlignmentMatrix) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "Cross-Layer Subspace Alignment  ({} adjacent pairs)\n",
        matrix.pairs.len()
    ));
    out.push_str(&format!(
        "  Mean Grassmannian distance : {:.4} rad\n",
        matrix.mean_grassmannian_distance
    ));
    out.push_str(&format!(
        "  Max Grassmannian distance  : {:.4} rad\n",
        matrix.max_grassmannian_distance
    ));
    out.push_str(&format!(
        "  Redundant (LS029)          : {}\n",
        matrix.redundant_pairs
    ));
    out.push_str(&format!(
        "  Maximally disjoint (LS030) : {}\n\n",
        matrix.disjoint_pairs
    ));
    if matrix.pairs.is_empty() {
        out.push_str("  No adjacent layer pairs found.\n");
        return out;
    }
    out.push_str(&format!(
        "{:<52} {:>5}  {:>8}  {:>8}  {:<20}  {}\n",
        "Layer A (lower-sorted)", "k", "G.dist", "MeanAng", "Class", "Issues"
    ));
    out.push_str(&"-".repeat(115));
    out.push('\n');
    for pair in &matrix.pairs {
        let la_short = short_name(&pair.layer_a);
        let issues_str = if pair.issues.is_empty() {
            String::new()
        } else {
            pair.issues.join(", ")
        };
        out.push_str(&format!(
            "{:<52} {:>5}  {:>8.4}  {:>7.1}d  {:<20}  {}\n",
            la_short,
            pair.k,
            pair.grassmannian_distance,
            pair.mean_angle_deg,
            pair.classification.as_str(),
            issues_str
        ));
    }
    out
}

fn html_esc_align(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

/// Render a self-contained dark-mode HTML Cross-Layer Grassmannian Subspace Alignment Report (`align --html`).
pub fn render_alignment_html(matrix: &AdapterAlignmentMatrix) -> String {
    let mut rows = String::new();
    for pair in &matrix.pairs {
        let col = match pair.classification {
            AlignmentClass::HighlyAligned | AlignmentClass::MaximallyDisjoint => "#f85149",
            AlignmentClass::Orthogonal => "#d29922",
            AlignmentClass::Diverse => "#3fb950",
        };
        let issues = if pair.issues.is_empty() {
            "—".to_string()
        } else {
            pair.issues.join(", ")
        };
        rows.push_str(&format!(
            "<tr><td><code>{}</code> &harr; <code>{}</code></td><td>{}</td><td>{:.4}</td><td>{:.1}&deg;</td><td style=\"color:{}\"><b>{}</b></td><td><code>{}</code></td></tr>\n",
            html_esc_align(&pair.layer_a),
            html_esc_align(&pair.layer_b),
            pair.k,
            pair.grassmannian_distance,
            pair.mean_angle_deg,
            col,
            pair.classification.as_str(),
            html_esc_align(&issues)
        ));
    }
    format!(
        r##"<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>loraspectra Cross-Layer Grassmannian Alignment Report</title>
<style>
body{{margin:0;padding:28px;background:#0b0f17;color:#e6edf3;font-family:system-ui,sans-serif}}
.wrap{{max-width:1080px;margin:0 auto}}.card{{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:18px;margin-bottom:16px}}
table{{width:100%;border-collapse:collapse;font-size:12.5px}}th,td{{padding:8px 9px;border-bottom:1px solid #21262d;text-align:left}}
th{{color:#8b949e}}code{{font-family:ui-monospace,monospace;color:#58a6ff}}
</style></head><body><div class="wrap">
<div class="card"><h1>loraspectra Cross-Layer Grassmannian Subspace Alignment</h1>
<p>Adjacent Pairs: <b>{}</b> &middot; Mean d_G: <b>{:.4}</b> &middot; Max d_G: <b>{:.4}</b> &middot; Redundant (LS029): <b>{}</b> &middot; Disjoint (LS030): <b>{}</b></p></div>
<div class="card"><table><thead><tr><th>Adjacent Pair</th><th>k</th><th>d_G (rad)</th><th>Mean Angle</th><th>Class</th><th>SARIF</th></tr></thead>
<tbody>{}</tbody></table></div></div></body></html>"##,
        matrix.pairs.len(),
        matrix.mean_grassmannian_distance,
        matrix.max_grassmannian_distance,
        matrix.redundant_pairs,
        matrix.disjoint_pairs,
        rows
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::safetensors::{DType, LoraLayer, Tensor2D};
    use crate::Rng64;

    fn rand_layer(name: &str, rank: usize, d_in: usize, d_out: usize, seed: u64) -> LoraLayer {
        let mut rng = Rng64::new(seed);
        let a_data: Vec<f64> = (0..rank * d_in).map(|_| rng.next_gaussian()).collect();
        let b_data: Vec<f64> = (0..d_out * rank).map(|_| rng.next_gaussian()).collect();
        LoraLayer {
            base_name: name.to_string(),
            a_key: format!("{}.lora_A.weight", name),
            b_key: format!("{}.lora_B.weight", name),
            a: Tensor2D { rows: rank, cols: d_in, data: a_data, dtype: DType::F32 },
            b: Tensor2D { rows: d_out, cols: rank, data: b_data, dtype: DType::F32 },
        }
    }

    fn standard_basis_layer(name: &str, rank: usize, d_in: usize, d_out: usize, start: usize, seed_b: u64) -> LoraLayer {
        // A rows = e_{start}, e_{start+1}, ..., e_{start+rank-1}  (standard basis vectors)
        let mut a_data = vec![0.0_f64; rank * d_in];
        for r in 0..rank {
            a_data[r * d_in + start + r] = 1.0;
        }
        let mut rng = Rng64::new(seed_b);
        let b_data: Vec<f64> = (0..d_out * rank).map(|_| rng.next_gaussian()).collect();
        LoraLayer {
            base_name: name.to_string(),
            a_key: format!("{}.lora_A.weight", name),
            b_key: format!("{}.lora_B.weight", name),
            a: Tensor2D { rows: rank, cols: d_in, data: a_data, dtype: DType::F32 },
            b: Tensor2D { rows: d_out, cols: rank, data: b_data, dtype: DType::F32 },
        }
    }

    #[test]
    fn test_principal_angles_identical_subspace() {
        // Q1 == Q2: all principal angles should be 0
        let cols = vec![vec![1.0_f64, 0.0, 0.0], vec![0.0, 1.0, 0.0]];
        let angles = principal_angles(&cols, &cols);
        assert_eq!(angles.len(), 2);
        for a in &angles {
            assert!(a.abs() < 1e-6, "angle should be ~0 for identical subspaces, got {}", a);
        }
    }

    #[test]
    fn test_principal_angles_orthogonal_subspace() {
        // e_0 vs e_1: principal angle = pi/2
        let q1 = vec![vec![1.0_f64, 0.0, 0.0]];
        let q2 = vec![vec![0.0_f64, 1.0, 0.0]];
        let angles = principal_angles(&q1, &q2);
        assert_eq!(angles.len(), 1);
        let expected = std::f64::consts::FRAC_PI_2;
        assert!(
            (angles[0] - expected).abs() < 1e-6,
            "angle should be pi/2 for orthogonal subspaces, got {}",
            angles[0]
        );
    }

    #[test]
    fn test_grassmannian_distance_identical_layer() {
        let la = rand_layer("layers.0.q_proj", 4, 32, 64, 42);
        let mut lb = la.clone();
        lb.base_name = "layers.1.q_proj".to_string();
        let pair = compute_pair_alignment(&la, &lb).expect("pair should compute");
        assert!(
            pair.grassmannian_distance < 0.05,
            "Grassmannian dist for identical A should be near 0, got {}",
            pair.grassmannian_distance
        );
        assert_eq!(pair.classification, AlignmentClass::HighlyAligned);
        assert!(pair.issues.contains(&LS029_REDUNDANT_ALIGNMENT));
    }

    #[test]
    fn test_grassmannian_distance_nonzero_random() {
        let la = rand_layer("layers.0.q_proj", 4, 32, 64, 1);
        let lb = rand_layer("layers.1.q_proj", 4, 32, 64, 2);
        let pair = compute_pair_alignment(&la, &lb).expect("pair should compute");
        assert!(pair.grassmannian_distance > 0.0);
        assert_eq!(pair.k, 4);
        assert!(pair.mean_angle_deg >= 0.0 && pair.mean_angle_deg <= 90.0);
    }

    #[test]
    fn test_grassmannian_distance_fully_orthogonal() {
        // A rows in e_0,e_1 vs e_2,e_3 -> principal angles all = pi/2
        let d_in = 32;
        let rank = 2;
        let la = standard_basis_layer("layers.0.q_proj", rank, d_in, 64, 0, 10);
        let lb = standard_basis_layer("layers.1.q_proj", rank, d_in, 64, 2, 11);
        let pair = compute_pair_alignment(&la, &lb).expect("pair");
        for &a in &pair.principal_angles {
            assert!(
                (a - std::f64::consts::FRAC_PI_2).abs() < 1e-5,
                "expected pi/2, got {}",
                a
            );
        }
        assert!(pair.grassmannian_distance > std::f64::consts::PI / 2.5);
        assert_eq!(pair.classification, AlignmentClass::MaximallyDisjoint);
        assert!(pair.issues.contains(&LS030_MAXIMAL_DISJOINT));
    }

    #[test]
    fn test_alignment_matrix_single_layer_empty() {
        let la = rand_layer("layers.0.q_proj", 4, 32, 64, 10);
        let matrix = analyze_cross_layer_alignment(&[la]);
        assert_eq!(matrix.pairs.len(), 0);
    }

    #[test]
    fn test_alignment_matrix_two_layers() {
        let la = rand_layer("layers.0.q_proj", 4, 32, 64, 10);
        let lb = rand_layer("layers.1.q_proj", 4, 32, 64, 11);
        let matrix = analyze_cross_layer_alignment(&[la, lb]);
        assert_eq!(matrix.pairs.len(), 1);
        let pair = &matrix.pairs[0];
        assert_eq!(pair.k, 4);
        assert!(pair.grassmannian_distance.is_finite());
    }

    #[test]
    fn test_alignment_matrix_sorted_by_name() {
        let l0 = rand_layer("model.layers.0.q_proj", 2, 16, 32, 100);
        let l1 = rand_layer("model.layers.1.q_proj", 2, 16, 32, 101);
        let l2 = rand_layer("model.layers.2.q_proj", 2, 16, 32, 102);
        // Insert in reverse order; matrix should sort by base_name
        let matrix = analyze_cross_layer_alignment(&[l2, l0, l1]);
        assert_eq!(matrix.pairs.len(), 2);
        assert_eq!(matrix.pairs[0].layer_a, "model.layers.0.q_proj");
        assert_eq!(matrix.pairs[1].layer_a, "model.layers.1.q_proj");
    }

    #[test]
    fn test_alignment_matrix_redundant_flag() {
        let la = rand_layer("model.layers.0.q_proj", 4, 32, 64, 7);
        let mut lb = la.clone();
        lb.base_name = "model.layers.1.q_proj".to_string();
        let matrix = analyze_cross_layer_alignment(&[la, lb]);
        assert!(matrix.redundant_pairs >= 1);
    }

    #[test]
    fn test_alignment_pair_k_is_min_rank() {
        // rank_a=2, rank_b=4 -> k = 2
        let la = rand_layer("layers.0.q", 2, 32, 64, 200);
        let lb = rand_layer("layers.1.q", 4, 32, 64, 201);
        let pair = compute_pair_alignment(&la, &lb).expect("pair");
        assert_eq!(pair.k, 2);
    }

    #[test]
    fn test_alignment_matrix_many_layers() {
        let layers: Vec<LoraLayer> = (0..8_usize)
            .map(|i| rand_layer(&format!("model.layers.{}.q_proj", i), 4, 32, 64, i as u64 + 300))
            .collect();
        let matrix = analyze_cross_layer_alignment(&layers);
        assert_eq!(matrix.pairs.len(), 7);
        assert!(matrix.mean_grassmannian_distance.is_finite());
        assert!(matrix.max_grassmannian_distance >= matrix.mean_grassmannian_distance);
        assert!(matrix.most_disjoint_pair.is_some());
    }

    #[test]
    fn test_render_json_contains_expected_keys() {
        let l0 = rand_layer("layers.0.q_proj", 2, 16, 32, 50);
        let l1 = rand_layer("layers.1.q_proj", 2, 16, 32, 51);
        let matrix = analyze_cross_layer_alignment(&[l0, l1]);
        let json = render_alignment_json(&matrix);
        assert!(json.contains("\"total_pairs\""));
        assert!(json.contains("\"grassmannian_distance\""));
        assert!(json.contains("\"classification\""));
        assert!(json.contains("\"mean_angle_deg\""));
    }

    #[test]
    fn test_render_table_contains_header() {
        let l0 = rand_layer("layers.0.q_proj", 2, 16, 32, 60);
        let l1 = rand_layer("layers.1.q_proj", 2, 16, 32, 61);
        let matrix = analyze_cross_layer_alignment(&[l0, l1]);
        let table = render_alignment_table(&matrix);
        assert!(table.contains("Cross-Layer Subspace Alignment"));
        assert!(table.contains("Grassmannian"));
    }

    #[test]
    fn test_mean_distance_between_zero_and_max() {
        let layers: Vec<LoraLayer> = (0..6_usize)
            .map(|i| rand_layer(&format!("layers.{}.q", i), 3, 24, 48, i as u64 + 400))
            .collect();
        let matrix = analyze_cross_layer_alignment(&layers);
        assert!(matrix.mean_grassmannian_distance <= matrix.max_grassmannian_distance + 1e-10);
        assert!(matrix.mean_grassmannian_distance >= 0.0);
    }
}

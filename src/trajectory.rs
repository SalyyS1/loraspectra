//! Cross-Checkpoint LoRA Training Trajectory & Convergence Telemetry (`loraspectra trajectory`).
//!
//! Given an ordered sequence of training checkpoints (`ckpt_1`, `ckpt_2`, ..., `ckpt_N`),
//! computes per-step QR-Core SVD effective rank, Frobenius norm, consecutive Grassmannian
//! subspace stability, Frobenius velocity, early-stopping / rank-collapse phase diagnosis,
//! and identifies the optimal checkpoint along the training curve — with zero PyTorch overhead.

use crate::audit::{audit_adapter, LayerHealth};
use crate::diff::diff_adapters_with_dora;
use crate::remote::resolve_source_path;
use crate::safetensors::{extract_dora_scales, extract_lora_alphas, extract_lora_layers, read_safetensors};

#[derive(Debug, Clone)]
pub struct CheckpointPoint {
    pub index: usize,
    pub source: String,
    pub layer_count: usize,
    pub mean_effective_rank: f64,
    pub mean_utilization: f64,
    pub total_frobenius_norm: f64,
    pub collapsed_layers: usize,
    pub health_score: f64,
}

#[derive(Debug, Clone)]
pub struct TrajectoryTransition {
    pub from_index: usize,
    pub to_index: usize,
    pub from_source: String,
    pub to_source: String,
    pub frobenius_velocity: f64,
    pub mean_subspace_overlap: f64,
    pub effective_rank_delta: f64,
}

#[derive(Debug, Clone)]
pub struct TrajectoryReport {
    pub checkpoint_count: usize,
    pub phase: String,
    pub early_stop_recommended: bool,
    pub best_checkpoint_index: usize,
    pub best_checkpoint_source: String,
    pub peak_effective_rank: f64,
    pub final_effective_rank: f64,
    pub peak_velocity: f64,
    pub final_velocity: f64,
    pub recommendation: String,
    pub points: Vec<CheckpointPoint>,
    pub transitions: Vec<TrajectoryTransition>,
}

fn json_esc(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c => out.push(c),
        }
    }
    out
}

/// Analyze an ordered list of >= 2 `.safetensors` LoRA checkpoints along a training run.
pub fn analyze_trajectory(sources: &[String]) -> Result<TrajectoryReport, String> {
    if sources.len() < 2 {
        return Err("Trajectory analysis requires at least 2 ordered checkpoints.".to_string());
    }

    let mut loaded = Vec::with_capacity(sources.len());
    let mut points = Vec::with_capacity(sources.len());

    for (idx, src) in sources.iter().enumerate() {
        let path = resolve_source_path(src)?;
        let tensors = read_safetensors(&path)?;
        let layers = extract_lora_layers(&tensors)?;
        let alphas = extract_lora_alphas(&tensors);
        let dora = extract_dora_scales(&tensors);
        let audit = audit_adapter(&layers);

        let n = audit.layers.len().max(1) as f64;
        let mean_eff = audit.layers.iter().map(|l| l.effective_rank).sum::<f64>() / n;
        let mean_util = audit.layers.iter().map(|l| l.utilization_ratio).sum::<f64>() / n;
        let total_frob = audit
            .layers
            .iter()
            .map(|l| l.frobenius_norm * l.frobenius_norm)
            .sum::<f64>()
            .sqrt();
        let collapsed = audit
            .layers
            .iter()
            .filter(|l| matches!(l.health, LayerHealth::RankCollapsed | LayerHealth::DeadLayer))
            .count();
        let collapse_ratio = (collapsed as f64) / n;
        let health_score = ((mean_util * 80.0) + (1.0 - collapse_ratio) * 20.0).clamp(0.0, 100.0);

        points.push(CheckpointPoint {
            index: idx,
            source: src.clone(),
            layer_count: layers.len(),
            mean_effective_rank: mean_eff,
            mean_utilization: mean_util,
            total_frobenius_norm: total_frob,
            collapsed_layers: collapsed,
            health_score,
        });
        loaded.push((src.clone(), layers, alphas, dora));
    }

    let mut transitions = Vec::with_capacity(loaded.len() - 1);
    for i in 1..loaded.len() {
        let (src_a, la, aa, da) = &loaded[i - 1];
        let (src_b, lb, ab, db) = &loaded[i];
        let diff = diff_adapters_with_dora(la, aa, da, lb, ab, db);
        let velocity = if diff.shared_layers.is_empty() {
            0.0
        } else {
            diff.shared_layers
                .iter()
                .map(|m| m.frobenius_delta * m.frobenius_delta)
                .sum::<f64>()
                .sqrt()
        };
        let overlap = diff.mean_subspace_similarity;
        let er_delta = points[i].mean_effective_rank - points[i - 1].mean_effective_rank;
        transitions.push(TrajectoryTransition {
            from_index: i - 1,
            to_index: i,
            from_source: src_a.clone(),
            to_source: src_b.clone(),
            frobenius_velocity: velocity,
            mean_subspace_overlap: overlap,
            effective_rank_delta: er_delta,
        });
    }

    let peak_effective_rank = points
        .iter()
        .map(|p| p.mean_effective_rank)
        .fold(0.0_f64, f64::max);
    let final_effective_rank = points.last().map(|p| p.mean_effective_rank).unwrap_or(0.0);
    let peak_velocity = transitions
        .iter()
        .map(|t| t.frobenius_velocity)
        .fold(0.0_f64, f64::max);
    let last_t = transitions.last().unwrap();
    let final_velocity = last_t.frobenius_velocity;

    let best_idx = points
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| {
            a.health_score
                .partial_cmp(&b.health_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(i, _)| i)
        .unwrap_or(0);

    let rank_drop_ratio = if peak_effective_rank > 1e-9 {
        (peak_effective_rank - final_effective_rank) / peak_effective_rank
    } else {
        0.0
    };

    let (phase, early_stop_recommended, recommendation) = if rank_drop_ratio > 0.15 {
        (
            "rank-collapsing".to_string(),
            true,
            format!(
                "Effective rank dropped {:.1}% from peak ({:.2} -> {:.2}). Revert to checkpoint #{} ({}) or lower learning rate.",
                rank_drop_ratio * 100.0,
                peak_effective_rank,
                final_effective_rank,
                best_idx,
                points[best_idx].source
            ),
        )
    } else if last_t.mean_subspace_overlap >= 0.96
        && (peak_velocity <= 1e-9 || final_velocity <= peak_velocity * 0.40)
    {
        (
            "converged-plateau".to_string(),
            true,
            format!(
                "Subspace overlap reached {:.3} and Frobenius velocity dropped to {:.4} (<=40% of peak {:.4}). Safe to early-stop to avoid memorization.",
                last_t.mean_subspace_overlap, final_velocity, peak_velocity
            ),
        )
    } else if transitions.len() >= 2
        && last_t.mean_subspace_overlap < 0.70
        && final_velocity > transitions[transitions.len() - 2].frobenius_velocity * 1.35
    {
        (
            "oscillating-or-unstable".to_string(),
            false,
            "Late-stage subspace overlap dropped below 0.70 while velocity surged. Reduce LR or enable cosine decay / LoRA+ ratio tuning.".to_string(),
        )
    } else {
        (
            "actively-learning".to_string(),
            false,
            format!(
                "Subspace is still adapting smoothly (overlap={:.3}, velocity={:.4}). Continue training or evaluate checkpoint #{} ({}).",
                last_t.mean_subspace_overlap, final_velocity, best_idx, points[best_idx].source
            ),
        )
    };

    Ok(TrajectoryReport {
        checkpoint_count: points.len(),
        phase,
        early_stop_recommended,
        best_checkpoint_index: best_idx,
        best_checkpoint_source: points[best_idx].source.clone(),
        peak_effective_rank,
        final_effective_rank,
        peak_velocity,
        final_velocity,
        recommendation,
        points,
        transitions,
    })
}

pub fn render_trajectory_json(rep: &TrajectoryReport) -> String {
    let pts: Vec<String> = rep
        .points
        .iter()
        .map(|p| {
            format!(
                r#"{{"index":{},"source":"{}","layer_count":{},"mean_effective_rank":{:.4},"mean_utilization":{:.4},"total_frobenius_norm":{:.6},"collapsed_layers":{},"health_score":{:.2}}}"#,
                p.index,
                json_esc(&p.source),
                p.layer_count,
                p.mean_effective_rank,
                p.mean_utilization,
                p.total_frobenius_norm,
                p.collapsed_layers,
                p.health_score
            )
        })
        .collect();
    let trs: Vec<String> = rep
        .transitions
        .iter()
        .map(|t| {
            format!(
                r#"{{"from_index":{},"to_index":{},"from_source":"{}","to_source":"{}","frobenius_velocity":{:.6},"mean_subspace_overlap":{:.4},"effective_rank_delta":{:.4}}}"#,
                t.from_index,
                t.to_index,
                json_esc(&t.from_source),
                json_esc(&t.to_source),
                t.frobenius_velocity,
                t.mean_subspace_overlap,
                t.effective_rank_delta
            )
        })
        .collect();
    format!(
        r#"{{"checkpoint_count":{},"phase":"{}","early_stop_recommended":{},"best_checkpoint_index":{},"best_checkpoint_source":"{}","peak_effective_rank":{:.4},"final_effective_rank":{:.4},"peak_velocity":{:.6},"final_velocity":{:.6},"recommendation":"{}","points":[{}],"transitions":[{}]}}"#,
        rep.checkpoint_count,
        json_esc(&rep.phase),
        if rep.early_stop_recommended { "true" } else { "false" },
        rep.best_checkpoint_index,
        json_esc(&rep.best_checkpoint_source),
        rep.peak_effective_rank,
        rep.final_effective_rank,
        rep.peak_velocity,
        rep.final_velocity,
        json_esc(&rep.recommendation),
        pts.join(","),
        trs.join(",")
    )
}

pub fn render_trajectory_table(rep: &TrajectoryReport) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "=== LoRASpectra Training Trajectory ({} checkpoints) ===\n",
        rep.checkpoint_count
    ));
    out.push_str(&format!(
        "Phase: {} | Early-Stop Recommended: {} | Best: #{} ({})\n",
        rep.phase, rep.early_stop_recommended, rep.best_checkpoint_index, rep.best_checkpoint_source
    ));
    out.push_str(&format!("Recommendation: {}\n\n", rep.recommendation));
    out.push_str("Idx | EffRank | Util%  | ||dW||_F | Collapsed | Health | Checkpoint\n");
    out.push_str("----+---------+--------+----------+-----------+--------+-----------\n");
    for p in &rep.points {
        out.push_str(&format!(
            "{:>3} | {:>7.2} | {:>5.1}% | {:>8.4} | {:>9} | {:>6.1} | {}\n",
            p.index,
            p.mean_effective_rank,
            p.mean_utilization * 100.0,
            p.total_frobenius_norm,
            p.collapsed_layers,
            p.health_score,
            p.source
        ));
    }
    out.push_str("\nTransitions:\n");
    for t in &rep.transitions {
        out.push_str(&format!(
            "  #{} -> #{}: velocity={:.5}, subspace_overlap={:.4}, delta_eff_rank={:+.3}\n",
            t.from_index, t.to_index, t.frobenius_velocity, t.mean_subspace_overlap, t.effective_rank_delta
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::safetensors::{layers_to_tensor_map, write_safetensors, DType, LoraLayer, Tensor2D};
    use crate::Rng64;

    fn make_ckpt(path: &std::path::Path, base: &[f64], noise_scale: f64, seed: u64) {
        let mut rng = Rng64::new(seed);
        let a_data: Vec<f64> = base.iter().map(|v| v + rng.next_gaussian() * noise_scale).collect();
        let b_data: Vec<f64> = base.iter().map(|v| v * 0.8 + rng.next_gaussian() * noise_scale).collect();
        let layer = LoraLayer {
            base_name: "model.layers.0.self_attn.q_proj".to_string(),
            a_key: "model.layers.0.self_attn.q_proj.lora_A.weight".to_string(),
            b_key: "model.layers.0.self_attn.q_proj.lora_B.weight".to_string(),
            a: Tensor2D { rows: 4, cols: 16, dtype: DType::F32, data: a_data },
            b: Tensor2D { rows: 16, cols: 4, dtype: DType::F32, data: b_data },
        };
        write_safetensors(path, &layers_to_tensor_map(&[layer])).unwrap();
    }

    #[test]
    fn detects_converged_plateau_across_three_checkpoints() {
        let dir = std::env::temp_dir().join("loraspectra_traj_test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        let mut rng = Rng64::new(77);
        let base: Vec<f64> = (0..64).map(|_| rng.next_gaussian() * 0.35).collect();
        let p1 = dir.join("step_100.safetensors");
        let p2 = dir.join("step_200.safetensors");
        let p3 = dir.join("step_300.safetensors");

        // Big jump from step_100 -> step_200, then near-identical plateau step_200 -> step_300
        let start_vec: Vec<f64> = base.iter().map(|v| v * 0.15).collect();
        make_ckpt(&p1, &start_vec, 0.05, 1);
        make_ckpt(&p2, &base, 0.002, 2);
        make_ckpt(&p3, &base, 0.001, 3);

        let sources = vec![
            p1.to_string_lossy().to_string(),
            p2.to_string_lossy().to_string(),
            p3.to_string_lossy().to_string(),
        ];
        let rep = analyze_trajectory(&sources).unwrap();
        assert_eq!(rep.checkpoint_count, 3);
        assert_eq!(rep.transitions.len(), 2);
        assert_eq!(rep.phase, "converged-plateau");
        assert!(rep.early_stop_recommended);
        assert!(render_trajectory_json(&rep).contains("\"converged-plateau\""));
        let _ = std::fs::remove_dir_all(&dir);
    }
}

//! AdaLoRA / Spectral Sensitivity Rank-Budget Schedule Optimizer (`LS049`-`LS050`).
//!
//! Analyzes per-layer Frobenius spectral importance and tail energy concentration to emit an
//! optimal heterogeneous PEFT `rank_pattern` schedule under a target average rank budget, while
//! flagging capacity-starved bottleneck layers (`LS049`) and severely over-allocated waste layers (`LS050`).

use crate::audit::audit_layer;
use crate::safetensors::LoraLayer;
use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RankScheduleStatus {
    Bottleneck,
    Balanced,
    OverAllocated,
}

impl RankScheduleStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            RankScheduleStatus::Bottleneck => "BOTTLENECK",
            RankScheduleStatus::Balanced => "BALANCED",
            RankScheduleStatus::OverAllocated => "OVER_ALLOCATED",
        }
    }
}

#[derive(Debug, Clone)]
pub struct RankScheduleLayer {
    pub base_name: String,
    pub current_rank: usize,
    pub recommended_rank: usize,
    pub rank_99_energy: usize,
    pub effective_rank: f64,
    pub frobenius_energy: f64,
    pub tail_energy_share: f64,
    pub status: RankScheduleStatus,
}

#[derive(Debug, Clone)]
pub struct RankScheduleDiagnostic {
    pub rule_id: &'static str,
    pub level: &'static str,
    pub layer_name: String,
    pub message: String,
    pub current_rank: usize,
    pub recommended_rank: usize,
}

#[derive(Debug, Clone)]
pub struct RankScheduleSummary {
    pub source_uri: String,
    pub target_avg_rank: f64,
    pub min_rank: usize,
    pub max_rank: usize,
    pub current_total_rank: usize,
    pub recommended_total_rank: usize,
    pub bottleneck_count: usize,
    pub over_allocated_count: usize,
    pub rank_pattern: BTreeMap<String, usize>,
    pub layers: Vec<RankScheduleLayer>,
    pub diagnostics: Vec<RankScheduleDiagnostic>,
}

/// Optimize per-layer AdaLoRA / PEFT rank allocation via spectral importance water-filling.
pub fn optimize_rank_schedule(
    layers: &[LoraLayer],
    source_uri: &str,
    target_avg_rank: f64,
    min_rank: usize,
    max_rank: usize,
) -> RankScheduleSummary {
    let min_r = min_rank.max(1);
    let max_r = max_rank.max(min_r);
    let n = layers.len();
    let total_budget = ((target_avg_rank.max(min_r as f64)) * (n as f64)).round() as usize;

    let mut audited = Vec::with_capacity(n);
    for layer in layers {
        let rep = audit_layer(layer);
        let energy = rep.frobenius_norm * rep.frobenius_norm;
        // Fraction of energy sitting in the bottom 25% of singular modes (bottleneck indicator)
        let r = rep.singular_values.len();
        let tail_start = (r * 3) / 4;
        let tail_energy: f64 = rep
            .singular_values
            .iter()
            .skip(tail_start)
            .map(|s| s * s)
            .sum();
        let tail_share = if energy > 1e-12 { tail_energy / energy } else { 0.0 };
        audited.push((layer, rep, energy, tail_share));
    }

    // Importance weight per layer: combines effective rank demand and log-energy weight.
    let weights: Vec<f64> = audited
        .iter()
        .map(|(_, rep, energy, tail_share)| {
            let demand = (rep.rank_at_99_energy as f64).max(rep.effective_rank).max(1.0);
            let boost = 1.0 + 2.0 * tail_share;
            demand * boost * (1.0 + energy.sqrt()).ln_1p().max(0.25)
        })
        .collect();
    let weight_sum: f64 = weights.iter().sum::<f64>().max(1e-12);

    let mut allocated: Vec<usize> = weights
        .iter()
        .map(|w| {
            let raw = ((*w / weight_sum) * (total_budget as f64)).round() as usize;
            raw.clamp(min_r, max_r)
        })
        .collect();

    // Adjust greedily to match target total budget closely within [min_r, max_r]
    let mut cur_sum: usize = allocated.iter().sum();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| weights[b].partial_cmp(&weights[a]).unwrap_or(std::cmp::Ordering::Equal));
    while cur_sum < total_budget {
        let mut changed = false;
        for &idx in &order {
            if cur_sum >= total_budget {
                break;
            }
            if allocated[idx] < max_r {
                allocated[idx] += 1;
                cur_sum += 1;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    while cur_sum > total_budget {
        let mut changed = false;
        for &idx in order.iter().rev() {
            if cur_sum <= total_budget {
                break;
            }
            if allocated[idx] > min_r {
                allocated[idx] -= 1;
                cur_sum -= 1;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }

    let mut schedule_layers = Vec::with_capacity(n);
    let mut diagnostics = Vec::new();
    let mut rank_pattern = BTreeMap::new();

    for (idx, (layer, rep, energy, tail_share)) in audited.into_iter().enumerate() {
        let rec_r = allocated[idx];
        let cur_r = layer.rank();
        let util = if cur_r > 0 {
            rep.rank_at_99_energy as f64 / cur_r as f64
        } else {
            0.0
        };
        let status = if util >= 0.90 && tail_share >= 0.05 {
            RankScheduleStatus::Bottleneck
        } else if util <= 0.45 && cur_r >= 8 {
            RankScheduleStatus::OverAllocated
        } else {
            RankScheduleStatus::Balanced
        };

        if status == RankScheduleStatus::Bottleneck {
            diagnostics.push(RankScheduleDiagnostic {
                rule_id: "LS049",
                level: "warning",
                layer_name: layer.base_name.clone(),
                message: format!(
                    "Layer '{}' is capacity-starved (k_99%={}/{}, bottom-quartile tail energy {:.1}%); increase rank {} -> {}",
                    layer.base_name, rep.rank_at_99_energy, cur_r, tail_share * 100.0, cur_r, rec_r.max(cur_r)
                ),
                current_rank: cur_r,
                recommended_rank: rec_r,
            });
        } else if status == RankScheduleStatus::OverAllocated {
            diagnostics.push(RankScheduleDiagnostic {
                rule_id: "LS050",
                level: "note",
                layer_name: layer.base_name.clone(),
                message: format!(
                    "Layer '{}' is over-allocated (k_99%={}/{}); reallocate rank {} -> {} to bottleneck modules",
                    layer.base_name, rep.rank_at_99_energy, cur_r, cur_r, rec_r
                ),
                current_rank: cur_r,
                recommended_rank: rec_r,
            });
        }

        rank_pattern.insert(layer.base_name.clone(), rec_r);
        schedule_layers.push(RankScheduleLayer {
            base_name: layer.base_name.clone(),
            current_rank: cur_r,
            recommended_rank: rec_r,
            rank_99_energy: rep.rank_at_99_energy,
            effective_rank: rep.effective_rank,
            frobenius_energy: energy,
            tail_energy_share: tail_share,
            status,
        });
    }

    let current_total_rank = schedule_layers.iter().map(|l| l.current_rank).sum();
    let recommended_total_rank = schedule_layers.iter().map(|l| l.recommended_rank).sum();
    let bottleneck_count = schedule_layers
        .iter()
        .filter(|l| l.status == RankScheduleStatus::Bottleneck)
        .count();
    let over_allocated_count = schedule_layers
        .iter()
        .filter(|l| l.status == RankScheduleStatus::OverAllocated)
        .count();

    RankScheduleSummary {
        source_uri: source_uri.to_string(),
        target_avg_rank,
        min_rank: min_r,
        max_rank: max_r,
        current_total_rank,
        recommended_total_rank,
        bottleneck_count,
        over_allocated_count,
        rank_pattern,
        layers: schedule_layers,
        diagnostics,
    }
}

pub fn render_rank_schedule_table(summary: &RankScheduleSummary) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "{:<44} {:>5} {:>5} {:>5} {:>7} {:>8} {:<15}\n",
        "LAYER", "CUR_R", "REC_R", "K_99%", "R_EFF", "TAIL_E%", "STATUS"
    ));
    out.push_str(&"-".repeat(96));
    out.push('\n');
    for l in &summary.layers {
        out.push_str(&format!(
            "{:<44} {:>5} {:>5} {:>5} {:>7.2} {:>7.1}% {:<15}\n",
            l.base_name,
            l.current_rank,
            l.recommended_rank,
            l.rank_99_energy,
            l.effective_rank,
            l.tail_energy_share * 100.0,
            l.status.as_str()
        ));
    }
    out.push_str(&"-".repeat(96));
    out.push('\n');
    out.push_str(&format!(
        "Summary: {} layers | Total rank {} -> {} (target avg {:.1}) | Bottlenecks (LS049): {} | Over-allocated (LS050): {}\n",
        summary.layers.len(),
        summary.current_total_rank,
        summary.recommended_total_rank,
        summary.target_avg_rank,
        summary.bottleneck_count,
        summary.over_allocated_count
    ));
    out
}

pub fn render_rank_schedule_json(summary: &RankScheduleSummary) -> String {
    let pat_entries: Vec<String> = summary
        .rank_pattern
        .iter()
        .map(|(k, v)| format!("    \"{}\": {}", k.replace('"', "\\\""), v))
        .collect();
    format!(
        "{{\n  \"source\": \"{}\",\n  \"target_avg_rank\": {:.2},\n  \"current_total_rank\": {},\n  \"recommended_total_rank\": {},\n  \"bottleneck_count\": {},\n  \"over_allocated_count\": {},\n  \"rank_pattern\": {{\n{}\n  }}\n}}\n",
        summary.source_uri.replace('"', "\\\""),
        summary.target_avg_rank,
        summary.current_total_rank,
        summary.recommended_total_rank,
        summary.bottleneck_count,
        summary.over_allocated_count,
        pat_entries.join(",\n")
    )
}

fn html_esc(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

/// Render a self-contained dark-mode HTML AdaLoRA Water-Filling Rank Schedule Report.
pub fn render_rank_schedule_html(summary: &RankScheduleSummary) -> String {
    let mut rows = String::new();
    for l in &summary.layers {
        let cls = match l.status {
            RankScheduleStatus::Bottleneck => "bad",
            RankScheduleStatus::OverAllocated => "warn",
            _ => "ok",
        };
        let delta: isize = l.recommended_rank as isize - l.current_rank as isize;
        let delta_str = if delta > 0 {
            format!("+{}", delta)
        } else {
            format!("{}", delta)
        };
        rows.push_str(&format!(
            "<tr><td><code>{}</code></td><td>{}</td><td><b>{}</b> ({})</td><td>{}</td><td>{:.2}</td><td>{:.1}%</td><td><span class=\"pill {}\">{}</span></td></tr>\n",
            html_esc(&l.base_name),
            l.current_rank,
            l.recommended_rank,
            delta_str,
            l.rank_99_energy,
            l.effective_rank,
            l.tail_energy_share * 100.0,
            cls,
            html_esc(l.status.as_str()),
        ));
    }
    format!(
        r##"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>loraspectra AdaLoRA Rank-Schedule Report</title>
<style>
:root{{--bg:#0d1117;--card:#161b22;--bd:#30363d;--tx:#e6edf3;--mu:#8b949e;--ac:#58a6ff;--ok:#3fb950;--wa:#d29922;--ba:#f85149}}
body{{margin:0;padding:24px;background:var(--bg);color:var(--tx);font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif}}
.wrap{{max-width:1160px;margin:0 auto}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(175px,1fr));gap:12px;margin:16px 0 20px}}
.card{{background:var(--card);border:1px solid var(--bd);border-radius:10px;padding:14px}}
.k{{font-size:12px;color:var(--mu);text-transform:uppercase;letter-spacing:.04em}}
.v{{font-size:22px;font-weight:700;margin-top:4px}}
table{{width:100%;border-collapse:collapse;background:var(--card);border:1px solid var(--bd);border-radius:10px;overflow:hidden;font-size:13px}}
th,td{{padding:9px 10px;border-bottom:1px solid #21262d;text-align:left}}
th{{color:var(--mu);font-weight:600;background:#11161d}}
.pill{{padding:2px 8px;border-radius:999px;font-size:11px;font-weight:700}}
.pill.ok{{background:rgba(63,185,80,.15);color:var(--ok)}}
.pill.warn{{background:rgba(210,153,34,.15);color:var(--wa)}}
.pill.bad{{background:rgba(248,81,73,.15);color:var(--ba)}}
code{{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:12px}}
</style>
</head>
<body>
<div class="wrap">
  <h1 style="margin:0;font-size:22px">loraspectra AdaLoRA Water-Filling Rank Schedule</h1>
  <div style="color:var(--mu);font-size:13px;margin-top:4px">Source: <code>{source}</code> &middot; Layers: {layers} &middot; Target Avg Rank: {avg:.1}</div>
  <div class="grid">
    <div class="card"><div class="k">Current Total Rank</div><div class="v">{cur}</div></div>
    <div class="card"><div class="k">Recommended Total Rank</div><div class="v">{rec}</div></div>
    <div class="card"><div class="k">Bottlenecks (LS049)</div><div class="v">{bot}</div></div>
    <div class="card"><div class="k">Over-Allocated (LS050)</div><div class="v">{over}</div></div>
    <div class="card"><div class="k">Rank Bounds</div><div class="v">[{min_r}, {max_r}]</div></div>
  </div>
  <table>
    <thead><tr><th>Layer</th><th>Current R</th><th>Recommended R</th><th>K_99%</th><th>R_eff</th><th>Tail Energy</th><th>Status</th></tr></thead>
    <tbody>
{rows}    </tbody>
  </table>
</div>
</body>
</html>
"##,
        source = html_esc(&summary.source_uri),
        layers = summary.layers.len(),
        avg = summary.target_avg_rank,
        cur = summary.current_total_rank,
        rec = summary.recommended_total_rank,
        bot = summary.bottleneck_count,
        over = summary.over_allocated_count,
        min_r = summary.min_rank,
        max_r = summary.max_rank,
        rows = rows,
    )
}

/// Render a shields.io Endpoint Badge JSON summarizing AdaLoRA water-filling rank allocation.
pub fn render_rank_schedule_badge_json(summary: &RankScheduleSummary) -> String {
    let color = if summary.bottleneck_count == 0 {
        "brightgreen"
    } else if summary.bottleneck_count <= 2 {
        "yellow"
    } else {
        "red"
    };
    format!(
        r#"{{"schemaVersion":1,"label":"loraspectra adalora","message":"rank {}->{} ({} bottlenecks)","color":"{}"}}"#,
        summary.current_total_rank,
        summary.recommended_total_rank,
        summary.bottleneck_count,
        color
    )
}

#[cfg(test)]
mod html_tests {
    use super::*;
    use crate::safetensors::{DType, Tensor2D};

    #[test]
    fn rank_schedule_html_renders_summary_and_layers() {
        let layer = LoraLayer {
            base_name: "model.layers.4.mlp.down_proj".to_string(),
            a_key: "model.layers.4.mlp.down_proj.lora_A.weight".to_string(),
            b_key: "model.layers.4.mlp.down_proj.lora_B.weight".to_string(),
            a: Tensor2D { rows: 4, cols: 8, dtype: DType::F32, data: vec![0.2; 32] },
            b: Tensor2D { rows: 8, cols: 4, dtype: DType::F32, data: vec![0.2; 32] },
        };
        let summary = optimize_rank_schedule(&[layer], "demo.safetensors", 8.0, 4, 32);
        let html = render_rank_schedule_html(&summary);
        assert!(html.contains("<!DOCTYPE html>"));
        assert!(html.contains("model.layers.4.mlp.down_proj"));
        assert!(html.contains("AdaLoRA Water-Filling Rank Schedule"));
        let badge = render_rank_schedule_badge_json(&summary);
        assert!(badge.contains("\"schemaVersion\":1"));
        assert!(badge.contains("loraspectra adalora"));
    }
}

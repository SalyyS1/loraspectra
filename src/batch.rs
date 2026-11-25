//! Recursive Multi-Checkpoint LoRA Spectral Batch Auditor & Leaderboard (`loraspectra batch-audit`).
//!
//! Discovers `.safetensors` LoRA checkpoints across one or more directories or file paths,
//! evaluates spectral SVD utilization (`audit_adapter`) and LoRA+ variational factor balance
//! (`analyze_loraplus`), and ranks checkpoints with a deterministic 0-100 Spectral Health Score.

use std::fs;
use std::path::{Path, PathBuf};

use crate::audit::{audit_adapter, LayerHealth};
use crate::loraplus::analyze_loraplus;
use crate::safetensors::{
    extract_lora_alphas, extract_lora_layers, read_peft_adapter_config, read_safetensors,
};

#[derive(Debug, Clone)]
pub struct BatchCheckpointEntry {
    pub path: String,
    pub layers_audited: usize,
    pub nominal_rank_avg: f64,
    pub effective_rank_avg: f64,
    pub utilization_avg: f64,
    pub dead_layers: usize,
    pub collapsed_layers: usize,
    pub over_param_layers: usize,
    pub loraplus_mean_imbalance: f64,
    pub loraplus_imbalanced_layers: usize,
    pub health_score: f64,
    pub status: String,
    pub error: Option<String>,
}

#[derive(Debug, Clone)]
pub struct BatchAuditReport {
    pub total_discovered: usize,
    pub audited_count: usize,
    pub error_count: usize,
    pub mean_health_score: f64,
    pub entries: Vec<BatchCheckpointEntry>,
}

/// Recursively discover `.safetensors` files under the supplied file/directory paths.
pub fn discover_checkpoints(inputs: &[PathBuf], max_depth: usize) -> Vec<PathBuf> {
    let mut found = Vec::new();
    for input in inputs {
        if input.is_file() {
            if is_safetensors(input) {
                found.push(input.clone());
            }
        } else if input.is_dir() {
            walk_dir(input, 0, max_depth, &mut found);
        }
    }
    found.sort();
    found.dedup();
    found
}

fn is_safetensors(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|e| e.eq_ignore_ascii_case("safetensors"))
        .unwrap_or(false)
}

fn walk_dir(dir: &Path, depth: usize, max_depth: usize, out: &mut Vec<PathBuf>) {
    if depth > max_depth {
        return;
    }
    let Ok(rd) = fs::read_dir(dir) else {
        return;
    };
    for entry in rd.flatten() {
        let p = entry.path();
        if p.is_dir() {
            if let Some(name) = p.file_name().and_then(|n| n.to_str()) {
                if name.starts_with('.') || name == "target" || name == "node_modules" {
                    continue;
                }
            }
            walk_dir(&p, depth + 1, max_depth, out);
        } else if p.is_file() && is_safetensors(&p) {
            out.push(p);
        }
    }
}

/// Audit a list of checkpoint paths and return a leaderboard sorted by `health_score` descending.
pub fn run_batch_audit(paths: &[PathBuf]) -> BatchAuditReport {
    let mut entries = Vec::with_capacity(paths.len());
    for path in paths {
        entries.push(audit_single_checkpoint(path));
    }

    entries.sort_by(|a, b| {
        b.health_score
            .partial_cmp(&a.health_score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.path.cmp(&b.path))
    });

    let audited: Vec<&BatchCheckpointEntry> = entries.iter().filter(|e| e.error.is_none()).collect();
    let mean_health_score = if audited.is_empty() {
        0.0
    } else {
        audited.iter().map(|e| e.health_score).sum::<f64>() / (audited.len() as f64)
    };

    BatchAuditReport {
        total_discovered: paths.len(),
        audited_count: audited.len(),
        error_count: entries.len() - audited.len(),
        mean_health_score,
        entries,
    }
}

fn audit_single_checkpoint(path: &Path) -> BatchCheckpointEntry {
    let path_str = path.to_string_lossy().to_string();
    let tensors = match read_safetensors(path) {
        Ok(t) => t,
        Err(e) => {
            return BatchCheckpointEntry {
                path: path_str,
                layers_audited: 0,
                nominal_rank_avg: 0.0,
                effective_rank_avg: 0.0,
                utilization_avg: 0.0,
                dead_layers: 0,
                collapsed_layers: 0,
                over_param_layers: 0,
                loraplus_mean_imbalance: 0.0,
                loraplus_imbalanced_layers: 0,
                health_score: 0.0,
                status: "ERROR".to_string(),
                error: Some(format!("read_safetensors failed: {}", e)),
            };
        }
    };

    let layers = match extract_lora_layers(&tensors) {
        Ok(l) => l,
        Err(e) => {
            return BatchCheckpointEntry {
                path: path_str,
                layers_audited: 0,
                nominal_rank_avg: 0.0,
                effective_rank_avg: 0.0,
                utilization_avg: 0.0,
                dead_layers: 0,
                collapsed_layers: 0,
                over_param_layers: 0,
                loraplus_mean_imbalance: 0.0,
                loraplus_imbalanced_layers: 0,
                health_score: 0.0,
                status: "ERROR".to_string(),
                error: Some(format!("extract_lora_layers failed: {}", e)),
            };
        }
    };
    if layers.is_empty() {
        return BatchCheckpointEntry {
            path: path_str,
            layers_audited: 0,
            nominal_rank_avg: 0.0,
            effective_rank_avg: 0.0,
            utilization_avg: 0.0,
            dead_layers: 0,
            collapsed_layers: 0,
            over_param_layers: 0,
            loraplus_mean_imbalance: 0.0,
            loraplus_imbalanced_layers: 0,
            health_score: 0.0,
            status: "NO_LORA_LAYERS".to_string(),
            error: Some("No LoRA A/B tensor pairs found in checkpoint".to_string()),
        };
    }

    let sibling_cfg = path
        .parent()
        .map(|p| p.join("adapter_config.json"))
        .filter(|p| p.is_file())
        .and_then(|p| read_peft_adapter_config(&p).ok());
    let _alphas = extract_lora_alphas(&tensors);
    let _ = sibling_cfg;

    let audit = audit_adapter(&layers);
    let lp = analyze_loraplus(&layers, &path_str, 1.5, 50.0);

    let n = audit.layers.len().max(1) as f64;
    let nominal_rank_avg = audit.layers.iter().map(|l| l.nominal_rank as f64).sum::<f64>() / n;
    let effective_rank_avg = audit.layers.iter().map(|l| l.effective_rank).sum::<f64>() / n;
    let utilization_avg = audit.layers.iter().map(|l| l.utilization_ratio).sum::<f64>() / n;
    let dead_layers = audit
        .layers
        .iter()
        .filter(|l| l.health == LayerHealth::DeadLayer)
        .count();
    let collapsed_layers = audit
        .layers
        .iter()
        .filter(|l| l.health == LayerHealth::RankCollapsed)
        .count();
    let over_param_layers = audit
        .layers
        .iter()
        .filter(|l| l.health == LayerHealth::OverParameterized)
        .count();

    // Composite 0..100 score:
    // - Up to 55 pts from rank utilization (clamped [0, 1])
    // - Up to 25 pts from absence of dead/collapsed layers
    // - Up to 20 pts from LoRA+ factor balance (imbalance 1.0 => full 20 pts, >=3.0 => 0 pts)
    let util_pts = utilization_avg.clamp(0.0, 1.0) * 55.0;
    let bad_ratio = ((dead_layers + collapsed_layers) as f64) / n;
    let integrity_pts = (1.0 - bad_ratio).clamp(0.0, 1.0) * 25.0;
    let excess_imb = (lp.mean_imbalance_index - 1.0_f64).max(0.0);
    let balance_pts = (1.0_f64 - (excess_imb / 2.0)).clamp(0.0, 1.0) * 20.0;
    let health_score = (util_pts + integrity_pts + balance_pts).clamp(0.0, 100.0);

    let status = if dead_layers > 0 {
        "CRITICAL"
    } else if collapsed_layers > 0 || health_score < 55.0 {
        "WARN"
    } else if health_score >= 80.0 {
        "OPTIMAL"
    } else {
        "OK"
    };

    BatchCheckpointEntry {
        path: path_str,
        layers_audited: audit.layers.len(),
        nominal_rank_avg,
        effective_rank_avg,
        utilization_avg,
        dead_layers,
        collapsed_layers,
        over_param_layers,
        loraplus_mean_imbalance: lp.mean_imbalance_index,
        loraplus_imbalanced_layers: lp.imbalanced_layers,
        health_score,
        status: status.to_string(),
        error: None,
    }
}

pub fn render_batch_table(report: &BatchAuditReport) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "LoRASpectra Batch Audit Leaderboard ({} discovered, {} audited, {} errors, mean score {:.1}/100)\n",
        report.total_discovered, report.audited_count, report.error_count, report.mean_health_score
    ));
    out.push_str(&format!(
        "{:<4} {:<8} {:>6} {:>7} {:>8} {:>6} {:>6} {:>8}  {}\n",
        "#", "STATUS", "SCORE", "LAYERS", "UTIL%", "DEAD", "COLL", "IMB", "CHECKPOINT"
    ));
    out.push_str(&"-".repeat(92));
    out.push('\n');
    for (idx, e) in report.entries.iter().enumerate() {
        out.push_str(&format!(
            "{:<4} {:<8} {:>6.1} {:>7} {:>7.1}% {:>6} {:>6} {:>8.2}  {}\n",
            idx + 1,
            e.status,
            e.health_score,
            e.layers_audited,
            e.utilization_avg * 100.0,
            e.dead_layers,
            e.collapsed_layers,
            e.loraplus_mean_imbalance,
            e.path
        ));
    }
    out
}

pub fn render_batch_markdown(report: &BatchAuditReport) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "### LoRASpectra Batch Audit Leaderboard ({} checkpoints · mean score {:.1}/100)\n\n",
        report.audited_count, report.mean_health_score
    ));
    out.push_str("| Rank | Status | Score | Layers | Avg Util | Dead | Collapsed | LoRA+ Imbalance | Checkpoint |\n");
    out.push_str("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |\n");
    for (idx, e) in report.entries.iter().enumerate() {
        out.push_str(&format!(
            "| {} | **{}** | {:.1} | {} | {:.1}% | {} | {} | {:.2}x | `{}` |\n",
            idx + 1,
            e.status,
            e.health_score,
            e.layers_audited,
            e.utilization_avg * 100.0,
            e.dead_layers,
            e.collapsed_layers,
            e.loraplus_mean_imbalance,
            e.path
        ));
    }
    out
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

pub fn render_batch_json(report: &BatchAuditReport) -> String {
    let mut entries_json = Vec::with_capacity(report.entries.len());
    for e in &report.entries {
        let err_json = match &e.error {
            Some(msg) => format!("\"{}\"", json_esc(msg)),
            None => "null".to_string(),
        };
        entries_json.push(format!(
            r#"    {{
      "path": "{}",
      "status": "{}",
      "health_score": {:.2},
      "layers_audited": {},
      "nominal_rank_avg": {:.2},
      "effective_rank_avg": {:.2},
      "utilization_avg": {:.4},
      "dead_layers": {},
      "collapsed_layers": {},
      "over_param_layers": {},
      "loraplus_mean_imbalance": {:.4},
      "loraplus_imbalanced_layers": {},
      "error": {}
    }}"#,
            json_esc(&e.path),
            json_esc(&e.status),
            e.health_score,
            e.layers_audited,
            e.nominal_rank_avg,
            e.effective_rank_avg,
            e.utilization_avg,
            e.dead_layers,
            e.collapsed_layers,
            e.over_param_layers,
            e.loraplus_mean_imbalance,
            e.loraplus_imbalanced_layers,
            err_json
        ));
    }
    format!(
        r#"{{
  "total_discovered": {},
  "audited_count": {},
  "error_count": {},
  "mean_health_score": {:.2},
  "entries": [
{}
  ]
}}"#,
        report.total_discovered,
        report.audited_count,
        report.error_count,
        report.mean_health_score,
        entries_json.join(",\n")
    )
}

fn html_esc(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#39;")
}

/// Render a zero-CDN, single-file dark-mode HTML checkpoint leaderboard for ML teams & CI artifacts.
pub fn render_batch_html(report: &BatchAuditReport) -> String {
    let mut rows = String::new();
    for (idx, e) in report.entries.iter().enumerate() {
        let color = if e.health_score >= 85.0 {
            "#3fb950"
        } else if e.health_score >= 65.0 {
            "#d29922"
        } else {
            "#f85149"
        };
        let util_pct = (e.utilization_avg * 100.0).clamp(0.0, 100.0);
        rows.push_str(&format!(
            "<tr><td>#{}</td><td style=\"font-weight:700;color:{}\">{:.1}</td><td><code>{}</code></td><td>{}</td><td>{:.1}%</td><td>{}</td><td>{}</td><td>{:.2}x</td><td style=\"font-family:monospace;font-size:12px\">{}</td></tr>\n",
            idx + 1,
            color,
            e.health_score,
            html_esc(&e.status),
            e.layers_audited,
            util_pct,
            e.dead_layers,
            e.collapsed_layers,
            e.loraplus_mean_imbalance,
            html_esc(&e.path)
        ));
    }
    format!(
        "<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\"><title>loraspectra Batch Audit Leaderboard</title>\
<style>body{{margin:0;padding:28px;background:#0b0f17;color:#e6edf3;font-family:system-ui,-apple-system,sans-serif}}\
.wrap{{max-width:1100px;margin:0 auto}}h1{{margin:0 0 8px}}p.meta{{color:#8b949e;margin:0 0 20px}}\
table{{width:100%;border-collapse:collapse;background:#161b22;border:1px solid #30363d;border-radius:8px;overflow:hidden}}\
th,td{{padding:10px 12px;border-bottom:1px solid #21262d;text-align:left;font-size:13.5px}}th{{background:#111827;color:#8b949e;text-transform:uppercase;font-size:11.5px}}\
</style></head><body><div class=\"wrap\"><h1>loraspectra Spectral Batch Leaderboard</h1>\
<p class=\"meta\">Discovered: {} | Audited: {} | Errors: {} | Mean Health Score: {:.2}/100</p>\
<table><thead><tr><th>Rank</th><th>Score</th><th>Status</th><th>Layers</th><th>Util %</th><th>Dead</th><th>Collapsed</th><th>LoRA+ Imb</th><th>Checkpoint</th></tr></thead>\
<tbody>{}</tbody></table></div></body></html>\n",
        report.total_discovered,
        report.audited_count,
        report.error_count,
        report.mean_health_score,
        rows
    )
}

/// Render an OASIS SARIF v2.1.0 report covering all failing/degraded checkpoints in a batch audit.
pub fn render_batch_sarif(report: &BatchAuditReport, min_score: f64) -> String {
    let mut results = Vec::new();
    for e in &report.entries {
        let uri = json_esc(&e.path);
        if e.dead_layers > 0 {
            results.push(format!(
                r#"        {{ "ruleId": "LS001", "level": "error", "message": {{ "text": "{} dead LoRA layer(s) in checkpoint {} (score {:.1}/100)." }}, "locations": [{{ "physicalLocation": {{ "artifactLocation": {{ "uri": "{}" }} }} }}] }}"#,
                e.dead_layers, uri, e.health_score, uri
            ));
        } else if e.health_score < min_score {
            results.push(format!(
                r#"        {{ "ruleId": "LS047", "level": "warning", "message": {{ "text": "Checkpoint {} spectral health score {:.1}/100 is below threshold {:.1} (status: {}, collapsed: {})." }}, "locations": [{{ "physicalLocation": {{ "artifactLocation": {{ "uri": "{}" }} }} }}] }}"#,
                uri, e.health_score, min_score, json_esc(&e.status), e.collapsed_layers, uri
            ));
        }
    }
    format!(
        r#"{{
  "version": "2.1.0",
  "$schema": "https://json.schemastore.org/sarif-2.1.0.json",
  "runs": [
    {{
      "tool": {{
        "driver": {{
          "name": "loraspectra-batch-audit",
          "semanticVersion": "{}",
          "informationUri": "https://github.com/SalyyS1/loraspectra"
        }}
      }},
      "results": [
{}
      ]
    }}
  ]
}}
"#,
        env!("CARGO_PKG_VERSION"),
        results.join(",\n")
    )
}

/// Render a shields.io Endpoint Badge JSON summarizing a batch checkpoint audit.
pub fn render_batch_badge_json(report: &BatchAuditReport) -> String {
    let any_dead = report.entries.iter().any(|e| e.dead_layers > 0);
    let color = if any_dead || report.mean_health_score < 60.0 {
        "red"
    } else if report.mean_health_score < 80.0 {
        "yellow"
    } else {
        "brightgreen"
    };
    format!(
        "{{\"schemaVersion\":1,\"label\":\"loraspectra batch\",\"message\":\"{:.1}/100 ({} adapters)\",\"color\":\"{}\"}}",
        report.mean_health_score, report.audited_count, color
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::safetensors::{layers_to_tensor_map, write_safetensors, DType, LoraLayer, Tensor2D};
    use crate::Rng64;

    fn make_synthetic_adapter(path: &Path, scale_b: f64, zero_out: bool) {
        let mut rng = Rng64::new(77);
        let r = 4;
        let d_out = 16;
        let d_in = 16;
        let a_data: Vec<f64> = (0..r * d_in)
            .map(|_| if zero_out { 0.0 } else { rng.next_gaussian() * 0.2 })
            .collect();
        let b_data: Vec<f64> = (0..d_out * r)
            .map(|_| if zero_out { 0.0 } else { rng.next_gaussian() * 0.2 * scale_b })
            .collect();
        let layer = LoraLayer {
            base_name: "model.layers.0.self_attn.q_proj".to_string(),
            a_key: "model.layers.0.self_attn.q_proj.lora_A.weight".to_string(),
            b_key: "model.layers.0.self_attn.q_proj.lora_B.weight".to_string(),
            a: Tensor2D { rows: r, cols: d_in, dtype: DType::F32, data: a_data },
            b: Tensor2D { rows: d_out, cols: r, dtype: DType::F32, data: b_data },
        };
        let map = layers_to_tensor_map(&[layer]);
        write_safetensors(path, &map).expect("write synthetic adapter");
    }

    #[test]
    fn batch_audit_discovers_and_ranks_healthy_above_dead_checkpoints() {
        let dir = std::env::temp_dir().join("loraspectra_batch_test_v110");
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(dir.join("sub")).unwrap();
        let good = dir.join("healthy.safetensors");
        let dead = dir.join("sub").join("dead.safetensors");
        make_synthetic_adapter(&good, 1.0, false);
        make_synthetic_adapter(&dead, 1.0, true);

        let paths = discover_checkpoints(&[dir.clone()], 4);
        assert_eq!(paths.len(), 2);

        let report = run_batch_audit(&paths);
        assert_eq!(report.audited_count, 2);
        assert_eq!(report.error_count, 0);
        assert!(report.entries[0].path.ends_with("healthy.safetensors"));
        assert!(report.entries[0].health_score > report.entries[1].health_score);
        assert_eq!(report.entries[1].dead_layers, 1);

        let md = render_batch_markdown(&report);
        assert!(md.contains("healthy.safetensors"));
        let js = render_batch_json(&report);
        assert!(js.contains("\"audited_count\": 2"));
        let html = render_batch_html(&report);
        assert!(html.contains("<!doctype html>"));
        assert!(html.contains("healthy.safetensors"));
        assert!(html.contains("dead.safetensors"));
        let sarif = render_batch_sarif(&report, 80.0);
        assert!(sarif.contains("\"version\": \"2.1.0\""));
        assert!(sarif.contains("LS001"));
        let badge = render_batch_badge_json(&report);
        assert!(badge.contains("\"loraspectra batch\""));
        assert!(badge.contains("2 adapters"));

        let _ = fs::remove_dir_all(&dir);
    }
}

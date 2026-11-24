//! Spectral Layer-Wise Sensitivity & Catastrophic Forgetting Risk Auditor (`LS051`-`LS052`).
//!
//! Decomposes adapter spectral energy across transformer depth (Early / Mid / Deep blocks)
//! and architectural subsystem (Attention `q/k/v/o_proj` vs MLP `gate/up/down_proj` vs Other),
//! computing per-layer Top-1 spectral dominance (`sigma_1^2 / ||Delta W||_F^2`), participation
//! ratio (`r_eff / r`), and depth concentration skew so ML engineers catch overfitting spikes
//! in late MLP layers or brittle rank-1 attention collapse before deployment.

use std::collections::BTreeMap;

use crate::linalg::qr_core_singular_values;
use crate::safetensors::LoraLayer;

#[derive(Debug, Clone)]
pub struct SensitivityLayerMetric {
    pub base_name: String,
    pub subsystem: &'static str,
    pub block_index: Option<usize>,
    pub depth_band: &'static str,
    pub rank: usize,
    pub frobenius_energy: f64,
    pub energy_share_pct: f64,
    pub top1_dominance_pct: f64,
    pub spectral_norm: f64,
    pub risk_flag: Option<&'static str>,
}

#[derive(Debug, Clone)]
pub struct SensitivitySummary {
    pub source_uri: String,
    pub layer_count: usize,
    pub total_frobenius_energy: f64,
    pub attention_energy_pct: f64,
    pub mlp_energy_pct: f64,
    pub other_energy_pct: f64,
    pub early_depth_pct: f64,
    pub mid_depth_pct: f64,
    pub deep_depth_pct: f64,
    pub mean_top1_dominance_pct: f64,
    pub brittle_rank1_layers: usize,
    pub energy_spike_layers: usize,
    pub forgetting_risk_level: &'static str,
    pub layers: Vec<SensitivityLayerMetric>,
}

fn classify_subsystem(name: &str) -> &'static str {
    let lower = name.to_ascii_lowercase();
    if lower.contains("q_proj")
        || lower.contains("k_proj")
        || lower.contains("v_proj")
        || lower.contains("o_proj")
        || lower.contains("attn")
        || lower.contains("to_q")
        || lower.contains("to_k")
        || lower.contains("to_v")
        || lower.contains("to_out")
    {
        "attention"
    } else if lower.contains("gate_proj")
        || lower.contains("up_proj")
        || lower.contains("down_proj")
        || lower.contains("mlp")
        || lower.contains("ffn")
        || lower.contains("fc1")
        || lower.contains("fc2")
    {
        "mlp"
    } else {
        "other"
    }
}

fn extract_block_index(name: &str) -> Option<usize> {
    for part in name.split(['.', '_', '/']) {
        if !part.is_empty() && part.chars().all(|c| c.is_ascii_digit()) {
            if let Ok(idx) = part.parse::<usize>() {
                return Some(idx);
            }
        }
    }
    None
}

pub fn analyze_sensitivity(
    layers: &[LoraLayer],
    source_uri: &str,
    dominance_warn_pct: f64,
    spike_multiplier: f64,
) -> SensitivitySummary {
    let max_idx = layers
        .iter()
        .filter_map(|l| extract_block_index(&l.base_name))
        .max()
        .unwrap_or(0);

    let mut raw_rows: Vec<(String, &'static str, Option<usize>, &'static str, usize, f64, f64, f64)> =
        Vec::with_capacity(layers.len());
    let mut total_energy = 0.0_f64;

    for layer in layers {
        let sv = qr_core_singular_values(&layer.b, &layer.a);
        let energy: f64 = sv.iter().map(|s| s * s).sum();
        let sigma1 = sv.first().copied().unwrap_or(0.0);
        let top1_pct = if energy > 1e-18 {
            ((sigma1 * sigma1) / energy) * 100.0
        } else {
            0.0
        };
        let subsystem = classify_subsystem(&layer.base_name);
        let bidx = extract_block_index(&layer.base_name);
        let depth_band = match bidx {
            Some(idx) if max_idx > 0 => {
                let frac = (idx as f64) / (max_idx as f64);
                if frac < 0.34 {
                    "early"
                } else if frac < 0.67 {
                    "mid"
                } else {
                    "deep"
                }
            }
            _ => "unindexed",
        };
        total_energy += energy;
        raw_rows.push((
            layer.base_name.clone(),
            subsystem,
            bidx,
            depth_band,
            layer.rank(),
            energy,
            top1_pct,
            sigma1,
        ));
    }

    let n = raw_rows.len().max(1) as f64;
    let avg_share = 100.0 / n;
    let mut attn_e = 0.0;
    let mut mlp_e = 0.0;
    let mut other_e = 0.0;
    let mut early_e = 0.0;
    let mut mid_e = 0.0;
    let mut deep_e = 0.0;
    let mut brittle_rank1 = 0usize;
    let mut spike_layers = 0usize;
    let mut sum_top1 = 0.0;

    let mut metrics = Vec::with_capacity(raw_rows.len());
    for (name, sub, bidx, band, rank, energy, top1_pct, sigma1) in raw_rows {
        let share_pct = if total_energy > 1e-18 {
            (energy / total_energy) * 100.0
        } else {
            0.0
        };
        match sub {
            "attention" => attn_e += energy,
            "mlp" => mlp_e += energy,
            _ => other_e += energy,
        }
        match band {
            "early" => early_e += energy,
            "mid" => mid_e += energy,
            "deep" => deep_e += energy,
            _ => {}
        }
        sum_top1 += top1_pct;

        let mut flag: Option<&'static str> = None;
        if rank >= 4 && top1_pct >= dominance_warn_pct {
            brittle_rank1 += 1;
            flag = Some("LS051");
        }
        if layers.len() >= 4 && share_pct >= avg_share * spike_multiplier {
            spike_layers += 1;
            if flag.is_none() {
                flag = Some("LS052");
            }
        }
        metrics.push(SensitivityLayerMetric {
            base_name: name,
            subsystem: sub,
            block_index: bidx,
            depth_band: band,
            rank,
            frobenius_energy: energy,
            energy_share_pct: share_pct,
            top1_dominance_pct: top1_pct,
            spectral_norm: sigma1,
            risk_flag: flag,
        });
    }

    let pct = |x: f64| if total_energy > 1e-18 { (x / total_energy) * 100.0 } else { 0.0 };
    let mean_top1 = sum_top1 / n;
    let risk_level = if brittle_rank1 > 0 && spike_layers > 0 {
        "HIGH"
    } else if brittle_rank1 > 0 || spike_layers > 0 {
        "MEDIUM"
    } else {
        "LOW"
    };

    SensitivitySummary {
        source_uri: source_uri.to_string(),
        layer_count: metrics.len(),
        total_frobenius_energy: total_energy,
        attention_energy_pct: pct(attn_e),
        mlp_energy_pct: pct(mlp_e),
        other_energy_pct: pct(other_e),
        early_depth_pct: pct(early_e),
        mid_depth_pct: pct(mid_e),
        deep_depth_pct: pct(deep_e),
        mean_top1_dominance_pct: mean_top1,
        brittle_rank1_layers: brittle_rank1,
        energy_spike_layers: spike_layers,
        forgetting_risk_level: risk_level,
        layers: metrics,
    }
}

pub fn render_sensitivity_table(s: &SensitivitySummary) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "LoRASpectra Spectral Sensitivity & Forgetting Risk Audit: {}\n",
        s.source_uri
    ));
    out.push_str(&format!(
        "  Layers: {} | Risk: {} | Attn: {:.1}% | MLP: {:.1}% | Early/Mid/Deep: {:.1}%/{:.1}%/{:.1}%\n",
        s.layer_count,
        s.forgetting_risk_level,
        s.attention_energy_pct,
        s.mlp_energy_pct,
        s.early_depth_pct,
        s.mid_depth_pct,
        s.deep_depth_pct
    ));
    out.push_str(&format!(
        "  Mean Top-1 Dominance: {:.1}% | Brittle Rank-1 (LS051): {} | Energy Spikes (LS052): {}\n\n",
        s.mean_top1_dominance_pct, s.brittle_rank1_layers, s.energy_spike_layers
    ));
    out.push_str(&format!(
        "{:<10} {:<8} {:>6} {:>9} {:>9} {:<7} {}\n",
        "SUBSYS", "DEPTH", "RANK", "SHARE%", "TOP1%", "RULE", "LAYER"
    ));
    out.push_str(&"-".repeat(86));
    out.push('\n');
    for l in &s.layers {
        out.push_str(&format!(
            "{:<10} {:<8} {:>6} {:>8.1}% {:>8.1}% {:<7} {}\n",
            l.subsystem,
            l.depth_band,
            l.rank,
            l.energy_share_pct,
            l.top1_dominance_pct,
            l.risk_flag.unwrap_or("OK"),
            l.base_name
        ));
    }
    out
}

fn json_esc(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

pub fn render_sensitivity_json(s: &SensitivitySummary) -> String {
    let rows: Vec<String> = s
        .layers
        .iter()
        .map(|l| {
            let rule_lit = match l.risk_flag {
                Some(r) => format!("\"{}\"", r),
                None => "null".to_string(),
            };
            let bidx_lit = match l.block_index {
                Some(i) => i.to_string(),
                None => "null".to_string(),
            };
            format!(
                r#"    {{"base_name":"{}","subsystem":"{}","block_index":{},"depth_band":"{}","rank":{},"energy_share_pct":{:.2},"top1_dominance_pct":{:.2},"spectral_norm":{:.6},"risk_flag":{}}}"#,
                json_esc(&l.base_name),
                l.subsystem,
                bidx_lit,
                l.depth_band,
                l.rank,
                l.energy_share_pct,
                l.top1_dominance_pct,
                l.spectral_norm,
                rule_lit
            )
        })
        .collect();
    let mut subsys_map = BTreeMap::new();
    subsys_map.insert("attention", s.attention_energy_pct);
    subsys_map.insert("mlp", s.mlp_energy_pct);
    subsys_map.insert("other", s.other_energy_pct);
    format!(
        r#"{{
  "source_uri": "{}",
  "layer_count": {},
  "forgetting_risk_level": "{}",
  "total_frobenius_energy": {:.6},
  "attention_energy_pct": {:.2},
  "mlp_energy_pct": {:.2},
  "other_energy_pct": {:.2},
  "early_depth_pct": {:.2},
  "mid_depth_pct": {:.2},
  "deep_depth_pct": {:.2},
  "mean_top1_dominance_pct": {:.2},
  "brittle_rank1_layers": {},
  "energy_spike_layers": {},
  "layers": [
{}
  ]
}}"#,
        json_esc(&s.source_uri),
        s.layer_count,
        s.forgetting_risk_level,
        s.total_frobenius_energy,
        s.attention_energy_pct,
        s.mlp_energy_pct,
        s.other_energy_pct,
        s.early_depth_pct,
        s.mid_depth_pct,
        s.deep_depth_pct,
        s.mean_top1_dominance_pct,
        s.brittle_rank1_layers,
        s.energy_spike_layers,
        rows.join(",\n")
    )
}

fn html_esc(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

/// Render a self-contained dark-mode HTML Catastrophic-Forgetting & Layer Sensitivity Report.
pub fn render_sensitivity_html(s: &SensitivitySummary) -> String {
    let risk_color = match s.forgetting_risk_level {
        "HIGH" => "#f85149",
        "MODERATE" => "#d29922",
        _ => "#3fb950",
    };
    let mut rows = String::new();
    for l in &s.layers {
        let (badge_txt, badge_cls) = match l.risk_flag {
            Some("LS051") => ("LS051 BRITTLE", "bad"),
            Some("LS052") => ("LS052 SPIKE", "warn"),
            Some(other) => (other, "warn"),
            None => ("OK", "ok"),
        };
        let bar_w = l.energy_share_pct.clamp(0.0, 100.0);
        rows.push_str(&format!(
            "<tr><td><code>{}</code></td><td>{}</td><td>{}</td><td>{}</td><td><div class=\"bar\"><span style=\"width:{:.1}%\"></span></div> {:.1}%</td><td>{:.1}%</td><td>{:.4}</td><td><span class=\"pill {}\">{}</span></td></tr>\n",
            html_esc(&l.base_name),
            html_esc(l.subsystem),
            html_esc(l.depth_band),
            l.rank,
            bar_w,
            l.energy_share_pct,
            l.top1_dominance_pct,
            l.spectral_norm,
            badge_cls,
            html_esc(badge_txt),
        ));
    }
    format!(
        r##"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>loraspectra Sensitivity &amp; Forgetting Risk Report</title>
<style>
:root{{--bg:#0d1117;--card:#161b22;--bd:#30363d;--tx:#e6edf3;--mu:#8b949e;--ac:#58a6ff;--ok:#3fb950;--wa:#d29922;--ba:#f85149}}
body{{margin:0;padding:24px;background:var(--bg);color:var(--tx);font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif}}
.wrap{{max-width:1160px;margin:0 auto}}
.head{{display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px;margin-bottom:20px}}
.risk{{padding:6px 14px;border-radius:999px;font-weight:700;background:rgba(255,255,255,.06);border:1px solid {risk_color};color:{risk_color}}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin-bottom:20px}}
.card{{background:var(--card);border:1px solid var(--bd);border-radius:10px;padding:14px}}
.k{{font-size:12px;color:var(--mu);text-transform:uppercase;letter-spacing:.04em}}
.v{{font-size:22px;font-weight:700;margin-top:4px}}
.split{{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:20px}}
.bar{{display:inline-block;width:72px;height:8px;background:#21262d;border-radius:4px;overflow:hidden;vertical-align:middle;margin-right:6px}}
.bar span{{display:block;height:100%;background:var(--ac)}}
table{{width:100%;border-collapse:collapse;background:var(--card);border:1px solid var(--bd);border-radius:10px;overflow:hidden;font-size:13px}}
th,td{{padding:9px 10px;border-bottom:1px solid #21262d;text-align:left}}
th{{color:var(--mu);font-weight:600;background:#11161d}}
.pill{{padding:2px 8px;border-radius:999px;font-size:11px;font-weight:700}}
.pill.ok{{background:rgba(63,185,80,.15);color:var(--ok)}}
.pill.warn{{background:rgba(210,153,34,.15);color:var(--wa)}}
.pill.bad{{background:rgba(248,81,73,.15);color:var(--ba)}}
code{{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:12px}}
@media(max-width:720px){{.split{{grid-template-columns:1fr}}}}
</style>
</head>
<body>
<div class="wrap">
  <div class="head">
    <div>
      <h1 style="margin:0;font-size:22px">loraspectra Sensitivity &amp; Forgetting Risk Report</h1>
      <div style="color:var(--mu);font-size:13px;margin-top:4px">Source: <code>{source}</code> &middot; Layers: {layers}</div>
    </div>
    <div class="risk">FORGETTING RISK: {risk}</div>
  </div>
  <div class="grid">
    <div class="card"><div class="k">Attention Energy</div><div class="v">{attn:.1}%</div></div>
    <div class="card"><div class="k">MLP Energy</div><div class="v">{mlp:.1}%</div></div>
    <div class="card"><div class="k">Mean Top-1 Dominance</div><div class="v">{top1:.1}%</div></div>
    <div class="card"><div class="k">Brittle Rank-1 (LS051)</div><div class="v">{r1}</div></div>
    <div class="card"><div class="k">Energy Spikes (LS052)</div><div class="v">{spikes}</div></div>
    <div class="card"><div class="k">Total Frobenius Energy</div><div class="v">{energy:.3}</div></div>
  </div>
  <div class="split">
    <div class="card">
      <div class="k">Subsystem Share</div>
      <div style="margin-top:8px">Attention: <strong>{attn:.1}%</strong> &middot; MLP: <strong>{mlp:.1}%</strong> &middot; Other: <strong>{other:.1}%</strong></div>
    </div>
    <div class="card">
      <div class="k">Depth Distribution</div>
      <div style="margin-top:8px">Early: <strong>{early:.1}%</strong> &middot; Mid: <strong>{mid:.1}%</strong> &middot; Deep: <strong>{deep:.1}%</strong></div>
    </div>
  </div>
  <table>
    <thead><tr><th>Layer</th><th>Subsystem</th><th>Depth</th><th>Rank</th><th>Energy Share</th><th>Top-1%</th><th>Spectral Norm</th><th>Rule</th></tr></thead>
    <tbody>
{rows}    </tbody>
  </table>
</div>
</body>
</html>
"##,
        risk_color = risk_color,
        source = html_esc(&s.source_uri),
        layers = s.layer_count,
        risk = html_esc(s.forgetting_risk_level),
        attn = s.attention_energy_pct,
        mlp = s.mlp_energy_pct,
        other = s.other_energy_pct,
        early = s.early_depth_pct,
        mid = s.mid_depth_pct,
        deep = s.deep_depth_pct,
        top1 = s.mean_top1_dominance_pct,
        r1 = s.brittle_rank1_layers,
        spikes = s.energy_spike_layers,
        energy = s.total_frobenius_energy,
        rows = rows,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::safetensors::{DType, Tensor2D};
    use crate::Rng64;

    #[test]
    fn sensitivity_detects_brittle_rank1_and_mlp_energy_spike() {
        let mut rng = Rng64::new(123);
        let mut mk_rand = |name: &str, scale: f64| LoraLayer {
            base_name: name.to_string(),
            a_key: format!("{name}.lora_A.weight"),
            b_key: format!("{name}.lora_B.weight"),
            a: Tensor2D { rows: 4, cols: 16, dtype: DType::F32, data: (0..64).map(|_| rng.next_gaussian() * 0.1 * scale).collect() },
            b: Tensor2D { rows: 16, cols: 4, dtype: DType::F32, data: (0..64).map(|_| rng.next_gaussian() * 0.1 * scale).collect() },
        };
        // Rank-1 dominated layer in deep MLP with huge scale
        let mut a_r1 = vec![0.0; 64];
        let mut b_r1 = vec![0.0; 64];
        for i in 0..16 {
            a_r1[i] = 1.5;
            b_r1[i * 4] = 1.5;
        }
        let spike = LoraLayer {
            base_name: "model.layers.31.mlp.down_proj".to_string(),
            a_key: "model.layers.31.mlp.down_proj.lora_A.weight".to_string(),
            b_key: "model.layers.31.mlp.down_proj.lora_B.weight".to_string(),
            a: Tensor2D { rows: 4, cols: 16, dtype: DType::F32, data: a_r1 },
            b: Tensor2D { rows: 16, cols: 4, dtype: DType::F32, data: b_r1 },
        };
        let layers = vec![
            mk_rand("model.layers.0.self_attn.q_proj", 1.0),
            mk_rand("model.layers.8.self_attn.v_proj", 1.0),
            mk_rand("model.layers.16.mlp.up_proj", 1.0),
            spike,
        ];
        let rep = analyze_sensitivity(&layers, "test.safetensors", 85.0, 2.5);
        assert_eq!(rep.forgetting_risk_level, "HIGH");
        assert!(rep.brittle_rank1_layers >= 1);
        assert!(rep.energy_spike_layers >= 1);
        assert!(rep.mlp_energy_pct > 90.0);
        assert!(render_sensitivity_json(&rep).contains("\"forgetting_risk_level\": \"HIGH\""));
        let html = render_sensitivity_html(&rep);
        assert!(html.contains("<!DOCTYPE html>"));
        assert!(html.contains("FORGETTING RISK: HIGH"));
        assert!(html.contains("LS051 BRITTLE"));
        assert!(html.contains("model.layers.31.mlp.down_proj"));
        let sarif = crate::sarif::build_sensitivity_sarif(&rep, "1.36.0");
        assert!(sarif.contains("\"ruleId\": \"LS051\""));
        assert!(sarif.contains("\"version\": \"2.1.0\""));
    }
}

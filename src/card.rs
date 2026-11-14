use crate::audit::{
    AdapterAuditSummary, AdapterIntegrityReport, IssueSeverity, LayerHealth, LayerSpectralReport,
};
use crate::safetensors::AdapterArchitectureInfo;

fn format_u64_commas(n: usize) -> String {
    let s = n.to_string();
    let mut out = String::with_capacity(s.len() + s.len() / 3);
    for (idx, ch) in s.chars().rev().enumerate() {
        if idx > 0 && idx % 3 == 0 {
            out.push(',');
        }
        out.push(ch);
    }
    out.chars().rev().collect()
}

fn format_bytes_human(bytes: usize) -> String {
    if bytes >= 1024 * 1024 {
        format!("{:.2} MiB", (bytes as f64) / (1024.0 * 1024.0))
    } else if bytes >= 1024 {
        format!("{:.1} KiB", (bytes as f64) / 1024.0)
    } else {
        format!("{} B", bytes)
    }
}

fn escape_md_cell(s: &str) -> String {
    s.replace('|', "\\|").replace('\n', " ")
}

fn module_role_label(rep: &LayerSpectralReport) -> &'static str {
    match rep.health {
        LayerHealth::DeadLayer => "DEAD_LAYER",
        LayerHealth::RankCollapsed => "RANK_COLLAPSED",
        LayerHealth::OverParameterized => "OVER_PARAMETERIZED",
        LayerHealth::Optimal => {
            if rep.utilization_ratio >= 0.85 {
                "BOTTLENECK_HIGH_UTIL"
            } else {
                "OPTIMAL"
            }
        }
    }
}

/// Select the most informative target modules for a HuggingFace Model Card table:
/// - For small adapters (`<= 12` layers), returns all layers ordered from highest utilization
///   (capacity bottlenecks) to lowest utilization (over-parameterized / collapsed).
/// - For larger adapters, returns up to 5 highest-utilization bottleneck modules and up to 5
///   lowest-utilization over-parameterized / rank-collapsed modules.
fn select_notable_layers<'a>(layers: &'a [LayerSpectralReport]) -> Vec<&'a LayerSpectralReport> {
    let mut ordered: Vec<(usize, &LayerSpectralReport)> = layers.iter().enumerate().collect();
    ordered.sort_by(|a, b| {
        b.1.utilization_ratio
            .partial_cmp(&a.1.utilization_ratio)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                b.1.effective_rank
                    .partial_cmp(&a.1.effective_rank)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .then_with(|| a.0.cmp(&b.0))
    });

    if ordered.len() <= 12 {
        return ordered.into_iter().map(|(_, r)| r).collect();
    }

    let mut picked: Vec<(usize, &LayerSpectralReport)> = Vec::with_capacity(10);
    for &item in ordered.iter().take(5) {
        picked.push(item);
    }
    for &item in ordered.iter().rev().take(5).collect::<Vec<_>>().into_iter().rev() {
        if !picked.iter().any(|p| p.0 == item.0) {
            picked.push(item);
        }
    }
    picked.into_iter().map(|(_, r)| r).collect()
}

/// Generate a ready-to-paste HuggingFace `README.md` Model Card Spectral Audit section,
/// including static shields.io badges linking back to `https://github.com/SalyyS1/loraspectra`,
/// summary spectral/scaling metrics, and top bottleneck / over-parameterized target modules.
pub fn generate_model_card(
    source: &str,
    arch: &AdapterArchitectureInfo,
    summary: &AdapterAuditSummary,
    integrity: &AdapterIntegrityReport,
) -> String {
    let k99_badge_rank = summary.mean_rank_at_99_energy.round().max(1.0) as usize;
    let rank_badge_color = if summary.layers.iter().any(|l| l.health == LayerHealth::DeadLayer) {
        "e05d44"
    } else if summary.compressible_byte_ratio <= 0.35 {
        "2ea44f"
    } else {
        "0969da"
    };

    let (integrity_badge_label, integrity_badge_color) = if integrity.issues.is_empty() {
        ("PASS", "2ea44f")
    } else if integrity
        .issues
        .iter()
        .any(|i| i.severity == IssueSeverity::Critical)
    {
        ("CRITICAL", "e05d44")
    } else {
        ("WARN", "d97706")
    };

    let n_layers = summary.layers.len().max(1) as f64;
    let mean_utilization_pct = (summary
        .layers
        .iter()
        .map(|l| l.utilization_ratio)
        .sum::<f64>()
        / n_layers)
        * 100.0;

    let rank_display = if arch.is_heterogeneous_rank {
        format!(
            "r={}..{} (mean {:.1})",
            arch.min_rank, arch.max_rank, summary.mean_nominal_rank
        )
    } else {
        format!("r={}", arch.max_rank)
    };

    let alpha_ratio_str = match (
        integrity.alpha_summary.min_alpha_r_ratio,
        integrity.alpha_summary.max_alpha_r_ratio,
    ) {
        (Some(lo), Some(hi)) if (hi - lo).abs() < 1e-4 => {
            if integrity.alpha_summary.use_rslora {
                format!("alpha/r={:.2} (rsLoRA)", lo)
            } else {
                format!("alpha/r={:.2}", lo)
            }
        }
        (Some(lo), Some(hi)) => format!("alpha/r={:.2}..{:.2}", lo, hi),
        _ => "implicit unit scale (alpha=r)".to_string(),
    };

    let dora_str = if integrity.alpha_summary.dora_tensor_count > 0 {
        format!(
            "{} DoRA magnitude vectors",
            integrity.alpha_summary.dora_tensor_count
        )
    } else {
        "Standard LoRA (no DoRA vectors)".to_string()
    };

    let health_summary = if integrity.issues.is_empty() {
        format!("HEALTHY ({}; {})", alpha_ratio_str, dora_str)
    } else {
        format!(
            "{} ({} issue(s); {}; {})",
            integrity.status,
            integrity.issues.len(),
            alpha_ratio_str,
            dora_str
        )
    };

    let mut md = String::with_capacity(2048);
    md.push_str("## LoRA Spectral & Integrity Audit\n\n");
    md.push_str(&format!(
        "[![Spectral Audit](https://img.shields.io/badge/LoRASpectra-99%25_Rank_r%3D{}-{})](https://github.com/SalyyS1/loraspectra) ",
        k99_badge_rank, rank_badge_color
    ));
    md.push_str(&format!(
        "[![Adapter Integrity](https://img.shields.io/badge/LoRA_Integrity-{}-{})](https://github.com/SalyyS1/loraspectra)\n\n",
        integrity_badge_label, integrity_badge_color
    ));

    md.push_str("| Metric | Value |\n");
    md.push_str("| :--- | :--- |\n");
    md.push_str(&format!(
        "| **Checkpoint** | `{}` |\n",
        escape_md_cell(source)
    ));
    md.push_str(&format!(
        "| **Base Architecture & Format** | {} (`{}`, `{}`) |\n",
        escape_md_cell(arch.inferred_architecture),
        escape_md_cell(arch.naming_format),
        arch.primary_dtype.as_str()
    ));
    md.push_str(&format!(
        "| **Total LoRA Parameters** | **{}** ({}, {} adapted modules) |\n",
        format_u64_commas(arch.total_parameters),
        format_bytes_human(arch.total_bytes),
        arch.layer_count
    ));
    md.push_str(&format!(
        "| **Nominal vs Effective / Stable Rank** | `{}` vs `r_eff = {:.2}` / `r_stable = {:.2}` |\n",
        rank_display, summary.mean_effective_rank, summary.mean_stable_rank
    ));
    md.push_str(&format!(
        "| **Mean Energy Utilization (99% Frobenius)** | **{:.1}%** |\n",
        mean_utilization_pct
    ));
    md.push_str(&format!(
        "| **Optimal Eckart-Young Rank (99% Energy)** | **k_99% = {:.1}** (Marchenko-Pastur `k_MP = {:.1}`, **-{:.1}%** compressible bytes -> {}) |\n",
        summary.mean_rank_at_99_energy,
        summary.mean_rank_mp,
        summary.compressible_byte_ratio * 100.0,
        format_bytes_human(summary.total_optimal_99_bytes)
    ));
    md.push_str(&format!(
        "| **DoRA / Alpha & SFT Health** | {} |\n\n",
        escape_md_cell(&health_summary)
    ));

    let notable = select_notable_layers(&summary.layers);
    if !notable.is_empty() {
        md.push_str("### Top Bottleneck & Over-Parameterized Target Modules\n\n");
        md.push_str("| Target Module | Shape (`d_out x d_in`) | Nominal `r` | `r_eff` | `r_stable` | `k_99%` | Utilization | Profile |\n");
        md.push_str("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |\n");
        for rep in notable {
            md.push_str(&format!(
                "| `{}` | `{} x {}` | `{}` | `{:.2}` | `{:.2}` | `{}` | `{:.1}%` | `{}` |\n",
                escape_md_cell(&rep.base_name),
                rep.d_out,
                rep.d_in,
                rep.nominal_rank,
                rep.effective_rank,
                rep.stable_rank,
                rep.rank_at_99_energy,
                rep.utilization_ratio * 100.0,
                module_role_label(rep)
            ));
        }
        md.push('\n');
    }

    if !integrity.issues.is_empty() {
        md.push_str("### Detected Adapter Integrity Diagnostics\n\n");
        for issue in &integrity.issues {
            md.push_str(&format!(
                "- **[{}] `{}`** (`{}`): {}\n",
                issue.severity.as_str(),
                issue.code,
                escape_md_cell(&issue.subject),
                escape_md_cell(&issue.message)
            ));
        }
        md.push('\n');
    }

    md.push_str(&format!(
        "*Generated by [loraspectra](https://github.com/SalyyS1/loraspectra) (`loraspectra card {}`) - zero-dependency QR-Core SVD auditor & Eckart-Young compressor for `.safetensors` LoRA/DoRA adapters.*\n",
        escape_md_cell(source)
    ));
    md
}

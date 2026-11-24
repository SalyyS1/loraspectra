use std::collections::BTreeMap;
use std::env;
use std::path::{Path, PathBuf};
use std::process;

use loraspectra::audit::{
    audit_adapter, audit_adapter_integrity_with_pissa, audit_truncation_error_bounds,
    render_audit_json,
};
use loraspectra::batch::{
    discover_checkpoints, render_batch_html, render_batch_json, render_batch_markdown,
    render_batch_table, run_batch_audit,
};
use loraspectra::card::generate_model_card;
use loraspectra::compress::{
    compress_adapter_with_options, extract_lora_from_dense_models, parse_byte_budget, BudgetTarget,
    CompressionPolicy,
};
use loraspectra::diff::{diff_adapters_full, render_diff_json, render_diff_table, analyze_subspace_drift, render_subspace_drift_json, render_subspace_drift_table};
use loraspectra::doctor::{render_doctor_html, render_doctor_json, render_doctor_markdown, run_doctor};
use loraspectra::merge::{
    analyze_adapter_interference, merge_multiple_adapters, parse_layer_weights, BlockWeightRule,
};
use loraspectra::merge_check::{
    analyze_multi_adapter_merge, render_merge_check_json, render_merge_check_table,
    AdapterMergeInput,
};
use loraspectra::adalora::{
    optimize_rank_schedule, render_rank_schedule_json, render_rank_schedule_table,
};
use loraspectra::align::{analyze_cross_layer_alignment, render_alignment_json, render_alignment_table};
use loraspectra::loraplus::{
    analyze_loraplus, rebalance_adapter_factors, render_loraplus_json, render_loraplus_table,
};
use loraspectra::mcp;
use loraspectra::knots::{
    compose_adapters_knots, render_knots_json, render_knots_table, KnotsConfig, KnotsMode,
    KnotsSeverity,
};
use loraspectra::lora_ga::{
    apply_lora_ga_base_compensation, optimize_lora_ga_init, render_lora_ga_json,
    render_lora_ga_table, GradientSource, LoraGaConfig, LoraGaScaleMode, LoraGaSeverity,
};
use loraspectra::ortho_proj::{
    project_adapter_orthogonal, render_ortho_project_json, render_ortho_project_table,
    OrthoDomain, OrthoProjectConfig, OrthoSeverity,
};
use loraspectra::quant::{analyze_adapter_quantization, QuantScheme};
use loraspectra::quant_fp8::{validate_fp8_scale_factors, render_fp8_scale_json, render_fp8_scale_table, Fp8Format};
use loraspectra::remote::{resolve_companion_config_path, resolve_source_path};
use loraspectra::safetensors::{
    bake_adapter_scaling, cast_layers_dtype, compute_peft_alpha_pattern, convert_layers_key_format,
    extract_dora_scales, extract_lora_alphas, extract_lora_layers, extract_pissa_init_layers,
    filter_dora_scales_for_layers, infer_peft_target_modules, insert_dora_scales,
    insert_unit_alpha_tensors, inspect_adapter_architecture, layers_to_tensor_map,
    merge_dora_scales, read_peft_adapter_config, read_safetensors, resolve_layer_scaling_params,
    strip_peft_base_prefix, write_peft_adapter_config, write_safetensors, DType, LoraLayer,
    PeftAdapterConfig, Tensor2D,
};
use loraspectra::sarif::{
    build_audit_sarif, build_knots_sarif, build_lora_ga_sarif, build_loraplus_sarif,
    build_merge_check_sarif, build_ortho_proj_sarif, build_rank_schedule_sarif,
    build_sensitivity_sarif, build_ties_dare_sarif,
};
use loraspectra::sensitivity::{
    analyze_sensitivity, render_sensitivity_json, render_sensitivity_table,
};
use loraspectra::ties_dare::{
    compose_adapters_ties_dare, render_ties_dare_json, render_ties_dare_table, DisjointAggregation,
    TiesDareConfig, TiesDareMode,
};
use loraspectra::vera::{audit_vera_adapter, render_vera_json, render_vera_table};

fn print_usage() {
    eprintln!(
        "loraspectra v1.23.0 - Spectral SVD Auditor, Executive Doctor (A-F + HTML), 15-Tool Stdio MCP Server, Runtime Compat Matrix, Quantization Planner, Trajectory Analyzer & Batch HTML Leaderboard (.safetensors)

SOURCES:
  All input checkpoints accept local paths, HuggingFace Hub URIs (hf://<org>/<repo> or hf://<org>/<repo>/<file.safetensors>),
  or direct https://.../*.safetensors URLs (cached automatically in ${{TMPDIR:-/tmp}}/loraspectra-cache/; respects HF_TOKEN).

USAGE:
  loraspectra info <adapter_or_hf_uri> [--json]
  loraspectra audit <adapter_or_hf_uri> [--json] [--config <adapter_config.json>] [--quant nf4|fp4|int4|int8] [--block-size 64] [--truncation-rank K] [--max-truncation-error TOL] [--sarif <output.sarif>]
  loraspectra diff <adapter_a_or_hf_uri> <adapter_b_or_hf_uri> [--config-a <cfg_a.json>] [--config-b <cfg_b.json>] [--json]
  loraspectra card <adapter_or_hf_uri> [-o <model_card.md>] [--config <adapter_config.json>] [--quant nf4|fp4|int4|int8]
  loraspectra compress <input_or_hf_uri> [output.safetensors | --out-dir <dir>] [--energy 0.99 | --rank K | --mp-denoise | --target-rank AVG_R | --budget-ratio 0.25 | --max-bytes 16M | --max-frobenius-error TOL | --max-spectral-error TOL] [--min-rank R] [--max-rank R] [--config <in_config.json>] [--peft-config <out_config.json>] [--bake-alpha] [--rslora] [--alpha A] [--format peft|kohya|diffusers] [--dtype f16|bf16|f32] [--json]
  loraspectra extract <base_or_hf_uri> <tuned_or_hf_uri> <output_lora.safetensors> [--energy 0.99 | --rank K | --mp-denoise | --target-rank AVG_R | --budget-ratio 0.05 | --max-bytes 16M | --max-frobenius-error TOL | --max-spectral-error TOL] [--format peft|kohya|diffusers] [--dtype f16|bf16|f32] [--peft-config <adapter_config.json>]
  loraspectra interference <adapter_a_or_hf_uri> <adapter_b_or_hf_uri> [--json]
  loraspectra merge-check <in_1> <in_2> [in_3 ...] [--weights W1:W2[:W3...]] [--json] [--sarif <output.sarif>] [--fail-on-conflict]
  loraspectra merge <in_1> <in_2> [in_3 ...] <output.safetensors> [--mode ortho-svd|linear|ties|dare-ties|dare-linear] [--rank K] [--density 0.5] [--drop-rate 0.5] [--seed 42] [--aggregation disjoint-mean|weighted-mean|sum] [--weights W1:W2[:W3...]] [--layer-weights \"attn=1.0:0.2,mlp=0.2:1.0\"] [--preserve-norm] [--bake-alpha] [--rslora] [--format peft|kohya|diffusers] [--dtype f16|bf16|f32] [--sarif <output.sarif>]
  loraspectra ties-dare <in_1> <in_2> [in_3 ...] [-o <output.safetensors>] [--mode ties|dare-ties|dare-linear] [--density 0.5] [--drop-rate 0.5] [--seed 42] [--rank K] [--weights W1:W2[:W3...]] [--aggregation disjoint-mean|weighted-mean|sum] [--preserve-norm] [--bake-alpha] [--rslora] [--format peft|kohya|diffusers] [--dtype f16|bf16|f32] [--sarif <output.sarif>] [--fail-on-conflict] [--json]
  loraspectra lora-ga <grad_or_warmup_uri> [-o <init_lora.safetensors>] [--rank R] [--scale stable|direction|gd|svd] [--gamma 16.0] [--alpha A] [--rslora] [--emit-init] [--base <base.safetensors> --out-base <res_base.safetensors>] [--format peft|kohya|diffusers] [--dtype f16|bf16|f32] [--sarif <output.sarif>] [--json]
  loraspectra project <candidate_uri> <anchor_1_uri> [anchor_2_uri ...] [-o <projected_lora.safetensors>] [--domain right|left|bilateral] [--anchor-energy 0.99] [--strength 1.0] [--rank K] [--preserve-norm] [--bake-alpha] [--format peft|kohya|diffusers] [--dtype f16|bf16|f32] [--sarif <output.sarif>] [--fail-on-annihilation] [--json]
  loraspectra knots <in_1> <in_2> [in_3 ...] [-o <output.safetensors>] [--mode knots-ties|iso|tsv] [--density 0.5] [--isotropy 1.0] [--rank K] [--weights W1:W2[:W3...]] [--normalize-tasks] [--preserve-norm] [--format peft|kohya|diffusers] [--dtype f16|bf16|f32] [--sarif <output.sarif>] [--fail-on-conflict] [--json]
  loraspectra vera <adapter_or_hf_uri> [--json]
  loraspectra subspace-drift <adapter_a_or_hf_uri> <adapter_b_or_hf_uri> [--config-a <cfg_a.json>] [--config-b <cfg_b.json>] [--json]
  loraspectra align <adapter_or_hf_uri> [--config <adapter_config.json>] [--json]
  loraspectra fp8-validate <adapter_or_hf_uri> --format e4m3|e5m2 [--json]
  loraspectra loraplus <adapter_or_hf_uri> [--rebalance <out.safetensors>] [--imbalance-warn 1.5] [--cond-warn 50.0] [--sarif <output.sarif>] [--json]
  loraspectra rank-schedule <adapter_or_hf_uri> [--target-avg-rank 16] [--min-rank 4] [--max-rank 64] [--sarif <output.sarif>] [--json]
  loraspectra batch-audit <dir_or_file...> [--max-depth 6] [--fail-below-score 60] [--max-dead-layers 0] [--html <board.html>] [--markdown] [--json]
  loraspectra sensitivity <adapter_or_hf_uri> [--dominance-warn 85.0] [--spike-mult 2.5] [--sarif <output.sarif>] [--json]
  loraspectra doctor <adapter_or_hf_uri> [--html <report.html>] [--json]
  loraspectra trajectory <ckpt_1> <ckpt_2> [ckpt_3 ...] [--json]
  loraspectra quant-plan <adapter_or_hf_uri> [--quant nf4|int8|fp8|bf16] [--energy 0.99] [--json]
  loraspectra compat <adapter_or_hf_uri> [--json]
  loraspectra mcp                                          # run built-in zero-dependency 15-tool JSON-RPC 2.0 stdio MCP server"
    );
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

fn resolve_input_or_exit(source_str: &str) -> PathBuf {
    match resolve_source_path(source_str) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("Error resolving {}: {}", source_str, e);
            process::exit(1);
        }
    }
}

fn load_full_adapter_bundle_or_exit(
    source_str: &str,
    explicit_config: Option<&str>,
) -> (
    BTreeMap<String, Tensor2D>,
    Vec<LoraLayer>,
    BTreeMap<String, f64>,
    BTreeMap<String, (String, Tensor2D)>,
    Option<PeftAdapterConfig>,
) {
    let resolved_path = resolve_input_or_exit(source_str);
    let tensors = match read_safetensors(&resolved_path) {
        Ok(t) => t,
        Err(e) => {
            eprintln!(
                "Error reading {} ({}): {}",
                source_str,
                resolved_path.display(),
                e
            );
            process::exit(1);
        }
    };
    let alphas = extract_lora_alphas(&tensors);
    let dora_scales = extract_dora_scales(&tensors);
    let layers = match extract_lora_layers(&tensors) {
        Ok(layers) if !layers.is_empty() => layers,
        Ok(_) => {
            eprintln!(
                "No matched (lora_A, lora_B) 2D tensor pairs found in {}",
                source_str
            );
            process::exit(1);
        }
        Err(e) => {
            eprintln!("Error extracting LoRA layers from {}: {}", source_str, e);
            process::exit(1);
        }
    };

    let peft_config = if let Some(cfg_str) = explicit_config {
        match read_peft_adapter_config(Path::new(cfg_str)) {
            Ok(cfg) => Some(cfg),
            Err(e) => {
                eprintln!("Error reading {}: {}", cfg_str, e);
                process::exit(1);
            }
        }
    } else {
        resolve_companion_config_path(source_str, &resolved_path)
            .and_then(|p| read_peft_adapter_config(&p).ok())
    };

    (tensors, layers, alphas, dora_scales, peft_config)
}

fn load_companion_pissa_init(
    source_str: &str,
    tensors: &BTreeMap<String, Tensor2D>,
) -> BTreeMap<String, LoraLayer> {
    let mut pissa_map = extract_pissa_init_layers(tensors).unwrap_or_default();
    let candidate = Path::new(source_str);
    let base_dir = if candidate.is_dir() {
        Some(candidate)
    } else {
        candidate.parent()
    };
    if let Some(dir) = base_dir {
        let sub_sf = dir.join("pissa_init").join("adapter_model.safetensors");
        if sub_sf.is_file() {
            if let Ok(sub_t) = read_safetensors(&sub_sf) {
                if let Ok(sub_layers) = extract_lora_layers(&sub_t) {
                    for l in sub_layers {
                        pissa_map.entry(l.base_name.clone()).or_insert(l);
                    }
                }
                if let Ok(sub_pissa) = extract_pissa_init_layers(&sub_t) {
                    for (k, l) in sub_pissa {
                        pissa_map.entry(k).or_insert(l);
                    }
                }
            }
        }
    }
    pissa_map
}

fn load_layers_alphas_dora_or_exit(
    source_str: &str,
) -> (
    Vec<LoraLayer>,
    BTreeMap<String, f64>,
    BTreeMap<String, (String, Tensor2D)>,
) {
    let (_, layers, alphas, dora_scales, _) = load_full_adapter_bundle_or_exit(source_str, None);
    (layers, alphas, dora_scales)
}

fn load_layers_and_alphas_or_exit(path_str: &str) -> (Vec<LoraLayer>, BTreeMap<String, f64>) {
    let (l, a, _) = load_layers_alphas_dora_or_exit(path_str);
    (l, a)
}

fn load_layers_or_exit(path_str: &str) -> Vec<LoraLayer> {
    load_layers_and_alphas_or_exit(path_str).0
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        print_usage();
        process::exit(1);
    }

    match args[1].as_str() {
        "-V" | "--version" | "version" => {
            println!("loraspectra {}", env!("CARGO_PKG_VERSION"));
        }
        "info" => {
            let json_mode = args[2..].iter().any(|a| a == "--json");
            let pos: Vec<&String> = args[2..].iter().filter(|a| !a.starts_with("--")).collect();
            if pos.is_empty() {
                print_usage();
                process::exit(1);
            }
            let source = pos[0];
            let (layers, alphas) = load_layers_and_alphas_or_exit(source);
            let info = inspect_adapter_architecture(&layers, &alphas);
            let summary = audit_adapter(&layers);

            if json_mode {
                println!("{{");
                println!("  \"source\": \"{}\",", json_escape(source));
                println!("  \"naming_format\": \"{}\",", json_escape(info.naming_format));
                println!(
                    "  \"inferred_architecture\": \"{}\",",
                    json_escape(info.inferred_architecture)
                );
                println!("  \"layer_count\": {},", info.layer_count);
                println!("  \"min_rank\": {},", info.min_rank);
                println!("  \"max_rank\": {},", info.max_rank);
                println!("  \"is_heterogeneous_rank\": {},", info.is_heterogeneous_rank);
                println!("  \"max_d_in\": {},", info.max_d_in);
                println!("  \"max_d_out\": {},", info.max_d_out);
                println!("  \"total_parameters\": {},", info.total_parameters);
                println!("  \"total_bytes\": {},", info.total_bytes);
                println!("  \"primary_dtype\": \"{}\",", info.primary_dtype.as_str());
                println!("  \"alpha_tensor_count\": {},", info.alpha_tensor_count);
                println!("  \"mean_effective_rank\": {:.2},", summary.mean_effective_rank);
                println!("  \"compressible_byte_ratio\": {:.6}", summary.compressible_byte_ratio);
                println!("}}");
            } else {
                println!("Checkpoint:             {}", source);
                println!("Naming Convention:      {}", info.naming_format);
                println!("Inferred Base Family:   {}", info.inferred_architecture);
                println!("LoRA Layer Pairs:       {}", info.layer_count);
                println!(
                    "Nominal Rank Range:     r={}..{} (heterogeneous: {})",
                    info.min_rank, info.max_rank, info.is_heterogeneous_rank
                );
                println!(
                    "Max Matrix Dimensions:  d_out={}, d_in={}",
                    info.max_d_out, info.max_d_in
                );
                println!(
                    "Storage Precision:      {} ({} parameters, {} bytes, {} .alpha scalars)",
                    info.primary_dtype.as_str(),
                    info.total_parameters,
                    info.total_bytes,
                    info.alpha_tensor_count
                );
                println!(
                    "Spectral Health:        mean r_eff={:.2}, k_99%={:.1}, k_MP={:.1} (-{:.1}% compressible at 99% energy)",
                    summary.mean_effective_rank,
                    summary.mean_rank_at_99_energy,
                    summary.mean_rank_mp,
                    summary.compressible_byte_ratio * 100.0
                );
            }
        }
        "audit" => {
            let mut json_mode = false;
            let mut config_opt: Option<String> = None;
            let mut sarif_opt: Option<String> = None;
            let mut quant_opt: Option<QuantScheme> = None;
            let mut block_size: usize = 64;
            let mut trunc_rank_opt: Option<usize> = None;
            let mut max_trunc_err_opt: Option<f64> = None;
            let mut pos: Vec<String> = Vec::new();
            let mut i = 2;
            while i < args.len() {
                match args[i].as_str() {
                    "--json" => {
                        json_mode = true;
                        i += 1;
                    }
                    "--sarif" if i + 1 < args.len() => {
                        sarif_opt = Some(args[i + 1].clone());
                        i += 2;
                    }
                    "--config" | "--peft-config" if i + 1 < args.len() => {
                        config_opt = Some(args[i + 1].clone());
                        i += 2;
                    }
                    "--quant" if i + 1 < args.len() => {
                        quant_opt = Some(QuantScheme::from_name(&args[i + 1]).unwrap_or_else(|e| {
                            eprintln!("Error: {}", e);
                            process::exit(1);
                        }));
                        i += 2;
                    }
                    "--block-size" if i + 1 < args.len() => {
                        block_size = args[i + 1].parse().expect("Invalid --block-size integer");
                        i += 2;
                    }
                    "--truncation-rank" if i + 1 < args.len() => {
                        trunc_rank_opt =
                            Some(args[i + 1].parse().expect("Invalid --truncation-rank integer"));
                        i += 2;
                    }
                    "--max-truncation-error" if i + 1 < args.len() => {
                        max_trunc_err_opt = Some(
                            args[i + 1]
                                .parse()
                                .expect("Invalid --max-truncation-error float"),
                        );
                        i += 2;
                    }
                    a if !a.starts_with("--") => {
                        pos.push(a.to_string());
                        i += 1;
                    }
                    _ => i += 1,
                }
            }
            if pos.is_empty() {
                print_usage();
                process::exit(1);
            }
            let source = &pos[0];
            let (tensors, layers, alphas, dora_scales, peft_config) =
                load_full_adapter_bundle_or_exit(source, config_opt.as_deref());
            let companion_pissa = load_companion_pissa_init(source, &tensors);
            let summary = audit_adapter(&layers);
            let mut integrity = audit_adapter_integrity_with_pissa(
                &layers,
                &tensors,
                &alphas,
                &dora_scales,
                peft_config.as_ref(),
                if companion_pissa.is_empty() {
                    None
                } else {
                    Some(&companion_pissa)
                },
            );

            let trunc_issues =
                audit_truncation_error_bounds(&summary.layers, trunc_rank_opt, max_trunc_err_opt);
            integrity.issues.extend(trunc_issues);

            if let Some(scheme) = quant_opt {
                let scales: Vec<f64> = layers
                    .iter()
                    .map(|l| {
                        resolve_layer_scaling_params(
                            l,
                            &alphas,
                            None,
                            peft_config.as_ref(),
                            false,
                        )
                        .1
                    })
                    .collect();
                let (qsum, q_issues) =
                    analyze_adapter_quantization(&layers, &scales, scheme, block_size);
                integrity.quant_summary = Some(qsum);
                integrity.issues.extend(q_issues);
            }
            if !integrity.issues.is_empty() {
                integrity.status = "ISSUES_DETECTED";
            }

            if let Some(ref s_path) = sarif_opt {
                let sarif_str = build_audit_sarif(
                    Some(&summary),
                    Some(&integrity),
                    source,
                    env!("CARGO_PKG_VERSION"),
                );
                if let Err(e) = std::fs::write(s_path, &sarif_str) {
                    eprintln!("Error writing SARIF report to {}: {}", s_path, e);
                } else {
                    println!("Saved OASIS SARIF v2.1.0 report to {}", s_path);
                }
            }

            if json_mode {
                println!("{}", render_audit_json(source, &summary, &integrity));
            } else {
                println!(
                    "{:<42} {:>6} {:>8} {:>8} {:>7} {:>6} {:>8} {:<18}",
                    "LAYER", "RANK", "R_EFF", "R_STABLE", "K_99%", "K_MP", "NORM_F", "STATUS"
                );
                println!("{}", "-".repeat(110));
                for r in &summary.layers {
                    println!(
                        "{:<42} {:>6} {:>8.2} {:>8.2} {:>7} {:>6} {:>8.4} {:<18}",
                        r.base_name,
                        r.nominal_rank,
                        r.effective_rank,
                        r.stable_rank,
                        r.rank_at_99_energy,
                        r.rank_marchenko_pastur,
                        r.frobenius_norm,
                        r.health.as_str()
                    );
                }
                println!("{}", "-".repeat(110));
                println!(
                    "Summary: {} layers | Mean rank {:.1} -> k_99% {:.1} / k_MP {:.1} (r_eff {:.2}) | Potential size reduction at 99% energy: -{:.1}%",
                    summary.layers.len(),
                    summary.mean_nominal_rank,
                    summary.mean_rank_at_99_energy,
                    summary.mean_rank_mp,
                    summary.mean_effective_rank,
                    summary.compressible_byte_ratio * 100.0
                );
                let scale_key = if integrity.alpha_summary.use_rslora {
                    "alpha/sqrt(r)"
                } else {
                    "alpha/r"
                };
                let alpha_scale_str = match (
                    integrity.alpha_summary.min_effective_scale,
                    integrity.alpha_summary.max_effective_scale,
                ) {
                    (Some(lo), Some(hi)) if (hi - lo).abs() < 1e-4 => format!("{:.2}", lo),
                    (Some(lo), Some(hi)) => format!("{:.2}..{:.2}", lo, hi),
                    _ => "implicit(1.0)".to_string(),
                };
                println!(
                    "Integrity: {} | Scaling: {} ({}={}, .alpha={}, dora={}, pissa_init={}) | Token modules inspected: {}",
                    integrity.status,
                    integrity.alpha_summary.status,
                    scale_key,
                    alpha_scale_str,
                    integrity.alpha_summary.alpha_tensor_count,
                    integrity.alpha_summary.dora_tensor_count,
                    integrity.alpha_summary.pissa_init_tensor_pairs,
                    integrity.embedding_entries.len()
                );
                if let Some(ref qs) = integrity.quant_summary {
                    println!(
                        "Quantization ({}, block {}): mean SQNR {:.2} dB (min {:.2} dB) | rel_frob {:.2}% (max {:.2}%, SVD-balanced {:.2}%) | max rel_spec {:.2}%",
                        qs.scheme.as_str(),
                        qs.block_size,
                        qs.mean_sqnr_db,
                        qs.min_sqnr_db,
                        qs.mean_relative_frobenius_error * 100.0,
                        qs.max_relative_frobenius_error * 100.0,
                        qs.mean_balanced_relative_frobenius_error * 100.0,
                        qs.max_relative_spectral_error * 100.0
                    );
                }
                for issue in &integrity.issues {
                    println!(
                        "  [{}] {} ({}): {}",
                        issue.severity.as_str(),
                        issue.code,
                        issue.subject,
                        issue.message
                    );
                }
            }
        }
        "diff" => {
            let mut json_mode = false;
            let mut config_a_opt: Option<String> = None;
            let mut config_b_opt: Option<String> = None;
            let mut pos: Vec<String> = Vec::new();
            let mut i = 2;
            while i < args.len() {
                match args[i].as_str() {
                    "--json" => {
                        json_mode = true;
                        i += 1;
                    }
                    "--config-a" if i + 1 < args.len() => {
                        config_a_opt = Some(args[i + 1].clone());
                        i += 2;
                    }
                    "--config-b" if i + 1 < args.len() => {
                        config_b_opt = Some(args[i + 1].clone());
                        i += 2;
                    }
                    a if !a.starts_with("--") => {
                        pos.push(a.to_string());
                        i += 1;
                    }
                    _ => i += 1,
                }
            }
            if pos.len() < 2 {
                print_usage();
                process::exit(1);
            }
            let source_a = &pos[0];
            let source_b = &pos[1];
            let (tensors_a, layers_a, alphas_a, dora_a, cfg_a) =
                load_full_adapter_bundle_or_exit(source_a, config_a_opt.as_deref());
            let (tensors_b, layers_b, alphas_b, dora_b, cfg_b) =
                load_full_adapter_bundle_or_exit(source_b, config_b_opt.as_deref());
            let pissa_a = load_companion_pissa_init(source_a, &tensors_a);
            let pissa_b = load_companion_pissa_init(source_b, &tensors_b);
            let summary = diff_adapters_full(
                &layers_a,
                &alphas_a,
                &dora_a,
                cfg_a.as_ref(),
                if pissa_a.is_empty() { None } else { Some(&pissa_a) },
                &layers_b,
                &alphas_b,
                &dora_b,
                cfg_b.as_ref(),
                if pissa_b.is_empty() { None } else { Some(&pissa_b) },
            );

            if json_mode {
                println!("{}", render_diff_json(&summary));
            } else {
                println!("{}", render_diff_table(&summary));
            }
        }
        "card" => {
            let mut output_opt: Option<String> = None;
            let mut config_opt: Option<String> = None;
            let mut quant_opt: Option<QuantScheme> = None;
            let mut pos: Vec<String> = Vec::new();
            let mut i = 2;
            while i < args.len() {
                match args[i].as_str() {
                    "-o" | "--output" if i + 1 < args.len() => {
                        output_opt = Some(args[i + 1].clone());
                        i += 2;
                    }
                    "--config" | "--peft-config" if i + 1 < args.len() => {
                        config_opt = Some(args[i + 1].clone());
                        i += 2;
                    }
                    "--quant" if i + 1 < args.len() => {
                        quant_opt = Some(QuantScheme::from_name(&args[i + 1]).unwrap_or_else(|e| {
                            eprintln!("Error: {}", e);
                            process::exit(1);
                        }));
                        i += 2;
                    }
                    a if !a.starts_with('-') => {
                        pos.push(a.to_string());
                        i += 1;
                    }
                    _ => i += 1,
                }
            }
            if pos.is_empty() {
                print_usage();
                process::exit(1);
            }
            let source = &pos[0];
            let (tensors, layers, alphas, dora_scales, peft_config) =
                load_full_adapter_bundle_or_exit(source, config_opt.as_deref());
            let companion_pissa = load_companion_pissa_init(source, &tensors);
            let arch = inspect_adapter_architecture(&layers, &alphas);
            let summary = audit_adapter(&layers);
            let mut integrity = audit_adapter_integrity_with_pissa(
                &layers,
                &tensors,
                &alphas,
                &dora_scales,
                peft_config.as_ref(),
                if companion_pissa.is_empty() {
                    None
                } else {
                    Some(&companion_pissa)
                },
            );
            if let Some(scheme) = quant_opt {
                let scales: Vec<f64> = layers
                    .iter()
                    .map(|l| {
                        resolve_layer_scaling_params(
                            l,
                            &alphas,
                            None,
                            peft_config.as_ref(),
                            false,
                        )
                        .1
                    })
                    .collect();
                let (qsum, q_issues) = analyze_adapter_quantization(&layers, &scales, scheme, 64);
                integrity.quant_summary = Some(qsum);
                integrity.issues.extend(q_issues);
                if !integrity.issues.is_empty() {
                    integrity.status = "ISSUES_DETECTED";
                }
            }
            let card_md = generate_model_card(source, &arch, &summary, &integrity);
            if let Some(out_path) = output_opt {
                if let Err(e) = std::fs::write(&out_path, card_md.as_bytes()) {
                    eprintln!("Failed to write model card to {}: {}", out_path, e);
                    process::exit(1);
                }
                eprintln!("Wrote HuggingFace Model Card spectral audit report to {}", out_path);
            } else {
                print!("{}", card_md);
            }
        }
        "compress" => {
            if args.len() < 3 {
                print_usage();
                process::exit(1);
            }
            #[derive(Clone, Copy)]
            enum PolicySpec {
                Fixed(usize),
                Energy(f64),
                MpDenoise,
                Budget(BudgetTarget),
                ErrorBound {
                    frob: Option<f64>,
                    spec: Option<f64>,
                },
            }

            let mut pos: Vec<String> = Vec::new();
            let mut policy_spec = PolicySpec::Energy(0.99);
            let mut max_frob_err_opt: Option<f64> = None;
            let mut max_spec_err_opt: Option<f64> = None;
            let mut min_rank: usize = 1;
            let mut max_rank: usize = 256;
            let mut input_config_opt: Option<String> = None;
            let mut peft_config_path: Option<String> = None;
            let mut out_dir_opt: Option<String> = None;
            let mut target_dtype: Option<DType> = None;
            let mut target_format: Option<String> = None;
            let mut bake_alpha = false;
            let mut rslora = false;
            let mut default_alpha: Option<f64> = None;
            let mut exclude_patterns: Vec<String> = Vec::new();
            let mut json_mode = false;

            let mut i = 2;
            while i < args.len() {
                match args[i].as_str() {
                    "--json" => {
                        json_mode = true;
                        i += 1;
                    }
                    "--exclude" if i + 1 < args.len() => {
                        exclude_patterns = args[i + 1]
                            .split(',')
                            .map(|s| s.trim().to_string())
                            .filter(|s| !s.is_empty())
                            .collect();
                        i += 2;
                    }
                    "--rank" if i + 1 < args.len() => {
                        let k: usize = args[i + 1].parse().expect("Invalid --rank integer");
                        policy_spec = PolicySpec::Fixed(k);
                        i += 2;
                    }
                    "--energy" if i + 1 < args.len() => {
                        let e: f64 = args[i + 1].parse().expect("Invalid --energy float");
                        policy_spec = PolicySpec::Energy(e);
                        i += 2;
                    }
                    "--mp-denoise" => {
                        policy_spec = PolicySpec::MpDenoise;
                        i += 1;
                    }
                    "--target-rank" if i + 1 < args.len() => {
                        let avg_r: f64 = args[i + 1].parse().expect("Invalid --target-rank float");
                        policy_spec = PolicySpec::Budget(BudgetTarget::TargetMeanRank(avg_r));
                        i += 2;
                    }
                    "--budget-ratio" if i + 1 < args.len() => {
                        let ratio: f64 = args[i + 1].parse().expect("Invalid --budget-ratio float");
                        policy_spec = PolicySpec::Budget(BudgetTarget::Ratio(ratio));
                        i += 2;
                    }
                    "--max-bytes" if i + 1 < args.len() => {
                        let bytes = parse_byte_budget(&args[i + 1]).unwrap_or_else(|e| {
                            eprintln!("Error: {}", e);
                            process::exit(1);
                        });
                        policy_spec = PolicySpec::Budget(BudgetTarget::MaxBytes(bytes));
                        i += 2;
                    }
                    "--max-frobenius-error" if i + 1 < args.len() => {
                        let tol: f64 = args[i + 1]
                            .parse()
                            .expect("Invalid --max-frobenius-error float");
                        max_frob_err_opt = Some(tol);
                        policy_spec = PolicySpec::ErrorBound {
                            frob: max_frob_err_opt,
                            spec: max_spec_err_opt,
                        };
                        i += 2;
                    }
                    "--max-spectral-error" if i + 1 < args.len() => {
                        let tol: f64 = args[i + 1]
                            .parse()
                            .expect("Invalid --max-spectral-error float");
                        max_spec_err_opt = Some(tol);
                        policy_spec = PolicySpec::ErrorBound {
                            frob: max_frob_err_opt,
                            spec: max_spec_err_opt,
                        };
                        i += 2;
                    }
                    "--min-rank" if i + 1 < args.len() => {
                        min_rank = args[i + 1].parse().expect("Invalid --min-rank integer");
                        i += 2;
                    }
                    "--max-rank" if i + 1 < args.len() => {
                        max_rank = args[i + 1].parse().expect("Invalid --max-rank integer");
                        i += 2;
                    }
                    "--bake-alpha" => {
                        bake_alpha = true;
                        i += 1;
                    }
                    "--rslora" => {
                        rslora = true;
                        bake_alpha = true;
                        i += 1;
                    }
                    "--alpha" if i + 1 < args.len() => {
                        let a: f64 = args[i + 1].parse().expect("Invalid --alpha float");
                        default_alpha = Some(a);
                        bake_alpha = true;
                        i += 2;
                    }
                    "--format" if i + 1 < args.len() => {
                        target_format = Some(args[i + 1].clone());
                        i += 2;
                    }
                    "--config" if i + 1 < args.len() => {
                        input_config_opt = Some(args[i + 1].clone());
                        i += 2;
                    }
                    "--peft-config" if i + 1 < args.len() => {
                        peft_config_path = Some(args[i + 1].clone());
                        i += 2;
                    }
                    "--out-dir" if i + 1 < args.len() => {
                        out_dir_opt = Some(args[i + 1].clone());
                        i += 2;
                    }
                    "--dtype" if i + 1 < args.len() => {
                        target_dtype =
                            Some(DType::from_name(&args[i + 1]).expect("Invalid --dtype (use f16, bf16, or f32)"));
                        i += 2;
                    }
                    a if !a.starts_with("--") => {
                        pos.push(a.to_string());
                        i += 1;
                    }
                    _ => i += 1,
                }
            }

            if pos.is_empty() {
                print_usage();
                process::exit(1);
            }
            let input_path = &pos[0];
            let output_path = if pos.len() >= 2 {
                pos[1].clone()
            } else if let Some(ref dir) = out_dir_opt {
                std::fs::create_dir_all(dir).expect("Failed to create --out-dir");
                Path::new(dir)
                    .join("adapter_model.safetensors")
                    .to_string_lossy()
                    .to_string()
            } else {
                print_usage();
                process::exit(1);
            };
            if peft_config_path.is_none() {
                if let Some(ref dir) = out_dir_opt {
                    peft_config_path = Some(
                        Path::new(dir)
                            .join("adapter_config.json")
                            .to_string_lossy()
                            .to_string(),
                    );
                }
            }

            let policy = match policy_spec {
                PolicySpec::Fixed(k) => CompressionPolicy::FixedRank(k),
                PolicySpec::Energy(e) => CompressionPolicy::AdaptiveEnergy {
                    target_energy: e,
                    min_rank,
                    max_rank,
                },
                PolicySpec::MpDenoise => CompressionPolicy::MarchenkoPastur { min_rank, max_rank },
                PolicySpec::Budget(target) => CompressionPolicy::TargetBudget {
                    target,
                    min_rank,
                    max_rank,
                },
                PolicySpec::ErrorBound { frob, spec } => CompressionPolicy::ErrorBound {
                    max_frobenius_error: frob,
                    max_spectral_error: spec,
                    min_rank,
                    max_rank,
                },
            };

            let (_, mut layers, alphas, dora_scales, source_peft_config) =
                load_full_adapter_bundle_or_exit(input_path, input_config_opt.as_deref());
            if !exclude_patterns.is_empty() {
                layers.retain(|l| !exclude_patterns.iter().any(|p| l.base_name.contains(p.as_str())));
            }
            if layers.is_empty() {
                eprintln!("Error: all LoRA layers were excluded by --exclude");
                process::exit(1);
            }

            let mut resolved_alphas: BTreeMap<String, f64> = BTreeMap::new();
            let mut pre_scales: Vec<f64> = Vec::with_capacity(layers.len());
            let mut any_rslora = rslora;
            for layer in &layers {
                let (a_val, s_val, layer_rs) = resolve_layer_scaling_params(
                    layer,
                    &alphas,
                    default_alpha,
                    source_peft_config.as_ref(),
                    rslora,
                );
                resolved_alphas.insert(layer.base_name.clone(), a_val);
                pre_scales.push(s_val);
                if layer_rs {
                    any_rslora = true;
                }
            }

            let baked = bake_alpha || !alphas.is_empty();
            if baked {
                bake_adapter_scaling(&mut layers, &resolved_alphas, default_alpha, any_rslora);
            }

            let alloc_scales: Vec<f64> = if baked {
                vec![1.0; layers.len()]
            } else {
                pre_scales.clone()
            };

            let mut res =
                compress_adapter_with_options(&layers, Some(&alloc_scales), policy, target_dtype);
            if let Some(ref fmt) = target_format {
                convert_layers_key_format(&mut res.layers, fmt).expect("Invalid --format");
            }
            if let Some(dt) = target_dtype {
                cast_layers_dtype(&mut res.layers, dt);
            }
            let retained_dora = filter_dora_scales_for_layers(&dora_scales, &res.layers);
            let final_bytes: usize = res.layers.iter().map(|l| l.raw_byte_size()).sum();
            let mut out_map = layers_to_tensor_map(&res.layers);
            if !alphas.is_empty() {
                insert_unit_alpha_tensors(&mut out_map, &res.layers);
            }
            if !retained_dora.is_empty() {
                insert_dora_scales(
                    &mut out_map,
                    &retained_dora,
                    target_format.as_deref(),
                    target_dtype,
                );
            }
            if let Some(parent) = Path::new(&output_path).parent() {
                if !parent.as_os_str().is_empty() {
                    let _ = std::fs::create_dir_all(parent);
                }
            }
            write_safetensors(Path::new(&output_path), &out_map)
                .expect("Failed to write output .safetensors");

            let out_use_rslora = if baked { false } else { any_rslora };
            let alpha_pattern =
                compute_peft_alpha_pattern(&res.layers, &pre_scales, baked, out_use_rslora);
            let inferred_targets = infer_peft_target_modules(&res.layers);
            let top_r = res
                .layers
                .iter()
                .map(|l| l.rank())
                .max()
                .unwrap_or_else(|| layers.first().map(|l| l.rank()).unwrap_or(16));

            if let Some(cfg_path) = &peft_config_path {
                write_peft_adapter_config(
                    Path::new(cfg_path),
                    &res.rank_pattern,
                    &alpha_pattern,
                    top_r,
                    out_use_rslora,
                    !retained_dora.is_empty(),
                    &inferred_targets,
                    source_peft_config.as_ref(),
                )
                .expect("Failed to write PEFT adapter_config.json");
            }

            let reduction =
                (1.0 - (final_bytes as f64) / (res.total_original_bytes.max(1) as f64)) * 100.0;
            let min_alloc = res
                .metrics
                .iter()
                .map(|m| m.compressed_rank)
                .min()
                .unwrap_or(0);
            let max_alloc = res
                .metrics
                .iter()
                .map(|m| m.compressed_rank)
                .max()
                .unwrap_or(0);
            let mean_alloc = res
                .metrics
                .iter()
                .map(|m| m.compressed_rank as f64)
                .sum::<f64>()
                / (res.metrics.len().max(1) as f64);

            if json_mode {
                println!("{{");
                println!("  \"input\": \"{}\",", json_escape(input_path));
                println!("  \"output\": \"{}\",", json_escape(&output_path));
                if let Some(ref cfg_p) = peft_config_path {
                    println!("  \"peft_config\": \"{}\",", json_escape(cfg_p));
                } else {
                    println!("  \"peft_config\": null,");
                }
                println!("  \"layer_count\": {},", res.layers.len());
                println!("  \"original_bytes\": {},", res.total_original_bytes);
                println!("  \"compressed_bytes\": {},", final_bytes);
                println!("  \"reduction_percent\": {:.4},", reduction);
                println!(
                    "  \"mean_retained_energy\": {:.6},",
                    res.overall_retained_energy
                );
                println!(
                    "  \"global_retained_energy\": {:.6},",
                    res.global_retained_energy
                );
                let max_rel_spec = res
                    .metrics
                    .iter()
                    .map(|m| m.relative_spectral_error)
                    .fold(0.0_f64, f64::max);
                let mean_rel_nuc = res
                    .metrics
                    .iter()
                    .map(|m| m.relative_nuclear_error)
                    .sum::<f64>()
                    / (res.metrics.len().max(1) as f64);
                println!("  \"min_allocated_rank\": {},", min_alloc);
                println!("  \"max_allocated_rank\": {},", max_alloc);
                println!("  \"mean_allocated_rank\": {:.4},", mean_alloc);
                println!("  \"max_relative_spectral_error\": {:.6},", max_rel_spec);
                println!("  \"mean_relative_nuclear_error\": {:.6},", mean_rel_nuc);
                println!("  \"rank_pattern\": {{");
                let rp_entries: Vec<(&String, &usize)> = res.rank_pattern.iter().collect();
                for (idx, (k, &r)) in rp_entries.iter().enumerate() {
                    let comma = if idx + 1 < rp_entries.len() { "," } else { "" };
                    println!(
                        "    \"{}\": {}{}",
                        json_escape(strip_peft_base_prefix(k)),
                        r,
                        comma
                    );
                }
                println!("  }},");
                println!("  \"alpha_pattern\": {{");
                let ap_entries: Vec<(&String, &f64)> = alpha_pattern.iter().collect();
                for (idx, (k, &a)) in ap_entries.iter().enumerate() {
                    let comma = if idx + 1 < ap_entries.len() { "," } else { "" };
                    println!("    \"{}\": {:.6}{}", json_escape(k), a, comma);
                }
                println!("  }}");
                println!("}}");
            } else {
                println!(
                    "Compressed {} -> {} ({} layers, rank {}..{}, mean {:.1}): {} bytes -> {} bytes (-{:.1}%), mean retained energy: {:.2}% (global: {:.2}%)",
                    input_path,
                    output_path,
                    res.layers.len(),
                    min_alloc,
                    max_alloc,
                    mean_alloc,
                    res.total_original_bytes,
                    final_bytes,
                    reduction,
                    res.overall_retained_energy * 100.0,
                    res.global_retained_energy * 100.0
                );
            }
        }
        "extract" => {
            if args.len() < 5 {
                print_usage();
                process::exit(1);
            }
            #[derive(Clone, Copy)]
            enum ExtractPolicySpec {
                Fixed(usize),
                Energy(f64),
                MpDenoise,
                Budget(BudgetTarget),
                ErrorBound {
                    frob: Option<f64>,
                    spec: Option<f64>,
                },
            }
            let base_path = &args[2];
            let tuned_path = &args[3];
            let output_path = &args[4];
            let mut policy_spec = ExtractPolicySpec::Fixed(16);
            let mut max_frob_err_opt: Option<f64> = None;
            let mut max_spec_err_opt: Option<f64> = None;
            let mut min_rank: usize = 1;
            let mut max_rank: usize = 128;
            let mut target_format = "peft".to_string();
            let mut target_dtype: Option<DType> = None;
            let mut peft_config_path: Option<String> = None;

            let mut i = 5;
            while i < args.len() {
                match args[i].as_str() {
                    "--rank" if i + 1 < args.len() => {
                        let k: usize = args[i + 1].parse().expect("Invalid --rank integer");
                        policy_spec = ExtractPolicySpec::Fixed(k);
                        i += 2;
                    }
                    "--energy" if i + 1 < args.len() => {
                        let e: f64 = args[i + 1].parse().expect("Invalid --energy float");
                        policy_spec = ExtractPolicySpec::Energy(e);
                        i += 2;
                    }
                    "--mp-denoise" => {
                        policy_spec = ExtractPolicySpec::MpDenoise;
                        i += 1;
                    }
                    "--target-rank" if i + 1 < args.len() => {
                        let avg_r: f64 = args[i + 1].parse().expect("Invalid --target-rank float");
                        policy_spec = ExtractPolicySpec::Budget(BudgetTarget::TargetMeanRank(avg_r));
                        i += 2;
                    }
                    "--budget-ratio" if i + 1 < args.len() => {
                        let ratio: f64 = args[i + 1].parse().expect("Invalid --budget-ratio float");
                        policy_spec = ExtractPolicySpec::Budget(BudgetTarget::Ratio(ratio));
                        i += 2;
                    }
                    "--max-bytes" if i + 1 < args.len() => {
                        let bytes = parse_byte_budget(&args[i + 1]).unwrap_or_else(|e| {
                            eprintln!("Error: {}", e);
                            process::exit(1);
                        });
                        policy_spec = ExtractPolicySpec::Budget(BudgetTarget::MaxBytes(bytes));
                        i += 2;
                    }
                    "--max-frobenius-error" if i + 1 < args.len() => {
                        let tol: f64 = args[i + 1]
                            .parse()
                            .expect("Invalid --max-frobenius-error float");
                        max_frob_err_opt = Some(tol);
                        policy_spec = ExtractPolicySpec::ErrorBound {
                            frob: max_frob_err_opt,
                            spec: max_spec_err_opt,
                        };
                        i += 2;
                    }
                    "--max-spectral-error" if i + 1 < args.len() => {
                        let tol: f64 = args[i + 1]
                            .parse()
                            .expect("Invalid --max-spectral-error float");
                        max_spec_err_opt = Some(tol);
                        policy_spec = ExtractPolicySpec::ErrorBound {
                            frob: max_frob_err_opt,
                            spec: max_spec_err_opt,
                        };
                        i += 2;
                    }
                    "--min-rank" if i + 1 < args.len() => {
                        min_rank = args[i + 1].parse().expect("Invalid --min-rank integer");
                        i += 2;
                    }
                    "--max-rank" if i + 1 < args.len() => {
                        max_rank = args[i + 1].parse().expect("Invalid --max-rank integer");
                        i += 2;
                    }
                    "--format" if i + 1 < args.len() => {
                        target_format = args[i + 1].clone();
                        i += 2;
                    }
                    "--dtype" if i + 1 < args.len() => {
                        target_dtype =
                            Some(DType::from_name(&args[i + 1]).expect("Invalid --dtype (use f16, bf16, or f32)"));
                        i += 2;
                    }
                    "--peft-config" if i + 1 < args.len() => {
                        peft_config_path = Some(args[i + 1].clone());
                        i += 2;
                    }
                    _ => i += 1,
                }
            }

            let policy = match policy_spec {
                ExtractPolicySpec::Fixed(k) => CompressionPolicy::FixedRank(k),
                ExtractPolicySpec::Energy(e) => CompressionPolicy::AdaptiveEnergy {
                    target_energy: e,
                    min_rank,
                    max_rank,
                },
                ExtractPolicySpec::MpDenoise => CompressionPolicy::MarchenkoPastur { min_rank, max_rank },
                ExtractPolicySpec::Budget(target) => CompressionPolicy::TargetBudget {
                    target,
                    min_rank,
                    max_rank,
                },
                ExtractPolicySpec::ErrorBound { frob, spec } => CompressionPolicy::ErrorBound {
                    max_frobenius_error: frob,
                    max_spectral_error: spec,
                    min_rank,
                    max_rank,
                },
            };

            let resolved_base = resolve_input_or_exit(base_path);
            let resolved_tuned = resolve_input_or_exit(tuned_path);
            let base_tensors = read_safetensors(&resolved_base).expect("Failed to read base .safetensors");
            let tuned_tensors = read_safetensors(&resolved_tuned).expect("Failed to read tuned .safetensors");
            let mut res = extract_lora_from_dense_models(&base_tensors, &tuned_tensors, policy, &target_format)
                .expect("LoRA extraction failed");
            if let Some(dt) = target_dtype {
                cast_layers_dtype(&mut res.layers, dt);
            }
            let out_map = layers_to_tensor_map(&res.layers);
            let bytes = write_safetensors(Path::new(output_path), &out_map).expect("Failed to write output .safetensors");
            if let Some(cfg_path) = &peft_config_path {
                let top_r = res.layers.iter().map(|l| l.rank()).max().unwrap_or(16);
                let unit_scales = vec![1.0; res.layers.len()];
                let alpha_pattern = compute_peft_alpha_pattern(&res.layers, &unit_scales, true, false);
                let inferred_targets = infer_peft_target_modules(&res.layers);
                write_peft_adapter_config(
                    Path::new(cfg_path),
                    &res.rank_pattern,
                    &alpha_pattern,
                    top_r,
                    false,
                    false,
                    &inferred_targets,
                    None,
                )
                .expect("Failed to write PEFT adapter_config.json");
            }
            println!(
                "Extracted {} LoRA layers from ({}, {}) -> {} ({} bytes, mean retained energy: {:.2}%)",
                res.layers.len(),
                base_path,
                tuned_path,
                output_path,
                bytes,
                res.overall_retained_energy * 100.0
            );
        }
        "interference" => {
            let json_mode = args[2..].iter().any(|a| a == "--json");
            let pos: Vec<&String> = args[2..].iter().filter(|a| !a.starts_with("--")).collect();
            if pos.len() < 2 {
                print_usage();
                process::exit(1);
            }
            let source_a = pos[0];
            let source_b = pos[1];
            let layers_a = load_layers_or_exit(source_a);
            let layers_b = load_layers_or_exit(source_b);
            let summary = match analyze_adapter_interference(&layers_a, &layers_b, source_a, source_b) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("Interference analysis failed: {}", e);
                    process::exit(1);
                }
            };

            if json_mode {
                let lw_json = match &summary.recommendation.suggested_layer_weights {
                    Some(lw) => format!("\"{}\"", json_escape(lw)),
                    None => "null".to_string(),
                };
                println!("{{");
                println!("  \"adapter_a\": \"{}\",", json_escape(source_a));
                println!("  \"adapter_b\": \"{}\",", json_escape(source_b));
                println!("  \"layer_count\": {},", summary.layers.len());
                println!("  \"high_conflict_count\": {},", summary.high_conflict_count);
                println!("  \"moderate_overlap_count\": {},", summary.moderate_overlap_count);
                println!("  \"orthogonal_safe_count\": {},", summary.orthogonal_safe_count);
                println!("  \"max_principal_cos\": {:.6},", summary.max_principal_cos);
                println!("  \"mean_principal_cos\": {:.6},", summary.mean_principal_cos);
                println!("  \"mean_frobenius_cosine\": {:.6},", summary.mean_frobenius_cosine);
                println!("  \"min_frobenius_cosine\": {:.6},", summary.min_frobenius_cosine);
                println!("  \"attn_layer_count\": {},", summary.attn_layer_count);
                println!("  \"attn_mean_conflict\": {:.6},", summary.attn_mean_conflict);
                println!("  \"attn_mean_frobenius_cos\": {:.6},", summary.attn_mean_frobenius_cos);
                println!("  \"mlp_layer_count\": {},", summary.mlp_layer_count);
                println!("  \"mlp_mean_conflict\": {:.6},", summary.mlp_mean_conflict);
                println!("  \"mlp_mean_frobenius_cos\": {:.6},", summary.mlp_mean_frobenius_cos);
                println!("  \"recommendation\": {{");
                println!(
                    "    \"mode\": \"{}\",",
                    summary.recommendation.recommended_mode
                );
                println!(
                    "    \"suggested_rank\": {},",
                    summary.recommendation.suggested_rank
                );
                println!("    \"suggested_layer_weights\": {},", lw_json);
                println!("    \"preserve_norm\": {},", summary.recommendation.preserve_norm);
                println!(
                    "    \"conflict_localization\": \"{}\",",
                    summary.recommendation.conflict_localization
                );
                println!(
                    "    \"rationale\": \"{}\",",
                    json_escape(&summary.recommendation.rationale)
                );
                println!(
                    "    \"cli_command\": \"{}\"",
                    json_escape(&summary.recommendation.cli_command)
                );
                println!("  }},");
                println!("  \"layers\": [");
                for (idx, rep) in summary.layers.iter().enumerate() {
                    let comma = if idx + 1 < summary.layers.len() { "," } else { "" };
                    println!(
                        "    {{\"base_name\": \"{}\", \"max_left_principal_cos\": {:.6}, \"max_right_principal_cos\": {:.6}, \"grassmann_left_overlap\": {:.6}, \"grassmann_right_overlap\": {:.6}, \"frobenius_cosine\": {:.6}, \"conflict_severity\": \"{}\"}}{}",
                        json_escape(&rep.base_name),
                        rep.max_left_principal_cos,
                        rep.max_right_principal_cos,
                        rep.grassmann_left_overlap,
                        rep.grassmann_right_overlap,
                        rep.frobenius_cosine,
                        rep.conflict_severity,
                        comma
                    );
                }
                println!("  ]");
                println!("}}");
            } else {
                println!(
                    "{:<42} {:>9} {:>9} {:>10} {:<18}",
                    "LAYER", "COS_U_MAX", "COS_V_MAX", "FROB_COS", "SEVERITY"
                );
                println!("{}", "-".repeat(92));
                for rep in &summary.layers {
                    println!(
                        "{:<42} {:>9.4} {:>9.4} {:>10.4} {:<18}",
                        rep.base_name,
                        rep.max_left_principal_cos,
                        rep.max_right_principal_cos,
                        rep.frobenius_cosine,
                        rep.conflict_severity
                    );
                }
                println!("{}", "-".repeat(92));
                println!(
                    "Summary: {} matched layers ({} high conflict, {} moderate overlap, {} orthogonal-safe) | max cos {:.4} | mean frob_cos {:+.4}",
                    summary.layers.len(),
                    summary.high_conflict_count,
                    summary.moderate_overlap_count,
                    summary.orthogonal_safe_count,
                    summary.max_principal_cos,
                    summary.mean_frobenius_cosine
                );
                if summary.attn_layer_count > 0 || summary.mlp_layer_count > 0 {
                    println!(
                        "Block Conflict Profile: attn={} layers (mean conflict {:.3}, frob {:+.3}) | mlp={} layers (mean conflict {:.3}, frob {:+.3}) [{}]",
                        summary.attn_layer_count,
                        summary.attn_mean_conflict,
                        summary.attn_mean_frobenius_cos,
                        summary.mlp_layer_count,
                        summary.mlp_mean_conflict,
                        summary.mlp_mean_frobenius_cos,
                        summary.recommendation.conflict_localization
                    );
                }
                println!("Automated Merge Recommendation:");
                println!("  Rationale: {}", summary.recommendation.rationale);
                println!("  Command:   {}", summary.recommendation.cli_command);
            }
        }
        "merge-check" => {
            let mut positional: Vec<String> = Vec::new();
            let mut custom_weights: Option<Vec<f64>> = None;
            let mut json_mode = false;
            let mut sarif_opt: Option<String> = None;
            let mut fail_on_conflict = false;

            let mut i = 2;
            while i < args.len() {
                match args[i].as_str() {
                    "--weights" if i + 1 < args.len() => {
                        let parsed: Vec<f64> = args[i + 1]
                            .split(':')
                            .map(|s| s.parse::<f64>().expect("Invalid float in --weights"))
                            .collect();
                        custom_weights = Some(parsed);
                        i += 2;
                    }
                    "--json" => {
                        json_mode = true;
                        i += 1;
                    }
                    "--sarif" if i + 1 < args.len() => {
                        sarif_opt = Some(args[i + 1].clone());
                        i += 2;
                    }
                    "--fail-on-conflict" => {
                        fail_on_conflict = true;
                        i += 1;
                    }
                    arg if !arg.starts_with("--") => {
                        positional.push(arg.to_string());
                        i += 1;
                    }
                    _ => i += 1,
                }
            }

            if positional.len() < 2 {
                print_usage();
                process::exit(1);
            }

            let weights = match custom_weights {
                Some(w) => {
                    if w.len() != positional.len() {
                        eprintln!(
                            "Error: --weights count ({}) must match adapter count ({})",
                            w.len(),
                            positional.len()
                        );
                        process::exit(1);
                    }
                    w
                }
                None => vec![1.0; positional.len()],
            };

            let mut inputs = Vec::with_capacity(positional.len());
            for (src, &w) in positional.iter().zip(weights.iter()) {
                let (_, layers, alphas, _, peft_config) =
                    load_full_adapter_bundle_or_exit(src, None);
                inputs.push(AdapterMergeInput {
                    name: src.clone(),
                    layers,
                    alphas,
                    peft_config,
                    weight: w,
                });
            }

            let summary = match analyze_multi_adapter_merge(&inputs) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("merge-check failed: {}", e);
                    process::exit(1);
                }
            };

            if let Some(ref s_path) = sarif_opt {
                let sarif_str = build_merge_check_sarif(&summary, env!("CARGO_PKG_VERSION"));
                if let Err(e) = std::fs::write(s_path, &sarif_str) {
                    eprintln!("Error writing SARIF report to {}: {}", s_path, e);
                    process::exit(1);
                }
            }

            if json_mode {
                println!("{}", render_merge_check_json(&summary));
            } else {
                print!("{}", render_merge_check_table(&summary));
                if let Some(ref s_path) = sarif_opt {
                    println!("Saved OASIS SARIF v2.1.0 merge-check report to {}", s_path);
                }
            }

            if fail_on_conflict
                && (!summary.incompatible_shapes.is_empty()
                    || summary.destructive_layer_count > 0
                    || summary.advisor.risk_level == "HIGH")
            {
                process::exit(2);
            }
        }
        "merge" => {
            if args.len() < 5 {
                print_usage();
                process::exit(1);
            }
            let mut positional: Vec<String> = Vec::new();
            let mut mode = "ortho-svd".to_string();
            let mut rank_opt: Option<usize> = None;
            let mut density_opt: Option<f64> = None;
            let mut drop_rate_opt: Option<f64> = None;
            let mut seed_opt: Option<u64> = None;
            let mut agg_opt: Option<DisjointAggregation> = None;
            let mut sarif_opt: Option<String> = None;
            let mut custom_weights: Option<Vec<f64>> = None;
            let mut block_rules: Vec<BlockWeightRule> = Vec::new();
            let mut preserve_norm = false;
            let mut bake_alpha = false;
            let mut rslora = false;
            let mut default_alpha: Option<f64> = None;
            let mut target_format: Option<String> = None;
            let mut target_dtype: Option<DType> = None;

            let mut i = 2;
            while i < args.len() {
                match args[i].as_str() {
                    "--mode" if i + 1 < args.len() => {
                        mode = args[i + 1].clone();
                        i += 2;
                    }
                    "--rank" if i + 1 < args.len() => {
                        rank_opt = Some(args[i + 1].parse().expect("Invalid --rank"));
                        i += 2;
                    }
                    "--density" if i + 1 < args.len() => {
                        density_opt = Some(args[i + 1].parse().expect("Invalid --density float"));
                        i += 2;
                    }
                    "--drop-rate" if i + 1 < args.len() => {
                        drop_rate_opt =
                            Some(args[i + 1].parse().expect("Invalid --drop-rate float"));
                        i += 2;
                    }
                    "--seed" if i + 1 < args.len() => {
                        seed_opt = Some(args[i + 1].parse().expect("Invalid --seed integer"));
                        i += 2;
                    }
                    "--aggregation" if i + 1 < args.len() => {
                        agg_opt = Some(
                            DisjointAggregation::from_name(&args[i + 1])
                                .expect("Invalid --aggregation"),
                        );
                        i += 2;
                    }
                    "--sarif" if i + 1 < args.len() => {
                        sarif_opt = Some(args[i + 1].clone());
                        i += 2;
                    }
                    "--weights" if i + 1 < args.len() => {
                        let parsed: Vec<f64> = args[i + 1]
                            .split(':')
                            .map(|s| s.parse::<f64>().expect("Invalid float in --weights"))
                            .collect();
                        custom_weights = Some(parsed);
                        i += 2;
                    }
                    "--layer-weights" if i + 1 < args.len() => {
                        block_rules = parse_layer_weights(&args[i + 1])
                            .expect("Invalid --layer-weights specification");
                        i += 2;
                    }
                    "--preserve-norm" => {
                        preserve_norm = true;
                        i += 1;
                    }
                    "--bake-alpha" => {
                        bake_alpha = true;
                        i += 1;
                    }
                    "--rslora" => {
                        rslora = true;
                        bake_alpha = true;
                        i += 1;
                    }
                    "--alpha" if i + 1 < args.len() => {
                        default_alpha = Some(args[i + 1].parse().expect("Invalid --alpha float"));
                        bake_alpha = true;
                        i += 2;
                    }
                    "--format" if i + 1 < args.len() => {
                        target_format = Some(args[i + 1].clone());
                        i += 2;
                    }
                    "--dtype" if i + 1 < args.len() => {
                        target_dtype = Some(
                            DType::from_name(&args[i + 1])
                                .expect("Invalid --dtype (use f16, bf16, or f32)"),
                        );
                        i += 2;
                    }
                    arg if !arg.starts_with("--") => {
                        positional.push(arg.to_string());
                        i += 1;
                    }
                    _ => i += 1,
                }
            }

            if positional.len() < 3 {
                print_usage();
                process::exit(1);
            }
            let output_path = positional.last().unwrap().clone();
            let input_paths = &positional[..positional.len() - 1];

            let mut adapters: Vec<Vec<LoraLayer>> = Vec::with_capacity(input_paths.len());
            let mut any_alphas = false;
            let mut adapter_doras: Vec<BTreeMap<String, (String, Tensor2D)>> =
                Vec::with_capacity(input_paths.len());
            for p in input_paths {
                let (mut l, alphas, dora) = load_layers_alphas_dora_or_exit(p);
                if !alphas.is_empty() {
                    any_alphas = true;
                }
                adapter_doras.push(dora);
                if bake_alpha || !alphas.is_empty() {
                    bake_adapter_scaling(&mut l, &alphas, default_alpha, rslora);
                }
                adapters.push(l);
            }

            let default_rank = adapters[0].first().map(|l| l.rank()).unwrap_or(16);
            let rank = rank_opt.unwrap_or(default_rank);
            let n_adapters = adapters.len();
            let weights = custom_weights.unwrap_or_else(|| {
                if mode == "linear" || mode == "dare-linear" || mode == "dare" {
                    vec![1.0 / (n_adapters as f64); n_adapters]
                } else {
                    vec![1.0; n_adapters]
                }
            });
            let combined_dora = merge_dora_scales(&adapter_doras, &weights);

            let mut merged = if block_rules.is_empty()
                && TiesDareMode::from_name(&mode).is_ok()
            {
                let td_mode = TiesDareMode::from_name(&mode).unwrap();
                let mut cfg = TiesDareConfig::new(td_mode, rank);
                if let Some(d) = density_opt {
                    cfg.density = d;
                }
                if let Some(dr) = drop_rate_opt {
                    cfg.drop_rate = dr;
                }
                if let Some(s) = seed_opt {
                    cfg.seed = s;
                }
                if let Some(agg) = agg_opt {
                    cfg.aggregation = agg;
                }
                cfg.preserve_spectral_norm = preserve_norm;
                let inputs: Vec<AdapterMergeInput> = input_paths
                    .iter()
                    .zip(adapters.into_iter())
                    .zip(weights.iter())
                    .map(|((src, layers), &w)| AdapterMergeInput {
                        name: src.clone(),
                        layers,
                        alphas: BTreeMap::new(),
                        peft_config: None,
                        weight: w,
                    })
                    .collect();
                let summary = compose_adapters_ties_dare(&inputs, &cfg).unwrap_or_else(|e| {
                    eprintln!("Merge failed: {}", e);
                    process::exit(1);
                });
                if !summary.incompatible_shapes.is_empty() {
                    let inc = &summary.incompatible_shapes[0];
                    eprintln!(
                        "Incompatible layer shapes on {}: ({}x{}) vs ({}x{})",
                        inc.base_name, inc.shape_a.0, inc.shape_a.1, inc.shape_b.0, inc.shape_b.1
                    );
                    process::exit(1);
                }
                if let Some(ref s_path) = sarif_opt {
                    let sarif_str = build_ties_dare_sarif(&summary, env!("CARGO_PKG_VERSION"));
                    if let Err(e) = std::fs::write(s_path, &sarif_str) {
                        eprintln!("Error writing SARIF report to {}: {}", s_path, e);
                        process::exit(1);
                    }
                }
                summary.merged_layers
            } else {
                merge_multiple_adapters(
                    &adapters,
                    &weights,
                    &mode,
                    rank,
                    &block_rules,
                    preserve_norm,
                )
                .expect("Merge failed")
            };

            if let Some(ref fmt) = target_format {
                convert_layers_key_format(&mut merged, fmt).expect("Invalid --format");
            }
            if let Some(dt) = target_dtype {
                cast_layers_dtype(&mut merged, dt);
            }
            let mut out_map = layers_to_tensor_map(&merged);
            if any_alphas {
                insert_unit_alpha_tensors(&mut out_map, &merged);
            }
            if !combined_dora.is_empty() {
                insert_dora_scales(&mut out_map, &combined_dora, target_format.as_deref(), target_dtype);
            }
            let bytes = write_safetensors(Path::new(&output_path), &out_map).expect("Write failed");
            println!(
                "Merged {} adapters ({} layers) using mode '{}' (target rank {}, {} block rules, preserve_norm={}) -> {} ({} bytes)",
                n_adapters,
                merged.len(),
                mode,
                rank,
                block_rules.len(),
                preserve_norm,
                output_path,
                bytes
            );
        }
        "ties-dare" | "compose" => {
            let mut positional: Vec<String> = Vec::new();
            let mut out_opt: Option<String> = None;
            let mut mode_str = "ties".to_string();
            let mut rank_opt: Option<usize> = None;
            let mut density_opt: Option<f64> = None;
            let mut drop_rate_opt: Option<f64> = None;
            let mut seed_opt: Option<u64> = None;
            let mut agg_opt: Option<DisjointAggregation> = None;
            let mut custom_weights: Option<Vec<f64>> = None;
            let mut preserve_norm = false;
            let mut bake_alpha = false;
            let mut rslora = false;
            let mut default_alpha: Option<f64> = None;
            let mut target_format: Option<String> = None;
            let mut target_dtype: Option<DType> = None;
            let mut sarif_opt: Option<String> = None;
            let mut fail_on_conflict = false;
            let mut json_mode = false;

            let mut i = 2;
            while i < args.len() {
                match args[i].as_str() {
                    "-o" | "--output" if i + 1 < args.len() => {
                        out_opt = Some(args[i + 1].clone());
                        i += 2;
                    }
                    "--mode" if i + 1 < args.len() => {
                        mode_str = args[i + 1].clone();
                        i += 2;
                    }
                    "--rank" if i + 1 < args.len() => {
                        rank_opt = Some(args[i + 1].parse().expect("Invalid --rank"));
                        i += 2;
                    }
                    "--density" if i + 1 < args.len() => {
                        density_opt = Some(args[i + 1].parse().expect("Invalid --density float"));
                        i += 2;
                    }
                    "--drop-rate" if i + 1 < args.len() => {
                        drop_rate_opt =
                            Some(args[i + 1].parse().expect("Invalid --drop-rate float"));
                        i += 2;
                    }
                    "--seed" if i + 1 < args.len() => {
                        seed_opt = Some(args[i + 1].parse().expect("Invalid --seed integer"));
                        i += 2;
                    }
                    "--aggregation" if i + 1 < args.len() => {
                        agg_opt = Some(
                            DisjointAggregation::from_name(&args[i + 1])
                                .unwrap_or_else(|e| {
                                    eprintln!("Error: {}", e);
                                    process::exit(1);
                                }),
                        );
                        i += 2;
                    }
                    "--weights" if i + 1 < args.len() => {
                        let parsed: Vec<f64> = args[i + 1]
                            .split(':')
                            .map(|s| s.parse::<f64>().expect("Invalid float in --weights"))
                            .collect();
                        custom_weights = Some(parsed);
                        i += 2;
                    }
                    "--preserve-norm" => {
                        preserve_norm = true;
                        i += 1;
                    }
                    "--bake-alpha" => {
                        bake_alpha = true;
                        i += 1;
                    }
                    "--rslora" => {
                        rslora = true;
                        bake_alpha = true;
                        i += 1;
                    }
                    "--alpha" if i + 1 < args.len() => {
                        default_alpha = Some(args[i + 1].parse().expect("Invalid --alpha float"));
                        bake_alpha = true;
                        i += 2;
                    }
                    "--format" if i + 1 < args.len() => {
                        target_format = Some(args[i + 1].clone());
                        i += 2;
                    }
                    "--dtype" if i + 1 < args.len() => {
                        target_dtype = Some(
                            DType::from_name(&args[i + 1])
                                .expect("Invalid --dtype (use f16, bf16, or f32)"),
                        );
                        i += 2;
                    }
                    "--sarif" if i + 1 < args.len() => {
                        sarif_opt = Some(args[i + 1].clone());
                        i += 2;
                    }
                    "--fail-on-conflict" => {
                        fail_on_conflict = true;
                        i += 1;
                    }
                    "--json" => {
                        json_mode = true;
                        i += 1;
                    }
                    arg if !arg.starts_with('-') => {
                        positional.push(arg.to_string());
                        i += 1;
                    }
                    _ => i += 1,
                }
            }

            if positional.len() < 2 {
                print_usage();
                process::exit(1);
            }

            let td_mode = match TiesDareMode::from_name(&mode_str) {
                Ok(m) => m,
                Err(e) => {
                    eprintln!("Error: {}", e);
                    process::exit(1);
                }
            };

            let n_inputs = positional.len();
            let weights = match custom_weights {
                Some(w) => {
                    if w.len() != n_inputs {
                        eprintln!(
                            "Error: --weights count ({}) must match adapter count ({})",
                            w.len(),
                            n_inputs
                        );
                        process::exit(1);
                    }
                    w
                }
                None => {
                    if td_mode == TiesDareMode::DareLinear {
                        vec![1.0 / (n_inputs as f64); n_inputs]
                    } else {
                        vec![1.0; n_inputs]
                    }
                }
            };

            let mut inputs = Vec::with_capacity(n_inputs);
            let mut adapter_doras = Vec::with_capacity(n_inputs);
            let mut any_alphas = false;
            for (src, &w) in positional.iter().zip(weights.iter()) {
                let (_, mut layers, alphas, dora, peft_config) =
                    load_full_adapter_bundle_or_exit(src, None);
                if !alphas.is_empty() {
                    any_alphas = true;
                }
                adapter_doras.push(dora);
                if bake_alpha {
                    bake_adapter_scaling(&mut layers, &alphas, default_alpha, rslora);
                    inputs.push(AdapterMergeInput {
                        name: src.clone(),
                        layers,
                        alphas: BTreeMap::new(),
                        peft_config: None,
                        weight: w,
                    });
                } else {
                    inputs.push(AdapterMergeInput {
                        name: src.clone(),
                        layers,
                        alphas,
                        peft_config,
                        weight: w,
                    });
                }
            }

            let default_rank = inputs[0]
                .layers
                .iter()
                .map(|l| l.rank())
                .max()
                .unwrap_or(16);
            let mut cfg = TiesDareConfig::new(td_mode, rank_opt.unwrap_or(default_rank));
            if let Some(d) = density_opt {
                cfg.density = d;
            }
            if let Some(dr) = drop_rate_opt {
                cfg.drop_rate = dr;
            }
            if let Some(s) = seed_opt {
                cfg.seed = s;
            }
            if let Some(agg) = agg_opt {
                cfg.aggregation = agg;
            }
            cfg.preserve_spectral_norm = preserve_norm;
            cfg.output_dtype = target_dtype;

            let mut summary = match compose_adapters_ties_dare(&inputs, &cfg) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("ties-dare failed: {}", e);
                    process::exit(1);
                }
            };

            if let Some(ref s_path) = sarif_opt {
                let sarif_str = build_ties_dare_sarif(&summary, env!("CARGO_PKG_VERSION"));
                if let Err(e) = std::fs::write(s_path, &sarif_str) {
                    eprintln!("Error writing SARIF report to {}: {}", s_path, e);
                    process::exit(1);
                }
            }

            if let Some(ref out_path) = out_opt {
                if !summary.incompatible_shapes.is_empty() {
                    eprintln!(
                        "Error: cannot write {} because {} layer shape mismatch(es) were detected",
                        out_path,
                        summary.incompatible_shapes.len()
                    );
                    process::exit(1);
                }
                if let Some(ref fmt) = target_format {
                    convert_layers_key_format(&mut summary.merged_layers, fmt)
                        .expect("Invalid --format");
                }
                if let Some(dt) = target_dtype {
                    cast_layers_dtype(&mut summary.merged_layers, dt);
                }
                let mut out_map = layers_to_tensor_map(&summary.merged_layers);
                if any_alphas {
                    insert_unit_alpha_tensors(&mut out_map, &summary.merged_layers);
                }
                let combined_dora = merge_dora_scales(&adapter_doras, &weights);
                if !combined_dora.is_empty() {
                    insert_dora_scales(
                        &mut out_map,
                        &combined_dora,
                        target_format.as_deref(),
                        target_dtype,
                    );
                }
                if let Err(e) = write_safetensors(Path::new(out_path), &out_map) {
                    eprintln!("Error writing {}: {}", out_path, e);
                    process::exit(1);
                }
            }

            if json_mode {
                println!("{}", render_ties_dare_json(&summary));
            } else {
                print!("{}", render_ties_dare_table(&summary));
                if let Some(ref out_path) = out_opt {
                    println!(
                        "Saved composed adapter ({} layers) to {}",
                        summary.merged_layers.len(),
                        out_path
                    );
                }
                if let Some(ref s_path) = sarif_opt {
                    println!("Saved OASIS SARIF v2.1.0 report to {}", s_path);
                }
            }

            if fail_on_conflict && summary.has_conflicts() {
                process::exit(2);
            }
        }
        "vera" => {
            // VeRA scaling vector fingerprinting
            if args.len() < 3 {
                eprintln!("Usage: loraspectra vera <adapter_or_hf_uri> [--json]");
                process::exit(1);
            }
            let input_path = resolve_input_or_exit(&args[2]);
            let use_json = args.iter().any(|a| a == "--json");
            let tensor_map = read_safetensors(&input_path).unwrap_or_else(|e| {
                eprintln!("Error reading {}: {}", args[2], e);
                process::exit(1);
            });
            let report = audit_vera_adapter(&tensor_map);
            if report.total_layers == 0 {
                eprintln!("No VeRA scaling vectors (vera_lambda_b/vera_lambda_d or scaling_b/scaling_d) found in {}", args[2]);
                process::exit(1);
            }
            if use_json {
                println!("{}", render_vera_json(&report));
            } else {
                print!("{}", render_vera_table(&report));
            }
            if report.non_finite_layers > 0 || report.dead_rank_layers > 0 || report.dead_output_layers > 0 {
                process::exit(2);
            }
        }
        "subspace-drift" => {
            // Subspace drift analysis between two LoRA checkpoints
            if args.len() < 4 {
                eprintln!("Usage: loraspectra subspace-drift <adapter_a> <adapter_b> [--config-a <cfg.json>] [--config-b <cfg.json>] [--json]");
                process::exit(1);
            }
            let path_a = resolve_input_or_exit(&args[2]);
            let path_b = resolve_input_or_exit(&args[3]);
            let use_json = args.iter().any(|a| a == "--json");

            let config_a_path = args.windows(2).find(|w| w[0] == "--config-a").map(|w| w[1].clone())
                .or_else(|| resolve_companion_config_path(&args[2], &path_a).map(|p| p.to_string_lossy().to_string()));
            let config_b_path = args.windows(2).find(|w| w[0] == "--config-b").map(|w| w[1].clone())
                .or_else(|| resolve_companion_config_path(&args[3], &path_b).map(|p| p.to_string_lossy().to_string()));

            let map_a = read_safetensors(&path_a).unwrap_or_else(|e| {
                eprintln!("Error reading {}: {}", args[2], e);
                process::exit(1);
            });
            let map_b = read_safetensors(&path_b).unwrap_or_else(|e| {
                eprintln!("Error reading {}: {}", args[3], e);
                process::exit(1);
            });

            let layers_a = extract_lora_layers(&map_a).unwrap_or_else(|e| {
                eprintln!("Error extracting LoRA layers from {}: {}", args[2], e);
                process::exit(1);
            });
            let layers_b = extract_lora_layers(&map_b).unwrap_or_else(|e| {
                eprintln!("Error extracting LoRA layers from {}: {}", args[3], e);
                process::exit(1);
            });

            let alphas_a = extract_lora_alphas(&map_a);
            let alphas_b = extract_lora_alphas(&map_b);

            let cfg_a = config_a_path.as_deref().and_then(|p| read_peft_adapter_config(Path::new(p)).ok());
            let cfg_b = config_b_path.as_deref().and_then(|p| read_peft_adapter_config(Path::new(p)).ok());

            let mut full_alphas_a = alphas_a;
            if let Some(cfg) = &cfg_a {
                if let Some(a) = cfg.lora_alpha {
                    for layer in &layers_a {
                        full_alphas_a.entry(layer.base_name.clone()).or_insert(a);
                    }
                }
            }
            let mut full_alphas_b = alphas_b;
            if let Some(cfg) = &cfg_b {
                if let Some(a) = cfg.lora_alpha {
                    for layer in &layers_b {
                        full_alphas_b.entry(layer.base_name.clone()).or_insert(a);
                    }
                }
            }

            let summary = analyze_subspace_drift(&layers_a, &full_alphas_a, &layers_b, &full_alphas_b);
            if summary.total_shared_layers == 0 {
                eprintln!("No shared LoRA layers found between the two adapters.");
                process::exit(1);
            }

            if use_json {
                println!("{}", render_subspace_drift_json(&summary));
            } else {
                print!("{}", render_subspace_drift_table(&summary));
            }
            if summary.catastrophic_drift_layers > 0 {
                process::exit(2);
            }
        }
        "align" => {
            // Cross-layer subspace alignment matrix (Grassmannian distance between adjacent layers)
            if args.len() < 3 {
                eprintln!("Usage: loraspectra align <adapter_or_hf_uri> [--config <adapter_config.json>] [--json]");
                process::exit(1);
            }
            let (tensors, layers, alphas, _dora_scales, cfg) =
                load_full_adapter_bundle_or_exit(&args[2], args.windows(2).find(|w| w[0] == "--config").map(|w| w[1].as_str()));
            let _ = (tensors, alphas, cfg);
            let use_json = args.iter().any(|a| a == "--json");
            if layers.is_empty() {
                eprintln!("No LoRA layers found in {}", args[2]);
                process::exit(1);
            }
            let matrix = analyze_cross_layer_alignment(&layers);
            if use_json {
                println!("{}", render_alignment_json(&matrix));
            } else {
                print!("{}", render_alignment_table(&matrix));
            }
            if matrix.redundant_pairs > 0 || matrix.disjoint_pairs > 0 {
                process::exit(2);
            }
        }
        "fp8-validate" => {
            // FP8 E4M3/E5M2 scale factor validation
            if args.len() < 3 {
                eprintln!("Usage: loraspectra fp8-validate <adapter_or_hf_uri> --format e4m3|e5m2 [--json]");
                process::exit(1);
            }
            let input_path = resolve_input_or_exit(&args[2]);
            let use_json = args.iter().any(|a| a == "--json");
            let fmt_str = args.windows(2)
                .find(|w| w[0] == "--format")
                .map(|w| w[1].as_str())
                .unwrap_or("e4m3");
            let fp8_format = match Fp8Format::from_name(fmt_str) {
                Ok(f) => f,
                Err(e) => {
                    eprintln!("Error: {}", e);
                    process::exit(1);
                }
            };
            let tensor_map = match loraspectra::safetensors::read_safetensors(&input_path) {
                Ok(m) => m,
                Err(e) => {
                    eprintln!("Error reading {}: {}", args[2], e);
                    process::exit(1);
                }
            };
            let report = validate_fp8_scale_factors(&tensor_map, fp8_format);
            if use_json {
                println!("{}", render_fp8_scale_json(&report));
            } else {
                print!("{}", render_fp8_scale_table(&report));
            }
            if report.has_blocking_issues() {
                process::exit(2);
            }
        }
        "lora-ga" => {
            if args.len() < 3 {
                eprintln!("Usage: loraspectra lora-ga <grad_or_warmup_uri> [-o <init_lora.safetensors>] [--rank R] [--scale stable|direction|gd|svd] [--gamma 16.0] [--alpha A] [--rslora] [--emit-init] [--base <base.safetensors> --out-base <res_base.safetensors>] [--format peft|kohya|diffusers] [--dtype f16|bf16|f32] [--sarif <output.sarif>] [--json]");
                process::exit(1);
            }
            let src_uri = &args[2];
            let mut out_path: Option<String> = None;
            let mut base_path: Option<String> = None;
            let mut out_base_path: Option<String> = None;
            let mut sarif_path: Option<String> = None;
            let mut explicit_cfg: Option<String> = None;
            let mut use_json = false;
            let mut cfg = LoraGaConfig::default();

            let mut i = 3;
            while i < args.len() {
                match args[i].as_str() {
                    "-o" | "--out" | "--output" => {
                        i += 1;
                        out_path = args.get(i).cloned();
                    }
                    "--rank" | "-r" => {
                        i += 1;
                        cfg.target_rank = args
                            .get(i)
                            .and_then(|s| s.parse::<usize>().ok())
                            .unwrap_or_else(|| {
                                eprintln!("Invalid --rank value");
                                process::exit(1);
                            });
                    }
                    "--scale" => {
                        i += 1;
                        let name = args.get(i).map(|s| s.as_str()).unwrap_or("");
                        cfg.scale_mode = LoraGaScaleMode::from_name(name).unwrap_or_else(|e| {
                            eprintln!("Error: {}", e);
                            process::exit(1);
                        });
                    }
                    "--gamma" => {
                        i += 1;
                        cfg.stable_gamma = args
                            .get(i)
                            .and_then(|s| s.parse::<f64>().ok())
                            .unwrap_or_else(|| {
                                eprintln!("Invalid --gamma value");
                                process::exit(1);
                            });
                    }
                    "--alpha" => {
                        i += 1;
                        cfg.lora_alpha = args
                            .get(i)
                            .and_then(|s| s.parse::<f64>().ok())
                            .unwrap_or_else(|| {
                                eprintln!("Invalid --alpha value");
                                process::exit(1);
                            });
                    }
                    "--rslora" => cfg.use_rslora = true,
                    "--emit-init" => cfg.emit_init_reference = true,
                    "--base" => {
                        i += 1;
                        base_path = args.get(i).cloned();
                    }
                    "--out-base" => {
                        i += 1;
                        out_base_path = args.get(i).cloned();
                    }
                    "--format" => {
                        i += 1;
                        cfg.key_format = args.get(i).cloned().unwrap_or_else(|| "peft".to_string());
                    }
                    "--dtype" => {
                        i += 1;
                        let dt_name = args.get(i).map(|s| s.as_str()).unwrap_or("");
                        cfg.output_dtype = Some(DType::from_name(dt_name).unwrap_or_else(|e| {
                            eprintln!("Error: {}", e);
                            process::exit(1);
                        }));
                    }
                    "--config" => {
                        i += 1;
                        explicit_cfg = args.get(i).cloned();
                    }
                    "--sarif" => {
                        i += 1;
                        sarif_path = args.get(i).cloned();
                    }
                    "--seed" => {
                        i += 1;
                        cfg.seed = args
                            .get(i)
                            .and_then(|s| s.parse::<u64>().ok())
                            .unwrap_or(42);
                    }
                    "--json" => use_json = true,
                    other => {
                        eprintln!("Unknown lora-ga argument: {}", other);
                        process::exit(1);
                    }
                }
                i += 1;
            }

            if base_path.is_some() != out_base_path.is_some() {
                eprintln!("Both --base <in.safetensors> and --out-base <out.safetensors> must be specified together.");
                process::exit(1);
            }
            if out_base_path.is_some() {
                cfg.base_compensated = true;
            }

            let resolved_src = resolve_input_or_exit(src_uri);
            let tensors = read_safetensors(&resolved_src).unwrap_or_else(|e| {
                eprintln!("Error reading {}: {}", src_uri, e);
                process::exit(1);
            });

            let summary = match extract_lora_layers(&tensors) {
                Ok(layers) if !layers.is_empty() => {
                    let alphas = extract_lora_alphas(&tensors);
                    let peft_cfg = explicit_cfg
                        .as_deref()
                        .and_then(|p| read_peft_adapter_config(Path::new(p)).ok())
                        .or_else(|| {
                            resolve_companion_config_path(src_uri, &resolved_src)
                                .and_then(|p| read_peft_adapter_config(&p).ok())
                        });
                    optimize_lora_ga_init(
                        src_uri,
                        GradientSource::LowRank {
                            layers: &layers,
                            alphas: &alphas,
                            peft_config: peft_cfg.as_ref(),
                        },
                        &cfg,
                    )
                }
                _ => optimize_lora_ga_init(src_uri, GradientSource::Dense { tensors: &tensors }, &cfg),
            }
            .unwrap_or_else(|e| {
                eprintln!("Error optimizing LoRA-GA initialization: {}", e);
                process::exit(1);
            });

            if let Some(ref out_file) = out_path {
                let mut out_map = layers_to_tensor_map(&summary.layers);
                if !summary.init_reference_layers.is_empty() {
                    for (k, v) in layers_to_tensor_map(&summary.init_reference_layers) {
                        out_map.insert(k, v);
                    }
                }
                if let Err(e) = write_safetensors(Path::new(out_file), &out_map) {
                    eprintln!("Error writing {}: {}", out_file, e);
                    process::exit(1);
                }
            }

            if let (Some(ref b_in), Some(ref b_out)) = (base_path, out_base_path) {
                let resolved_base = resolve_input_or_exit(b_in);
                let base_tensors = read_safetensors(&resolved_base).unwrap_or_else(|e| {
                    eprintln!("Error reading base weights {}: {}", b_in, e);
                    process::exit(1);
                });
                let (comp_tensors, updated_cnt) = apply_lora_ga_base_compensation(
                    &base_tensors,
                    &summary.layers,
                    summary.effective_forward_scale,
                )
                .unwrap_or_else(|e| {
                    eprintln!("Error compensating base weights: {}", e);
                    process::exit(1);
                });
                if updated_cnt == 0 {
                    eprintln!("Warning: zero base tensors matched LoRA-GA layer names.");
                }
                if let Err(e) = write_safetensors(Path::new(b_out), &comp_tensors) {
                    eprintln!("Error writing compensated base {}: {}", b_out, e);
                    process::exit(1);
                }
            }

            if let Some(ref sp) = sarif_path {
                let sarif_str = build_lora_ga_sarif(&summary, env!("CARGO_PKG_VERSION"));
                if let Err(e) = std::fs::write(sp, sarif_str) {
                    eprintln!("Error writing SARIF report {}: {}", sp, e);
                    process::exit(1);
                }
            }

            if use_json {
                print!("{}", render_lora_ga_json(&summary));
            } else {
                print!("{}", render_lora_ga_table(&summary));
            }

            if summary
                .diagnostics
                .iter()
                .any(|d| d.severity == LoraGaSeverity::Error)
            {
                process::exit(2);
            }
        }
        "project" | "ortho-project" => {
            if args.len() < 4 {
                eprintln!("Usage: loraspectra project <candidate_uri> <anchor_1_uri> [anchor_2_uri ...] [-o <projected_lora.safetensors>] [--domain right|left|bilateral] [--anchor-energy 0.99] [--strength 1.0] [--rank K] [--preserve-norm] [--bake-alpha] [--format peft|kohya|diffusers] [--dtype f16|bf16|f32] [--sarif <output.sarif>] [--fail-on-annihilation] [--json]");
                process::exit(1);
            }
            let mut positional: Vec<String> = Vec::new();
            let mut out_path: Option<String> = None;
            let mut sarif_path: Option<String> = None;
            let mut fail_on_annihilation = false;
            let mut use_json = false;
            let mut cfg = OrthoProjectConfig::default();

            let mut i = 2;
            while i < args.len() {
                match args[i].as_str() {
                    "-o" | "--out" | "--output" => {
                        i += 1;
                        out_path = args.get(i).cloned();
                    }
                    "--domain" => {
                        i += 1;
                        let d_str = args.get(i).map(|s| s.as_str()).unwrap_or("");
                        cfg.domain = OrthoDomain::from_name(d_str).unwrap_or_else(|e| {
                            eprintln!("Error: {}", e);
                            process::exit(1);
                        });
                    }
                    "--anchor-energy" => {
                        i += 1;
                        cfg.anchor_energy_threshold = args
                            .get(i)
                            .and_then(|s| s.parse::<f64>().ok())
                            .unwrap_or_else(|| {
                                eprintln!("Invalid --anchor-energy value");
                                process::exit(1);
                            });
                    }
                    "--strength" => {
                        i += 1;
                        cfg.strength = args
                            .get(i)
                            .and_then(|s| s.parse::<f64>().ok())
                            .unwrap_or_else(|| {
                                eprintln!("Invalid --strength value");
                                process::exit(1);
                            });
                    }
                    "--rank" | "-r" => {
                        i += 1;
                        let r_val = args
                            .get(i)
                            .and_then(|s| s.parse::<usize>().ok())
                            .unwrap_or_else(|| {
                                eprintln!("Invalid --rank value");
                                process::exit(1);
                            });
                        cfg.target_rank = Some(r_val);
                    }
                    "--preserve-norm" => cfg.preserve_norm = true,
                    "--bake-alpha" => cfg.bake_alpha = true,
                    "--format" => {
                        i += 1;
                        cfg.key_format = args.get(i).cloned().unwrap_or_else(|| "peft".to_string());
                    }
                    "--dtype" => {
                        i += 1;
                        let dt_name = args.get(i).map(|s| s.as_str()).unwrap_or("");
                        cfg.output_dtype = Some(DType::from_name(dt_name).unwrap_or_else(|e| {
                            eprintln!("Error: {}", e);
                            process::exit(1);
                        }));
                    }
                    "--sarif" => {
                        i += 1;
                        sarif_path = args.get(i).cloned();
                    }
                    "--fail-on-annihilation" => fail_on_annihilation = true,
                    "--json" => use_json = true,
                    other if other.starts_with('-') => {
                        eprintln!("Unknown project argument: {}", other);
                        process::exit(1);
                    }
                    pos => positional.push(pos.to_string()),
                }
                i += 1;
            }

            if positional.len() < 2 {
                eprintln!("Error: expected <candidate_uri> followed by at least one <anchor_uri>.");
                process::exit(1);
            }

            let (_, cand_layers, cand_alphas, cand_dora, cand_cfg) =
                load_full_adapter_bundle_or_exit(&positional[0], None);
            let cand_input = AdapterMergeInput {
                name: positional[0].clone(),
                layers: cand_layers,
                alphas: cand_alphas,
                peft_config: cand_cfg,
                weight: 1.0,
            };

            let mut anchor_inputs = Vec::with_capacity(positional.len() - 1);
            for anc_uri in &positional[1..] {
                let (_, anc_layers, anc_alphas, _, anc_cfg) =
                    load_full_adapter_bundle_or_exit(anc_uri, None);
                anchor_inputs.push(AdapterMergeInput {
                    name: anc_uri.clone(),
                    layers: anc_layers,
                    alphas: anc_alphas,
                    peft_config: anc_cfg,
                    weight: 1.0,
                });
            }

            let summary = project_adapter_orthogonal(&cand_input, &anchor_inputs, &cfg)
                .unwrap_or_else(|e| {
                    eprintln!("Error running orthogonal projection: {}", e);
                    process::exit(1);
                });

            if let Some(ref out_file) = out_path {
                let mut out_map = layers_to_tensor_map(&summary.projected_layers);
                if !cand_dora.is_empty() {
                    let kept_dora = filter_dora_scales_for_layers(&cand_dora, &summary.projected_layers);
                    insert_dora_scales(&mut out_map, &kept_dora, Some(&cfg.key_format), cfg.output_dtype);
                }
                if let Err(e) = write_safetensors(Path::new(out_file), &out_map) {
                    eprintln!("Error writing {}: {}", out_file, e);
                    process::exit(1);
                }
            }

            if let Some(ref sp) = sarif_path {
                let sarif_str = build_ortho_proj_sarif(&summary, env!("CARGO_PKG_VERSION"));
                if let Err(e) = std::fs::write(sp, sarif_str) {
                    eprintln!("Error writing SARIF report {}: {}", sp, e);
                    process::exit(1);
                }
            }

            if use_json {
                print!("{}", render_ortho_project_json(&summary));
            } else {
                print!("{}", render_ortho_project_table(&summary));
            }

            if !summary.incompatible_shapes.is_empty()
                || (fail_on_annihilation
                    && summary
                        .diagnostics
                        .iter()
                        .any(|d| d.rule_id == "LS041" || d.severity == OrthoSeverity::Error))
            {
                process::exit(2);
            }
        }
        "knots" | "tsv" => {
            if args.len() < 4 {
                eprintln!("Usage: loraspectra knots <in_1> <in_2> [in_3 ...] [-o <output.safetensors>] [--mode knots-ties|iso|tsv] [--density 0.5] [--isotropy 1.0] [--rank K] [--weights W1:W2[:W3...]] [--normalize-tasks] [--preserve-norm] [--format peft|kohya|diffusers] [--dtype f16|bf16|f32] [--sarif <output.sarif>] [--fail-on-conflict] [--json]");
                process::exit(1);
            }
            let mut positional: Vec<String> = Vec::new();
            let mut out_path: Option<String> = None;
            let mut sarif_path: Option<String> = None;
            let mut weights_arg: Option<String> = None;
            let mut fail_on_conflict = false;
            let mut use_json = false;
            let mut cfg = KnotsConfig::default();
            if args[1] == "tsv" {
                cfg.mode = KnotsMode::TaskSingularVectors;
            }

            let mut i = 2;
            while i < args.len() {
                match args[i].as_str() {
                    "-o" | "--out" | "--output" => {
                        i += 1;
                        out_path = args.get(i).cloned();
                    }
                    "--mode" => {
                        i += 1;
                        let m_str = args.get(i).map(|s| s.as_str()).unwrap_or("");
                        cfg.mode = KnotsMode::from_name(m_str).unwrap_or_else(|e| {
                            eprintln!("Error: {}", e);
                            process::exit(1);
                        });
                    }
                    "--density" => {
                        i += 1;
                        cfg.density = args
                            .get(i)
                            .and_then(|s| s.parse::<f64>().ok())
                            .unwrap_or_else(|| {
                                eprintln!("Invalid --density value");
                                process::exit(1);
                            });
                    }
                    "--isotropy" => {
                        i += 1;
                        cfg.isotropy = args
                            .get(i)
                            .and_then(|s| s.parse::<f64>().ok())
                            .unwrap_or_else(|| {
                                eprintln!("Invalid --isotropy value");
                                process::exit(1);
                            });
                    }
                    "--rank" | "-r" => {
                        i += 1;
                        let r_val = args
                            .get(i)
                            .and_then(|s| s.parse::<usize>().ok())
                            .unwrap_or_else(|| {
                                eprintln!("Invalid --rank value");
                                process::exit(1);
                            });
                        cfg.target_rank = Some(r_val);
                    }
                    "--weights" | "-w" => {
                        i += 1;
                        weights_arg = args.get(i).cloned();
                    }
                    "--normalize-tasks" => cfg.normalize_tasks = true,
                    "--preserve-norm" => cfg.preserve_norm = true,
                    "--format" => {
                        i += 1;
                        cfg.key_format = args.get(i).cloned().unwrap_or_else(|| "peft".to_string());
                    }
                    "--dtype" => {
                        i += 1;
                        let dt_name = args.get(i).map(|s| s.as_str()).unwrap_or("");
                        cfg.output_dtype = Some(DType::from_name(dt_name).unwrap_or_else(|e| {
                            eprintln!("Error: {}", e);
                            process::exit(1);
                        }));
                    }
                    "--sarif" => {
                        i += 1;
                        sarif_path = args.get(i).cloned();
                    }
                    "--fail-on-conflict" => fail_on_conflict = true,
                    "--json" => use_json = true,
                    other if other.starts_with('-') => {
                        eprintln!("Unknown knots argument: {}", other);
                        process::exit(1);
                    }
                    pos => positional.push(pos.to_string()),
                }
                i += 1;
            }

            if positional.len() < 2 {
                eprintln!("Error: expected at least 2 input adapters for knots.");
                process::exit(1);
            }

            let parsed_weights: Vec<f64> = if let Some(ref w_str) = weights_arg {
                let parts: Vec<f64> = w_str
                    .split(':')
                    .map(|p| {
                        p.trim().parse::<f64>().unwrap_or_else(|_| {
                            eprintln!("Invalid weight '{}' in --weights {}", p, w_str);
                            process::exit(1);
                        })
                    })
                    .collect();
                if parts.len() != positional.len() {
                    eprintln!(
                        "Error: --weights has {} entries but {} adapters were provided",
                        parts.len(),
                        positional.len()
                    );
                    process::exit(1);
                }
                parts
            } else {
                vec![1.0; positional.len()]
            };

            let mut merge_inputs = Vec::with_capacity(positional.len());
            for (idx, uri) in positional.iter().enumerate() {
                let (_, layers, alphas, _, peft_cfg) = load_full_adapter_bundle_or_exit(uri, None);
                merge_inputs.push(AdapterMergeInput {
                    name: uri.clone(),
                    layers,
                    alphas,
                    peft_config: peft_cfg,
                    weight: parsed_weights[idx],
                });
            }

            let summary = compose_adapters_knots(&merge_inputs, &cfg).unwrap_or_else(|e| {
                eprintln!("Error running knots composition: {}", e);
                process::exit(1);
            });

            if let Some(ref out_file) = out_path {
                let out_map = layers_to_tensor_map(&summary.merged_layers);
                if let Err(e) = write_safetensors(Path::new(out_file), &out_map) {
                    eprintln!("Error writing {}: {}", out_file, e);
                    process::exit(1);
                }
            }

            if let Some(ref sp) = sarif_path {
                let sarif_str = build_knots_sarif(&summary, env!("CARGO_PKG_VERSION"));
                if let Err(e) = std::fs::write(sp, sarif_str) {
                    eprintln!("Error writing SARIF report {}: {}", sp, e);
                    process::exit(1);
                }
            }

            if use_json {
                print!("{}", render_knots_json(&summary));
            } else {
                print!("{}", render_knots_table(&summary));
            }

            if !summary.incompatible_shapes.is_empty()
                || (fail_on_conflict
                    && summary
                        .diagnostics
                        .iter()
                        .any(|d| d.rule_id == "LS043" || d.severity == KnotsSeverity::Error))
            {
                process::exit(2);
            }
        }
        "loraplus" => {
            if args.len() < 3 {
                eprintln!("Usage: loraspectra loraplus <adapter_or_hf_uri> [--rebalance <out.safetensors>] [--imbalance-warn 1.5] [--cond-warn 50.0] [--sarif <out.sarif>] [--json]");
                process::exit(1);
            }
            let input_uri = &args[2];
            let mut rebalance_out: Option<String> = None;
            let mut sarif_path: Option<String> = None;
            let mut imbalance_warn: f64 = 1.5;
            let mut cond_warn: f64 = 50.0;
            let mut use_json = false;
            let mut i = 3;
            while i < args.len() {
                match args[i].as_str() {
                    "--rebalance" => {
                        i += 1;
                        if i < args.len() {
                            rebalance_out = Some(args[i].clone());
                        }
                    }
                    "--sarif" => {
                        i += 1;
                        if i < args.len() {
                            sarif_path = Some(args[i].clone());
                        }
                    }
                    "--imbalance-warn" => {
                        i += 1;
                        if i < args.len() {
                            imbalance_warn = args[i].parse().unwrap_or(1.5);
                        }
                    }
                    "--cond-warn" => {
                        i += 1;
                        if i < args.len() {
                            cond_warn = args[i].parse().unwrap_or(50.0);
                        }
                    }
                    "--json" => use_json = true,
                    _ => {}
                }
                i += 1;
            }
            let (_, layers, _, _, _) = load_full_adapter_bundle_or_exit(input_uri, None);
            let summary = analyze_loraplus(&layers, input_uri, imbalance_warn, cond_warn);
            if let Some(ref out_file) = rebalance_out {
                let rebalanced = rebalance_adapter_factors(&layers);
                let out_map = layers_to_tensor_map(&rebalanced);
                if let Err(e) = write_safetensors(Path::new(out_file), &out_map) {
                    eprintln!("Error writing rebalanced adapter {}: {}", out_file, e);
                    process::exit(1);
                }
            }
            if let Some(ref sp) = sarif_path {
                let sarif_str = build_loraplus_sarif(&summary, env!("CARGO_PKG_VERSION"));
                if let Err(e) = std::fs::write(sp, sarif_str) {
                    eprintln!("Error writing SARIF report {}: {}", sp, e);
                    process::exit(1);
                }
            }
            if use_json {
                print!("{}", render_loraplus_json(&summary));
            } else {
                print!("{}", render_loraplus_table(&summary));
            }
        }
        "rank-schedule" => {
            if args.len() < 3 {
                eprintln!("Usage: loraspectra rank-schedule <adapter_or_hf_uri> [--target-avg-rank 16] [--min-rank 4] [--max-rank 64] [--sarif <out.sarif>] [--json]");
                process::exit(1);
            }
            let input_uri = &args[2];
            let mut target_avg_rank: f64 = 16.0;
            let mut min_rank: usize = 4;
            let mut max_rank: usize = 64;
            let mut sarif_path: Option<String> = None;
            let mut use_json = false;
            let mut i = 3;
            while i < args.len() {
                match args[i].as_str() {
                    "--target-avg-rank" => {
                        i += 1;
                        if i < args.len() {
                            target_avg_rank = args[i].parse().unwrap_or(16.0);
                        }
                    }
                    "--min-rank" => {
                        i += 1;
                        if i < args.len() {
                            min_rank = args[i].parse().unwrap_or(4);
                        }
                    }
                    "--max-rank" => {
                        i += 1;
                        if i < args.len() {
                            max_rank = args[i].parse().unwrap_or(64);
                        }
                    }
                    "--sarif" => {
                        i += 1;
                        if i < args.len() {
                            sarif_path = Some(args[i].clone());
                        }
                    }
                    "--json" => use_json = true,
                    _ => {}
                }
                i += 1;
            }
            let (_, layers, _, _, _) = load_full_adapter_bundle_or_exit(input_uri, None);
            let summary = optimize_rank_schedule(&layers, input_uri, target_avg_rank, min_rank, max_rank);
            if let Some(ref sp) = sarif_path {
                let sarif_str = build_rank_schedule_sarif(&summary, env!("CARGO_PKG_VERSION"));
                if let Err(e) = std::fs::write(sp, sarif_str) {
                    eprintln!("Error writing SARIF report {}: {}", sp, e);
                    process::exit(1);
                }
            }
            if use_json {
                print!("{}", render_rank_schedule_json(&summary));
            } else {
                print!("{}", render_rank_schedule_table(&summary));
            }
        }
        "batch-audit" => {
            if args.len() < 3 {
                eprintln!("Usage: loraspectra batch-audit <dir_or_file...> [--max-depth 6] [--fail-below-score 60] [--max-dead-layers 0] [--html <board.html>] [--markdown] [--json]");
                process::exit(1);
            }
            let mut inputs: Vec<PathBuf> = Vec::new();
            let mut max_depth: usize = 6;
            let mut fail_below_score: Option<f64> = None;
            let mut max_dead_layers: Option<usize> = None;
            let mut html_out: Option<String> = None;
            let mut use_json = false;
            let mut use_markdown = false;
            let mut i = 2;
            while i < args.len() {
                match args[i].as_str() {
                    "--max-depth" => {
                        i += 1;
                        if i < args.len() {
                            max_depth = args[i].parse().unwrap_or(6);
                        }
                    }
                    "--fail-below-score" => {
                        i += 1;
                        if i < args.len() {
                            fail_below_score = args[i].parse().ok();
                        }
                    }
                    "--max-dead-layers" => {
                        i += 1;
                        if i < args.len() {
                            max_dead_layers = args[i].parse().ok();
                        }
                    }
                    "--html" => {
                        i += 1;
                        if i < args.len() {
                            html_out = Some(args[i].clone());
                        }
                    }
                    "--json" => use_json = true,
                    "--markdown" => use_markdown = true,
                    other if !other.starts_with("--") => inputs.push(PathBuf::from(other)),
                    _ => {}
                }
                i += 1;
            }
            let checkpoints = discover_checkpoints(&inputs, max_depth);
            if checkpoints.is_empty() {
                eprintln!("No .safetensors checkpoints found in supplied paths.");
                process::exit(1);
            }
            let report = run_batch_audit(&checkpoints);
            if let Some(hp) = html_out {
                let html = render_batch_html(&report);
                if let Err(e) = std::fs::write(&hp, html) {
                    eprintln!("Error writing HTML leaderboard {}: {}", hp, e);
                    process::exit(1);
                }
            }
            if use_json {
                println!("{}", render_batch_json(&report));
            } else if use_markdown {
                print!("{}", render_batch_markdown(&report));
            } else {
                print!("{}", render_batch_table(&report));
            }
            let mut gate_failed = false;
            if let Some(min_score) = fail_below_score {
                if report.entries.iter().any(|e| e.health_score < min_score) {
                    gate_failed = true;
                }
            }
            if let Some(max_dead) = max_dead_layers {
                if report.entries.iter().any(|e| e.dead_layers > max_dead) {
                    gate_failed = true;
                }
            }
            if gate_failed {
                process::exit(2);
            }
        }
        "sensitivity" => {
            if args.len() < 3 {
                eprintln!("Usage: loraspectra sensitivity <adapter_or_hf_uri> [--dominance-warn 85.0] [--spike-mult 2.5] [--sarif <output.sarif>] [--json]");
                process::exit(1);
            }
            let input_uri = &args[2];
            let mut dom_warn: f64 = 85.0;
            let mut spike_mult: f64 = 2.5;
            let mut sarif_out: Option<String> = None;
            let mut use_json = false;
            let mut i = 3;
            while i < args.len() {
                match args[i].as_str() {
                    "--dominance-warn" => {
                        i += 1;
                        if i < args.len() {
                            dom_warn = args[i].parse().unwrap_or(85.0);
                        }
                    }
                    "--spike-mult" => {
                        i += 1;
                        if i < args.len() {
                            spike_mult = args[i].parse().unwrap_or(2.5);
                        }
                    }
                    "--sarif" => {
                        i += 1;
                        if i < args.len() {
                            sarif_out = Some(args[i].clone());
                        }
                    }
                    "--json" => use_json = true,
                    _ => {}
                }
                i += 1;
            }
            let (_, layers, _, _, _) = load_full_adapter_bundle_or_exit(input_uri, None);
            let rep = analyze_sensitivity(&layers, input_uri, dom_warn, spike_mult);
            if let Some(ref spath) = sarif_out {
                let sarif_doc = build_sensitivity_sarif(&rep, env!("CARGO_PKG_VERSION"));
                if let Err(e) = std::fs::write(spath, sarif_doc) {
                    eprintln!("Failed to write SARIF {}: {}", spath, e);
                    process::exit(1);
                }
            }
            if use_json {
                println!("{}", render_sensitivity_json(&rep));
            } else {
                print!("{}", render_sensitivity_table(&rep));
            }
        }
        "doctor" => {
            if args.len() < 3 {
                eprintln!("Usage: loraspectra doctor <adapter_or_hf_uri> [--html <report.html>] [--json]");
                process::exit(1);
            }
            let input_uri = &args[2];
            let mut use_json = false;
            let mut html_out: Option<String> = None;
            let mut i = 3;
            while i < args.len() {
                match args[i].as_str() {
                    "--json" => use_json = true,
                    "--html" => {
                        i += 1;
                        if i < args.len() {
                            html_out = Some(args[i].clone());
                        }
                    }
                    _ => {}
                }
                i += 1;
            }
            let (tensors, layers, _, _, peft_cfg) = load_full_adapter_bundle_or_exit(input_uri, None);
            let rep = run_doctor(input_uri, &layers, &tensors, peft_cfg.as_ref());
            if let Some(ref hpath) = html_out {
                if let Err(e) = std::fs::write(hpath, render_doctor_html(&rep)) {
                    eprintln!("Failed to write HTML doctor report {}: {}", hpath, e);
                    process::exit(1);
                }
            }
            if use_json {
                println!("{}", render_doctor_json(&rep));
            } else {
                print!("{}", render_doctor_markdown(&rep));
            }
        }
        "trajectory" => {
            let mut use_json = false;
            let mut sources: Vec<String> = Vec::new();
            for arg in &args[2..] {
                if arg == "--json" {
                    use_json = true;
                } else if !arg.starts_with('-') {
                    sources.push(arg.clone());
                }
            }
            if sources.len() < 2 {
                eprintln!("Usage: loraspectra trajectory <ckpt_1> <ckpt_2> [ckpt_3 ...] [--json]");
                process::exit(1);
            }
            match loraspectra::trajectory::analyze_trajectory(&sources) {
                Ok(rep) => {
                    if use_json {
                        println!("{}", loraspectra::trajectory::render_trajectory_json(&rep));
                    } else {
                        print!("{}", loraspectra::trajectory::render_trajectory_table(&rep));
                    }
                }
                Err(e) => {
                    eprintln!("Trajectory analysis error: {}", e);
                    process::exit(1);
                }
            }
        }
        "quant-plan" => {
            if args.len() < 3 {
                eprintln!("Usage: loraspectra quant-plan <adapter_or_hf_uri> [--quant nf4|fp4|int4|int8|fp8|bf16|fp16] [--energy 0.99] [--json]");
                process::exit(1);
            }
            let input_uri = &args[2];
            let mut quant = "nf4".to_string();
            let mut energy = 0.99_f64;
            let mut use_json = false;
            let mut i = 3;
            while i < args.len() {
                match args[i].as_str() {
                    "--json" => use_json = true,
                    "--quant" => {
                        i += 1;
                        if i < args.len() {
                            quant = args[i].clone();
                        }
                    }
                    "--energy" => {
                        i += 1;
                        if i < args.len() {
                            if let Ok(v) = args[i].parse::<f64>() {
                                energy = v;
                            }
                        }
                    }
                    _ => {}
                }
                i += 1;
            }
            let (_, layers, _, _, _) = load_full_adapter_bundle_or_exit(input_uri, None);
            let rep = loraspectra::quant_plan::build_quant_plan(input_uri, &layers, &quant, energy);
            if use_json {
                println!("{}", loraspectra::quant_plan::render_quant_plan_json(&rep));
            } else {
                print!("{}", loraspectra::quant_plan::render_quant_plan_table(&rep));
            }
        }
        "compat" => {
            if args.len() < 3 {
                eprintln!("Usage: loraspectra compat <adapter_or_hf_uri> [--json]");
                process::exit(1);
            }
            let input_uri = &args[2];
            let use_json = args[3..].iter().any(|a| a == "--json");
            let (tensors, _, _, _, _) = load_full_adapter_bundle_or_exit(input_uri, None);
            match loraspectra::compat::check_serving_compatibility(input_uri, &tensors) {
                Ok(rep) => {
                    if use_json {
                        println!("{}", loraspectra::compat::render_compat_json(&rep));
                    } else {
                        print!("{}", loraspectra::compat::render_compat_table(&rep));
                    }
                }
                Err(e) => {
                    eprintln!("Compatibility check error: {}", e);
                    process::exit(1);
                }
            }
        }
        "mcp" => {
            if let Err(e) = mcp::serve_stdio() {
                eprintln!("loraspectra mcp server error: {}", e);
                process::exit(1);
            }
        }
        _ => {
            print_usage();
            process::exit(1);
        }
    }
}

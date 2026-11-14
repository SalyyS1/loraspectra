use std::collections::BTreeMap;
use std::env;
use std::path::{Path, PathBuf};
use std::process;

use loraspectra::audit::{audit_adapter, audit_adapter_integrity, render_audit_json};
use loraspectra::card::generate_model_card;
use loraspectra::compress::{
    compress_adapter, extract_lora_from_dense_models, CompressionPolicy,
};
use loraspectra::diff::{diff_adapters, render_diff_json, render_diff_table};
use loraspectra::merge::{
    analyze_adapter_interference, merge_multiple_adapters, parse_layer_weights, BlockWeightRule,
};
use loraspectra::merge_check::{
    analyze_multi_adapter_merge, render_merge_check_json, render_merge_check_table,
    AdapterMergeInput,
};
use loraspectra::remote::{resolve_companion_config_path, resolve_source_path};
use loraspectra::safetensors::{
    bake_adapter_scaling, cast_layers_dtype, convert_layers_key_format, extract_dora_scales,
    extract_lora_alphas, extract_lora_layers, insert_dora_scales, insert_unit_alpha_tensors,
    inspect_adapter_architecture, layers_to_tensor_map, read_peft_adapter_config, read_safetensors,
    write_peft_rank_pattern_config, write_safetensors, DType, LoraLayer, PeftAdapterConfig,
    Tensor2D,
};
use loraspectra::sarif::{build_audit_sarif, build_merge_check_sarif};

fn print_usage() {
    eprintln!(
        "loraspectra v0.9.0 - Spectral SVD Auditor, Dynamic Rank Compressor, Extractor & Orthogonal Merger for LoRA/DoRA (.safetensors)

SOURCES:
  All input checkpoints accept local paths, HuggingFace Hub URIs (hf://<org>/<repo> or hf://<org>/<repo>/<file.safetensors>),
  or direct https://.../*.safetensors URLs (cached automatically in ${{TMPDIR:-/tmp}}/loraspectra-cache/; respects HF_TOKEN).

USAGE:
  loraspectra info <adapter_or_hf_uri> [--json]
  loraspectra audit <adapter_or_hf_uri> [--json] [--config <adapter_config.json>] [--sarif <output.sarif>]
  loraspectra diff <adapter_a_or_hf_uri> <adapter_b_or_hf_uri> [--json]
  loraspectra card <adapter_or_hf_uri> [-o <model_card.md>] [--config <adapter_config.json>]
  loraspectra compress <input_or_hf_uri> <output.safetensors> [--energy 0.99 | --rank K | --mp-denoise] [--bake-alpha] [--rslora] [--alpha A] [--format peft|kohya|diffusers] [--dtype f16|bf16|f32] [--peft-config <adapter_config.json>]
  loraspectra extract <base_or_hf_uri> <tuned_or_hf_uri> <output_lora.safetensors> [--energy 0.99 | --rank K | --mp-denoise] [--format peft|kohya|diffusers] [--dtype f16|bf16|f32] [--peft-config <adapter_config.json>]
  loraspectra interference <adapter_a_or_hf_uri> <adapter_b_or_hf_uri> [--json]
  loraspectra merge-check <in_1> <in_2> [in_3 ...] [--weights W1:W2[:W3...]] [--json] [--sarif <output.sarif>] [--fail-on-conflict]
  loraspectra merge <in_1> <in_2> [in_3 ...] <output.safetensors> [--mode ortho-svd|linear|ties] [--rank K] [--weights W1:W2[:W3...]] [--layer-weights \"attn=1.0:0.2,mlp=0.2:1.0\"] [--preserve-norm] [--bake-alpha] [--rslora] [--format peft|kohya|diffusers] [--dtype f16|bf16|f32]"
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
            let summary = audit_adapter(&layers);
            let integrity = audit_adapter_integrity(
                &layers,
                &tensors,
                &alphas,
                &dora_scales,
                peft_config.as_ref(),
            );

            if let Some(ref s_path) = sarif_opt {
                let sarif_str = build_audit_sarif(Some(&summary), Some(&integrity), source, "0.9.0");
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
                let alpha_scale_str = match (
                    integrity.alpha_summary.min_alpha_r_ratio,
                    integrity.alpha_summary.max_alpha_r_ratio,
                ) {
                    (Some(lo), Some(hi)) if (hi - lo).abs() < 1e-4 => format!("{:.2}", lo),
                    (Some(lo), Some(hi)) => format!("{:.2}..{:.2}", lo, hi),
                    _ => "implicit(1.0)".to_string(),
                };
                println!(
                    "Integrity: {} | Alpha/DoRA: {} (alpha/r={}, .alpha={}, dora={}) | Token modules inspected: {}",
                    integrity.status,
                    integrity.alpha_summary.status,
                    alpha_scale_str,
                    integrity.alpha_summary.alpha_tensor_count,
                    integrity.alpha_summary.dora_tensor_count,
                    integrity.embedding_entries.len()
                );
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
            let json_mode = args[2..].iter().any(|a| a == "--json");
            let pos: Vec<&String> = args[2..].iter().filter(|a| !a.starts_with("--")).collect();
            if pos.len() < 2 {
                print_usage();
                process::exit(1);
            }
            let source_a = pos[0];
            let source_b = pos[1];
            let (layers_a, alphas_a) = load_layers_and_alphas_or_exit(source_a);
            let (layers_b, alphas_b) = load_layers_and_alphas_or_exit(source_b);
            let summary = diff_adapters(&layers_a, &alphas_a, &layers_b, &alphas_b);

            if json_mode {
                println!("{}", render_diff_json(&summary));
            } else {
                println!("{}", render_diff_table(&summary));
            }
        }
        "card" => {
            let mut output_opt: Option<String> = None;
            let mut config_opt: Option<String> = None;
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
            let arch = inspect_adapter_architecture(&layers, &alphas);
            let summary = audit_adapter(&layers);
            let integrity = audit_adapter_integrity(
                &layers,
                &tensors,
                &alphas,
                &dora_scales,
                peft_config.as_ref(),
            );
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
            if args.len() < 4 {
                print_usage();
                process::exit(1);
            }
            let input_path = &args[2];
            let output_path = &args[3];
            let mut policy = CompressionPolicy::AdaptiveEnergy {
                target_energy: 0.99,
                min_rank: 1,
                max_rank: 256,
            };
            let mut peft_config_path: Option<String> = None;
            let mut target_dtype: Option<DType> = None;
            let mut target_format: Option<String> = None;
            let mut bake_alpha = false;
            let mut rslora = false;
            let mut default_alpha: Option<f64> = None;
            let mut exclude_patterns: Vec<String> = Vec::new();

            let mut i = 4;
            while i < args.len() {
                match args[i].as_str() {
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
                        policy = CompressionPolicy::FixedRank(k);
                        i += 2;
                    }
                    "--energy" if i + 1 < args.len() => {
                        let e: f64 = args[i + 1].parse().expect("Invalid --energy float");
                        policy = CompressionPolicy::AdaptiveEnergy {
                            target_energy: e,
                            min_rank: 1,
                            max_rank: 256,
                        };
                        i += 2;
                    }
                    "--mp-denoise" => {
                        policy = CompressionPolicy::MarchenkoPastur {
                            min_rank: 1,
                            max_rank: 256,
                        };
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
                        let a: f64 = args[i + 1].parse().expect("Invalid --alpha float");
                        default_alpha = Some(a);
                        bake_alpha = true;
                        i += 2;
                    }
                    "--format" if i + 1 < args.len() => {
                        target_format = Some(args[i + 1].clone());
                        i += 2;
                    }
                    "--peft-config" if i + 1 < args.len() => {
                        peft_config_path = Some(args[i + 1].clone());
                        i += 2;
                    }
                    "--dtype" if i + 1 < args.len() => {
                        target_dtype =
                            Some(DType::from_name(&args[i + 1]).expect("Invalid --dtype (use f16, bf16, or f32)"));
                        i += 2;
                    }
                    _ => i += 1,
                }
            }

            let (mut layers, alphas, dora_scales) = load_layers_alphas_dora_or_exit(input_path);
            if !exclude_patterns.is_empty() {
                layers.retain(|l| !exclude_patterns.iter().any(|p| l.base_name.contains(p.as_str())));
            }
            let default_r = layers.first().map(|l| l.rank()).unwrap_or(16);
            if bake_alpha || !alphas.is_empty() {
                bake_adapter_scaling(&mut layers, &alphas, default_alpha, rslora);
            }
            let mut res = compress_adapter(&layers, policy);
            if let Some(ref fmt) = target_format {
                convert_layers_key_format(&mut res.layers, fmt).expect("Invalid --format");
            }
            if let Some(dt) = target_dtype {
                cast_layers_dtype(&mut res.layers, dt);
            }
            let final_bytes: usize = res.layers.iter().map(|l| l.raw_byte_size()).sum();
            let mut out_map = layers_to_tensor_map(&res.layers);
            if !alphas.is_empty() {
                insert_unit_alpha_tensors(&mut out_map, &res.layers);
            }
            if !dora_scales.is_empty() {
                insert_dora_scales(&mut out_map, &dora_scales, target_format.as_deref(), target_dtype);
            }
            write_safetensors(Path::new(output_path), &out_map).expect("Failed to write output .safetensors");

            if let Some(cfg_path) = &peft_config_path {
                write_peft_rank_pattern_config(Path::new(cfg_path), &res.rank_pattern, default_r)
                    .expect("Failed to write PEFT adapter_config.json");
            }

            let reduction = (1.0 - (final_bytes as f64) / (res.total_original_bytes.max(1) as f64)) * 100.0;
            println!(
                "Compressed {} -> {} ({} layers): {} bytes -> {} bytes (-{:.1}%), mean retained spectral energy: {:.2}%",
                input_path,
                output_path,
                res.layers.len(),
                res.total_original_bytes,
                final_bytes,
                reduction,
                res.overall_retained_energy * 100.0
            );
        }
        "extract" => {
            if args.len() < 5 {
                print_usage();
                process::exit(1);
            }
            let base_path = &args[2];
            let tuned_path = &args[3];
            let output_path = &args[4];
            let mut policy = CompressionPolicy::FixedRank(16);
            let mut target_format = "peft".to_string();
            let mut target_dtype: Option<DType> = None;
            let mut peft_config_path: Option<String> = None;

            let mut i = 5;
            while i < args.len() {
                match args[i].as_str() {
                    "--rank" if i + 1 < args.len() => {
                        let k: usize = args[i + 1].parse().expect("Invalid --rank integer");
                        policy = CompressionPolicy::FixedRank(k);
                        i += 2;
                    }
                    "--energy" if i + 1 < args.len() => {
                        let e: f64 = args[i + 1].parse().expect("Invalid --energy float");
                        policy = CompressionPolicy::AdaptiveEnergy {
                            target_energy: e,
                            min_rank: 1,
                            max_rank: 128,
                        };
                        i += 2;
                    }
                    "--mp-denoise" => {
                        policy = CompressionPolicy::MarchenkoPastur {
                            min_rank: 1,
                            max_rank: 128,
                        };
                        i += 1;
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
                let default_r = res.layers.first().map(|l| l.rank()).unwrap_or(16);
                write_peft_rank_pattern_config(Path::new(cfg_path), &res.rank_pattern, default_r)
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
                let sarif_str = build_merge_check_sarif(&summary, "0.9.0");
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
                        target_dtype =
                            Some(DType::from_name(&args[i + 1]).expect("Invalid --dtype (use f16, bf16, or f32)"));
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
            let mut combined_dora: BTreeMap<String, (String, Tensor2D)> = BTreeMap::new();
            for p in input_paths {
                let (mut l, alphas, dora) = load_layers_alphas_dora_or_exit(p);
                if !alphas.is_empty() {
                    any_alphas = true;
                }
                for (k, v) in dora {
                    combined_dora.entry(k).or_insert(v);
                }
                if bake_alpha || !alphas.is_empty() {
                    bake_adapter_scaling(&mut l, &alphas, default_alpha, rslora);
                }
                adapters.push(l);
            }

            let default_rank = adapters[0].first().map(|l| l.rank()).unwrap_or(16);
            let rank = rank_opt.unwrap_or(default_rank);
            let n_adapters = adapters.len();
            let weights = custom_weights.unwrap_or_else(|| {
                if mode == "linear" || mode == "ties" {
                    vec![1.0 / (n_adapters as f64); n_adapters]
                } else {
                    vec![1.0; n_adapters]
                }
            });

            let mut merged = merge_multiple_adapters(
                &adapters,
                &weights,
                &mode,
                rank,
                &block_rules,
                preserve_norm,
            )
            .expect("Merge failed");

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
        _ => {
            print_usage();
            process::exit(1);
        }
    }
}

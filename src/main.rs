use std::collections::BTreeMap;
use std::env;
use std::path::Path;
use std::process;

use loraspectra::audit::audit_adapter;
use loraspectra::compress::{compress_adapter, CompressionPolicy};
use loraspectra::merge::{
    analyze_layer_interference, merge_multiple_adapters, parse_layer_weights, BlockWeightRule,
};
use loraspectra::safetensors::{
    bake_adapter_scaling, cast_layers_dtype, extract_lora_alphas, extract_lora_layers,
    insert_unit_alpha_tensors, inspect_adapter_architecture, layers_to_tensor_map,
    read_safetensors, write_peft_rank_pattern_config, write_safetensors, DType, LoraLayer,
};

fn print_usage() {
    eprintln!(
        "loraspectra v0.4.0 — Spectral SVD Auditor, Dynamic Rank Compressor & Orthogonal Merger for LoRA (.safetensors)

USAGE:
  loraspectra info <adapter.safetensors> [--json]
  loraspectra audit <adapter.safetensors> [--json]
  loraspectra compress <input.safetensors> <output.safetensors> [--energy 0.99 | --rank K | --mp-denoise] [--bake-alpha] [--rslora] [--alpha A] [--dtype f16|bf16|f32] [--peft-config <adapter_config.json>]
  loraspectra interference <adapter_a.safetensors> <adapter_b.safetensors>
  loraspectra merge <in_1.safetensors> <in_2.safetensors> [in_3.safetensors ...] <output.safetensors> [--mode ortho-svd|linear|ties] [--rank K] [--weights W1:W2[:W3...]] [--layer-weights \"attn=1.0:0.2,mlp=0.2:1.0\"] [--preserve-norm] [--bake-alpha] [--rslora] [--dtype f16|bf16|f32]"
    );
}

fn load_layers_and_alphas_or_exit(path_str: &str) -> (Vec<LoraLayer>, BTreeMap<String, f64>) {
    let path = Path::new(path_str);
    let tensors = match read_safetensors(path) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("Error reading {}: {}", path_str, e);
            process::exit(1);
        }
    };
    let alphas = extract_lora_alphas(&tensors);
    match extract_lora_layers(&tensors) {
        Ok(layers) if !layers.is_empty() => (layers, alphas),
        Ok(_) => {
            eprintln!("No matched (lora_A, lora_B) 2D tensor pairs found in {}", path_str);
            process::exit(1);
        }
        Err(e) => {
            eprintln!("Error extracting LoRA layers from {}: {}", path_str, e);
            process::exit(1);
        }
    }
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
            if args.len() < 3 {
                print_usage();
                process::exit(1);
            }
            let json_mode = args.iter().any(|a| a == "--json");
            let (layers, alphas) = load_layers_and_alphas_or_exit(&args[2]);
            let info = inspect_adapter_architecture(&layers, &alphas);
            let summary = audit_adapter(&layers);

            if json_mode {
                println!("{{");
                println!("  \"naming_format\": \"{}\",", info.naming_format);
                println!("  \"inferred_architecture\": \"{}\",", info.inferred_architecture);
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
                println!("Checkpoint:             {}", args[2]);
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
            if args.len() < 3 {
                print_usage();
                process::exit(1);
            }
            let json_mode = args.iter().any(|a| a == "--json");
            let layers = load_layers_or_exit(&args[2]);
            let summary = audit_adapter(&layers);

            if json_mode {
                println!("{{");
                println!("  \"layer_count\": {},", summary.layers.len());
                println!("  \"total_nominal_bytes\": {},", summary.total_nominal_bytes);
                println!("  \"total_optimal_99_bytes\": {},", summary.total_optimal_99_bytes);
                println!("  \"compressible_byte_ratio\": {:.6},", summary.compressible_byte_ratio);
                println!("  \"mean_nominal_rank\": {:.2},", summary.mean_nominal_rank);
                println!("  \"mean_effective_rank\": {:.2},", summary.mean_effective_rank);
                println!("  \"mean_stable_rank\": {:.2},", summary.mean_stable_rank);
                println!("  \"mean_rank_at_99_energy\": {:.2},", summary.mean_rank_at_99_energy);
                println!("  \"mean_rank_marchenko_pastur\": {:.2}", summary.mean_rank_mp);
                println!("}}");
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
            let mut bake_alpha = false;
            let mut rslora = false;
            let mut default_alpha: Option<f64> = None;

            let mut i = 4;
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

            let (mut layers, alphas) = load_layers_and_alphas_or_exit(input_path);
            let default_r = layers.first().map(|l| l.rank()).unwrap_or(16);
            if bake_alpha || !alphas.is_empty() {
                bake_adapter_scaling(&mut layers, &alphas, default_alpha, rslora);
            }
            let mut res = compress_adapter(&layers, policy);
            if let Some(dt) = target_dtype {
                cast_layers_dtype(&mut res.layers, dt);
            }
            let final_bytes: usize = res.layers.iter().map(|l| l.raw_byte_size()).sum();
            let mut out_map = layers_to_tensor_map(&res.layers);
            if !alphas.is_empty() {
                insert_unit_alpha_tensors(&mut out_map, &res.layers);
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
        "interference" => {
            if args.len() < 4 {
                print_usage();
                process::exit(1);
            }
            let layers_a = load_layers_or_exit(&args[2]);
            let layers_b = load_layers_or_exit(&args[3]);
            println!(
                "{:<42} {:>9} {:>9} {:>10} {:<18}",
                "LAYER", "COS_U_MAX", "COS_V_MAX", "FROB_COS", "SEVERITY"
            );
            println!("{}", "-".repeat(92));
            for (la, lb) in layers_a.iter().zip(layers_b.iter()) {
                let rep = analyze_layer_interference(la, lb);
                println!(
                    "{:<42} {:>9.4} {:>9.4} {:>10.4} {:<18}",
                    rep.base_name,
                    rep.max_left_principal_cos,
                    rep.max_right_principal_cos,
                    rep.frobenius_cosine,
                    rep.conflict_severity
                );
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
            for p in input_paths {
                let (mut l, alphas) = load_layers_and_alphas_or_exit(p);
                if !alphas.is_empty() {
                    any_alphas = true;
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

            if let Some(dt) = target_dtype {
                cast_layers_dtype(&mut merged, dt);
            }
            let mut out_map = layers_to_tensor_map(&merged);
            if any_alphas {
                insert_unit_alpha_tensors(&mut out_map, &merged);
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

use std::env;
use std::path::Path;
use std::process;

use loraspectra::audit::audit_adapter;
use loraspectra::compress::{compress_adapter, CompressionPolicy};
use loraspectra::merge::{
    analyze_layer_interference, merge_adapters_with_options, parse_layer_weights, BlockWeightRule,
    MergeStrategy,
};
use loraspectra::safetensors::{
    cast_layers_dtype, extract_lora_layers, layers_to_tensor_map, read_safetensors,
    write_peft_rank_pattern_config, write_safetensors, DType,
};

fn print_usage() {
    eprintln!(
        "loraspectra v0.3.0 — Spectral SVD Auditor, Dynamic Rank Compressor & Orthogonal Merger for LoRA (.safetensors)

USAGE:
  loraspectra audit <adapter.safetensors> [--json]
  loraspectra compress <input.safetensors> <output.safetensors> [--energy 0.99 | --rank K | --mp-denoise] [--dtype f16|bf16|f32] [--peft-config <adapter_config.json>]
  loraspectra interference <adapter_a.safetensors> <adapter_b.safetensors>
  loraspectra merge <adapter_a.safetensors> <adapter_b.safetensors> <output.safetensors> [--mode ortho-svd|linear|ties] [--rank K] [--weights W1:W2] [--layer-weights \"attn=1.0:0.2,mlp=0.2:1.0\"] [--preserve-norm] [--dtype f16|bf16|f32]"
    );
}

fn load_layers_or_exit(path_str: &str) -> Vec<loraspectra::safetensors::LoraLayer> {
    let path = Path::new(path_str);
    let tensors = match read_safetensors(path) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("Error reading {}: {}", path_str, e);
            process::exit(1);
        }
    };
    match extract_lora_layers(&tensors) {
        Ok(layers) if !layers.is_empty() => layers,
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

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        print_usage();
        process::exit(1);
    }

    match args[1].as_str() {
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

            let layers = load_layers_or_exit(input_path);
            let default_r = layers.first().map(|l| l.rank()).unwrap_or(16);
            let mut res = compress_adapter(&layers, policy);
            if let Some(dt) = target_dtype {
                cast_layers_dtype(&mut res.layers, dt);
            }
            let final_bytes: usize = res.layers.iter().map(|l| l.raw_byte_size()).sum();
            let out_map = layers_to_tensor_map(&res.layers);
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
            let layers_a = load_layers_or_exit(&args[2]);
            let layers_b = load_layers_or_exit(&args[3]);
            let output_path = &args[4];
            let default_rank = layers_a.first().map(|l| l.rank()).unwrap_or(16);

            let mut mode = "ortho-svd".to_string();
            let mut rank = default_rank;
            let mut custom_weights: Option<(f64, f64)> = None;
            let mut block_rules: Vec<BlockWeightRule> = Vec::new();
            let mut preserve_norm = false;
            let mut target_dtype: Option<DType> = None;
            let mut i = 5;
            while i < args.len() {
                match args[i].as_str() {
                    "--mode" if i + 1 < args.len() => {
                        mode = args[i + 1].clone();
                        i += 2;
                    }
                    "--rank" if i + 1 < args.len() => {
                        rank = args[i + 1].parse().expect("Invalid --rank");
                        i += 2;
                    }
                    "--weights" if i + 1 < args.len() => {
                        let pair = &args[i + 1];
                        let (w1_s, w2_s) = pair.split_once(':').expect("Expected --weights W1:W2");
                        let w1: f64 = w1_s.parse().expect("Invalid W1 float");
                        let w2: f64 = w2_s.parse().expect("Invalid W2 float");
                        custom_weights = Some((w1, w2));
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
                    "--dtype" if i + 1 < args.len() => {
                        target_dtype =
                            Some(DType::from_name(&args[i + 1]).expect("Invalid --dtype (use f16, bf16, or f32)"));
                        i += 2;
                    }
                    _ => i += 1,
                }
            }

            let base_strategy = match mode.as_str() {
                "linear" => {
                    let (w1, w2) = custom_weights.unwrap_or((0.5, 0.5));
                    MergeStrategy::Linear {
                        w1,
                        w2,
                        target_rank: rank,
                    }
                }
                "ties" => {
                    let (w1, w2) = custom_weights.unwrap_or((0.5, 0.5));
                    MergeStrategy::Ties {
                        w1,
                        w2,
                        density: 0.6,
                        target_rank: rank,
                    }
                }
                _ => {
                    let (w1, w2) = custom_weights.unwrap_or((1.0, 1.0));
                    MergeStrategy::OrthoSvd {
                        w1,
                        w2,
                        target_rank: rank,
                    }
                }
            };

            let mut merged = merge_adapters_with_options(
                &layers_a,
                &layers_b,
                base_strategy,
                &block_rules,
                preserve_norm,
            )
            .expect("Merge failed");
            if let Some(dt) = target_dtype {
                cast_layers_dtype(&mut merged, dt);
            }
            let out_map = layers_to_tensor_map(&merged);
            let bytes = write_safetensors(Path::new(output_path), &out_map).expect("Write failed");
            println!(
                "Merged {} layers using mode '{}' (target rank {}, {} block rules, preserve_norm={}) -> {} ({} bytes)",
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

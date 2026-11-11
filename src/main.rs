use std::env;
use std::path::Path;
use std::process;

use loraspectra::audit::audit_adapter;
use loraspectra::compress::{compress_adapter, CompressionPolicy};
use loraspectra::merge::{analyze_layer_interference, merge_adapters, MergeStrategy};
use loraspectra::safetensors::{
    extract_lora_layers, layers_to_tensor_map, read_safetensors, write_safetensors,
};

fn print_usage() {
    eprintln!(
        "loraspectra v0.1.0 — Spectral SVD Auditor, Dynamic Rank Compressor & Orthogonal Merger for LoRA (.safetensors)

USAGE:
  loraspectra audit <adapter.safetensors> [--json]
  loraspectra compress <input.safetensors> <output.safetensors> [--energy 0.99 | --rank K]
  loraspectra interference <adapter_a.safetensors> <adapter_b.safetensors>
  loraspectra merge <adapter_a.safetensors> <adapter_b.safetensors> <output.safetensors> [--mode ortho-svd|linear|ties] [--rank K]"
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
                println!("  \"mean_rank_at_99_energy\": {:.2}", summary.mean_rank_at_99_energy);
                println!("}}");
            } else {
                println!(
                    "{:<42} {:>6} {:>8} {:>8} {:>7} {:>8} {:<18}",
                    "LAYER", "RANK", "R_EFF", "R_STABLE", "K_99%", "NORM_F", "STATUS"
                );
                println!("{}", "-".repeat(103));
                for r in &summary.layers {
                    println!(
                        "{:<42} {:>6} {:>8.2} {:>8.2} {:>7} {:>8.4} {:<18}",
                        r.base_name,
                        r.nominal_rank,
                        r.effective_rank,
                        r.stable_rank,
                        r.rank_at_99_energy,
                        r.frobenius_norm,
                        r.health.as_str()
                    );
                }
                println!("{}", "-".repeat(103));
                println!(
                    "Summary: {} layers | Mean rank {:.1} -> k_99% {:.1} (r_eff {:.2}) | Potential size reduction at 99% energy: -{:.1}%",
                    summary.layers.len(),
                    summary.mean_nominal_rank,
                    summary.mean_rank_at_99_energy,
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
                    _ => i += 1,
                }
            }

            let layers = load_layers_or_exit(input_path);
            let res = compress_adapter(&layers, policy);
            let out_map = layers_to_tensor_map(&res.layers);
            write_safetensors(Path::new(output_path), &out_map).expect("Failed to write output .safetensors");

            let reduction = (1.0 - (res.total_compressed_bytes as f64) / (res.total_original_bytes.max(1) as f64)) * 100.0;
            println!(
                "Compressed {} -> {} ({} layers): {} bytes -> {} bytes (-{:.1}%), mean retained spectral energy: {:.2}%",
                input_path,
                output_path,
                res.layers.len(),
                res.total_original_bytes,
                res.total_compressed_bytes,
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
                    _ => i += 1,
                }
            }

            let strategy = match mode.as_str() {
                "linear" => MergeStrategy::Linear {
                    w1: 0.5,
                    w2: 0.5,
                    target_rank: rank,
                },
                "ties" => MergeStrategy::Ties {
                    w1: 0.5,
                    w2: 0.5,
                    density: 0.6,
                    target_rank: rank,
                },
                _ => MergeStrategy::OrthoSvd {
                    w1: 1.0,
                    w2: 1.0,
                    target_rank: rank,
                },
            };

            let merged = merge_adapters(&layers_a, &layers_b, strategy).expect("Merge failed");
            let out_map = layers_to_tensor_map(&merged);
            let bytes = write_safetensors(Path::new(output_path), &out_map).expect("Write failed");
            println!(
                "Merged {} layers using mode '{}' (target rank {}) -> {} ({} bytes)",
                merged.len(),
                mode,
                rank,
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

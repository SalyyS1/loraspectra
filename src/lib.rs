pub mod audit;
pub mod card;
pub mod compress;
pub mod diff;
pub mod linalg;
pub mod merge;
pub mod merge_check;
pub mod quant;
pub mod remote;
pub mod safetensors;
pub mod sarif;
pub mod vera;

/// Deterministic SplitMix64 pseudo-random generator for reproducible synthetic LoRA benchmarks and tests.
#[derive(Debug, Clone)]
pub struct Rng64 {
    state: u64,
}

impl Rng64 {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    #[inline]
    pub fn next_f64(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64) * (1.0 / ((1u64 << 53) as f64))
    }

    /// Standard normal `N(0, 1)` via Box-Muller transform.
    #[inline]
    pub fn next_gaussian(&mut self) -> f64 {
        let u1 = self.next_f64().max(1e-15);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;
    use crate::audit::{audit_adapter, audit_adapter_integrity, audit_layer, LayerHealth};
    use crate::compress::{compress_layer, extract_lora_from_dense_models, CompressionPolicy};
    use crate::linalg::{
        jacobi_svd, matmul, qr_core_singular_values, thin_qr_columns,
    };
    use crate::merge::{
        analyze_layer_interference, merge_adapters_with_block_weights, merge_adapters_with_options,
        merge_layer, merge_multiple_adapters, parse_layer_weights, MergeStrategy,
    };
    use crate::safetensors::{
        bake_adapter_scaling, bf16_to_f32, cast_layers_dtype, convert_layers_key_format,
        extract_dora_scales, extract_lora_alphas, extract_lora_layers, f16_to_f32, f32_to_bf16,
        f32_to_f16, insert_dora_scales, insert_unit_alpha_tensors, inspect_adapter_architecture,
        layers_to_tensor_map, read_safetensors, write_peft_rank_pattern_config, write_safetensors,
        DType, LoraLayer, Tensor2D,
    };

    fn random_matrix(rows: usize, cols: usize, rng: &mut Rng64, dtype: DType) -> Tensor2D {
        let mut t = Tensor2D::zeros(rows, cols, dtype);
        let scale = 1.0 / (cols as f64).sqrt();
        for v in &mut t.data {
            *v = rng.next_gaussian() * scale;
        }
        t
    }

    #[test]
    fn test_f16_and_bf16_roundtrip_accuracy() {
        let samples = [-12.5_f32, -1.0, -0.03125, 0.0, 0.0078125, 0.5, 1.0, 3.14159, 42.0];
        for &x in &samples {
            let h = f32_to_f16(x);
            let x_f16 = f16_to_f32(h);
            assert!(
                (x - x_f16).abs() <= x.abs() * 1e-2 + 1e-4,
                "f16 roundtrip failed for {}: got {}",
                x,
                x_f16
            );

            let b = f32_to_bf16(x);
            let x_bf16 = bf16_to_f32(b);
            assert!(
                (x - x_bf16).abs() <= x.abs() * 1e-2 + 1e-4,
                "bf16 roundtrip failed for {}: got {}",
                x,
                x_bf16
            );
        }
    }

    #[test]
    fn test_safetensors_read_write_roundtrip_all_dtypes() {
        let mut rng = Rng64::new(101);
        for dtype in [DType::F32, DType::F16, DType::BF16] {
            let layer = LoraLayer {
                base_name: "model.layers.0.self_attn.q_proj".to_string(),
                a_key: "model.layers.0.self_attn.q_proj.lora_A.weight".to_string(),
                b_key: "model.layers.0.self_attn.q_proj.lora_B.weight".to_string(),
                a: random_matrix(8, 32, &mut rng, dtype),
                b: random_matrix(32, 8, &mut rng, dtype),
            };
            let map = layers_to_tensor_map(&[layer.clone()]);
            let tmp_path = std::env::temp_dir().join(format!("test_lora_{}.safetensors", dtype.as_str()));
            let written = write_safetensors(&tmp_path, &map).expect("write safetensors");
            assert!(written > 64);

            let loaded_map = read_safetensors(&tmp_path).expect("read safetensors");
            let loaded_layers = extract_lora_layers(&loaded_map).expect("extract layers");
            assert_eq!(loaded_layers.len(), 1);
            assert_eq!(loaded_layers[0].rank(), 8);
            assert_eq!(loaded_layers[0].d_in(), 32);
            assert_eq!(loaded_layers[0].d_out(), 32);

            let tol = match dtype {
                DType::F32 => 1e-6,
                DType::F16 => 2e-3,
                DType::BF16 => 1e-2,
                DType::I8 | DType::U8 => 1.0,
            };
            for (&orig, &dec) in layer.a.data.iter().zip(loaded_layers[0].a.data.iter()) {
                assert!((orig - dec).abs() <= tol, "mismatch: {} vs {}", orig, dec);
            }
            let _ = std::fs::remove_file(&tmp_path);
        }
    }

    #[test]
    fn test_qr_core_svd_matches_full_dense_svd_to_machine_precision() {
        let mut rng = Rng64::new(4242);
        let d_out = 96;
        let d_in = 80;
        let r = 12;
        let b = random_matrix(d_out, r, &mut rng, DType::F32);
        let a = random_matrix(r, d_in, &mut rng, DType::F32);

        let qr_sigma = qr_core_singular_values(&b, &a);
        let dense_dw = matmul(&b, &a);
        let dense_svd = jacobi_svd(&dense_dw);

        for i in 0..r {
            let diff = (qr_sigma[i] - dense_svd.sigma[i]).abs();
            assert!(
                diff < 1e-11,
                "Singular value {} mismatch: qr={} vs dense={} (diff={})",
                i,
                qr_sigma[i],
                dense_svd.sigma[i],
                diff
            );
        }
    }

    #[test]
    fn test_eckart_young_optimal_truncation_theorem() {
        let mut rng = Rng64::new(777);
        let d_out = 128;
        let d_in = 128;
        let r = 16;
        let layer = LoraLayer {
            base_name: "layers.0.mlp.down_proj".to_string(),
            a_key: "layers.0.mlp.down_proj.lora_A.weight".to_string(),
            b_key: "layers.0.mlp.down_proj.lora_B.weight".to_string(),
            a: random_matrix(r, d_in, &mut rng, DType::F32),
            b: random_matrix(d_out, r, &mut rng, DType::F32),
        };

        let sigma = qr_core_singular_values(&layer.b, &layer.a);
        let k = 6;
        let (compressed, metric) = compress_layer(&layer, CompressionPolicy::FixedRank(k));
        assert_eq!(compressed.rank(), k);

        // Compute exact dense difference ||B*A - B_k*A_k||_F
        let dw_orig = matmul(&layer.b, &layer.a);
        let dw_comp = matmul(&compressed.b, &compressed.a);
        let mut err_sq = 0.0;
        for (&x, &y) in dw_orig.data.iter().zip(dw_comp.data.iter()) {
            let d = x - y;
            err_sq += d * d;
        }
        let actual_frob_err = err_sq.sqrt();
        let theoretical_frob_err = sigma[k..].iter().map(|&s| s * s).sum::<f64>().sqrt();

        assert!(
            (actual_frob_err - theoretical_frob_err).abs() < 1e-10,
            "Eckart-Young theorem violation: actual={} vs theoretical={}",
            actual_frob_err,
            theoretical_frob_err
        );

        let total_e: f64 = sigma.iter().map(|&s| s * s).sum();
        let expected_rel_err = theoretical_frob_err / total_e.sqrt();
        assert!((metric.relative_frobenius_error - expected_rel_err).abs() < 1e-10);
    }

    #[test]
    fn test_audit_detects_rank_collapse_and_optimal_layers() {
        let mut rng = Rng64::new(999);
        let d = 64;
        let r = 16;

        // 1. Construct a rank-1 collapsed layer embedded in rank-16 matrices
        let u_vec = random_matrix(d, 1, &mut rng, DType::F32);
        let v_vec = random_matrix(1, d, &mut rng, DType::F32);
        let mut b_col = Tensor2D::zeros(d, r, DType::F32);
        let mut a_col = Tensor2D::zeros(r, d, DType::F32);
        for i in 0..d {
            b_col.set(i, 0, u_vec.get(i, 0) * 5.0);
            a_col.set(0, i, v_vec.get(0, i) * 5.0);
        }
        let collapsed_layer = LoraLayer {
            base_name: "layers.1.self_attn.q_proj".to_string(),
            a_key: "layers.1.self_attn.q_proj.lora_A.weight".to_string(),
            b_key: "layers.1.self_attn.q_proj.lora_B.weight".to_string(),
            a: a_col,
            b: b_col,
        };
        let rep_collapsed = audit_layer(&collapsed_layer);
        assert_eq!(rep_collapsed.health, LayerHealth::RankCollapsed);
        assert_eq!(rep_collapsed.rank_at_99_energy, 1);
        assert!((rep_collapsed.effective_rank - 1.0).abs() < 1e-6);

        // 2. Construct an isotropic full-rank layer
        let full_layer = LoraLayer {
            base_name: "layers.1.self_attn.v_proj".to_string(),
            a_key: "layers.1.self_attn.v_proj.lora_A.weight".to_string(),
            b_key: "layers.1.self_attn.v_proj.lora_B.weight".to_string(),
            a: random_matrix(r, d, &mut rng, DType::F32),
            b: random_matrix(d, r, &mut rng, DType::F32),
        };
        let rep_full = audit_layer(&full_layer);
        assert_eq!(rep_full.health, LayerHealth::Optimal);
        assert!(rep_full.rank_at_99_energy >= 12);
    }

    #[test]
    fn test_ortho_svd_merge_eliminates_destructive_subspace_cancellation() {
        let mut rng = Rng64::new(2026);
        let d = 128;
        let r = 8;

        // Build orthonormal bases U (d x 2r) and V (d x 2r)
        let raw_u = random_matrix(d, 2 * r, &mut rng, DType::F32);
        let (u_cols, _) = thin_qr_columns(&raw_u);
        let raw_v = random_matrix(d, 2 * r, &mut rng, DType::F32);
        let (v_cols, _) = thin_qr_columns(&raw_v);

        // Adapter 1 uses modes 0..r with singular values 4.0
        let mut b1 = Tensor2D::zeros(d, r, DType::F32);
        let mut a1 = Tensor2D::zeros(r, d, DType::F32);
        for j in 0..r {
            for row in 0..d {
                b1.set(row, j, u_cols[j][row] * 2.0);
                a1.set(j, row, v_cols[j][row] * 2.0);
            }
        }

        // Adapter 2 shares modes 0..4 with OPPOSITE sign (destructive conflict!)
        // and uses novel orthogonal modes r..(r+4) for its remaining 4 dimensions
        let mut b2 = Tensor2D::zeros(d, r, DType::F32);
        let mut a2 = Tensor2D::zeros(r, d, DType::F32);
        for j in 0..4 {
            for row in 0..d {
                b2.set(row, j, -u_cols[j][row] * 2.0);
                a2.set(j, row, v_cols[j][row] * 2.0);
            }
        }
        for j in 4..r {
            let novel_idx = r + (j - 4);
            for row in 0..d {
                b2.set(row, j, u_cols[novel_idx][row] * 2.0);
                a2.set(j, row, v_cols[novel_idx][row] * 2.0);
            }
        }

        let l1 = LoraLayer {
            base_name: "attn.o_proj".to_string(),
            a_key: "attn.o_proj.lora_A.weight".to_string(),
            b_key: "attn.o_proj.lora_B.weight".to_string(),
            a: a1,
            b: b1,
        };
        let l2 = LoraLayer {
            base_name: "attn.o_proj".to_string(),
            a_key: "attn.o_proj.lora_A.weight".to_string(),
            b_key: "attn.o_proj.lora_B.weight".to_string(),
            a: a2,
            b: b2,
        };

        let interf = analyze_layer_interference(&l1, &l2);
        assert_eq!(interf.conflict_severity, "HIGH_INTERFERENCE");
        assert!(interf.frobenius_cosine < -0.4);

        let linear_merged = merge_layer(
            &l1,
            &l2,
            MergeStrategy::Linear {
                w1: 1.0,
                w2: 1.0,
                target_rank: 12,
            },
        );
        let ortho_merged = merge_layer(
            &l1,
            &l2,
            MergeStrategy::OrthoSvd {
                w1: 1.0,
                w2: 1.0,
                target_rank: 12,
            },
        );

        // Evaluate Task 1 preservation on Task 1's right singular basis v_cols[0]
        // Under Linear Merge, mode 0 cancels to ZERO! Under OrthoSvd, mode 0 is preserved!
        let dw1 = matmul(&l1.b, &l1.a);
        let dw_lin = matmul(&linear_merged.b, &linear_merged.a);
        let dw_ortho = matmul(&ortho_merged.b, &ortho_merged.a);

        let mut v0 = Tensor2D::zeros(d, 1, DType::F32);
        for row in 0..d {
            v0.set(row, 0, v_cols[0][row]);
        }

        let y1 = matmul(&dw1, &v0);
        let y_lin = matmul(&dw_lin, &v0);
        let y_ortho = matmul(&dw_ortho, &v0);

        let err_lin: f64 = y1
            .data
            .iter()
            .zip(y_lin.data.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();
        let err_ortho: f64 = y1
            .data
            .iter()
            .zip(y_ortho.data.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();

        assert!(
            err_lin > 3.5,
            "Linear merge should suffer cancellation on conflicting mode 0, got {}",
            err_lin
        );
        assert!(
            err_ortho < 1e-10,
            "OrthoSvd merge should preserve Task 1 mode 0 with near-zero error, got {}",
            err_ortho
        );
    }

    #[test]
    fn test_marchenko_pastur_automatic_noise_floor_recovery() {
        let mut rng = Rng64::new(31415);
        let d = 128;
        let r = 16;
        let true_signal_k = 4;

        let raw_u = random_matrix(d, r, &mut rng, DType::F32);
        let (u_cols, _) = thin_qr_columns(&raw_u);
        let raw_v = random_matrix(d, r, &mut rng, DType::F32);
        let (v_cols, _) = thin_qr_columns(&raw_v);

        // Modes 0..4 have strong signal singular values (8.0, 6.0, 5.0, 4.0);
        // modes 4..16 are stochastic SGD noise floor (~0.15).
        let mut b = Tensor2D::zeros(d, r, DType::F32);
        let mut a = Tensor2D::zeros(r, d, DType::F32);
        for j in 0..r {
            let sigma = if j < true_signal_k {
                8.0 - (j as f64) * 1.2
            } else {
                0.15 + 0.02 * rng.next_f64()
            };
            let s_sqrt = sigma.sqrt();
            for row in 0..d {
                b.set(row, j, u_cols[j][row] * s_sqrt);
                a.set(j, row, v_cols[j][row] * s_sqrt);
            }
        }

        let layer = LoraLayer {
            base_name: "model.layers.0.mlp.gate_proj".to_string(),
            a_key: "model.layers.0.mlp.gate_proj.lora_A.weight".to_string(),
            b_key: "model.layers.0.mlp.gate_proj.lora_B.weight".to_string(),
            a,
            b,
        };

        let rep = audit_layer(&layer);
        assert_eq!(rep.rank_marchenko_pastur, true_signal_k);

        let (compressed, metric) = compress_layer(
            &layer,
            CompressionPolicy::MarchenkoPastur {
                min_rank: 1,
                max_rank: 64,
            },
        );
        assert_eq!(compressed.rank(), true_signal_k);
        assert_eq!(metric.compressed_rank, true_signal_k);
        assert!(metric.retained_energy_ratio > 0.99);
    }

    #[test]
    fn test_diffusers_and_onetrainer_keys_and_peft_rank_pattern_export() {
        let mut rng = Rng64::new(8080);
        let mut map = std::collections::BTreeMap::new();
        map.insert(
            "unet.down_blocks.0.attentions.0.to_q.lora.down.weight".to_string(),
            random_matrix(4, 32, &mut rng, DType::F16),
        );
        map.insert(
            "unet.down_blocks.0.attentions.0.to_q.lora.up.weight".to_string(),
            random_matrix(32, 4, &mut rng, DType::F16),
        );
        map.insert(
            "transformer.blocks.0.attn.to_k_lora.down.weight".to_string(),
            random_matrix(8, 32, &mut rng, DType::F16),
        );
        map.insert(
            "transformer.blocks.0.attn.to_k_lora.up.weight".to_string(),
            random_matrix(32, 8, &mut rng, DType::F16),
        );

        let layers = extract_lora_layers(&map).expect("extract diffusers/onetrainer layers");
        assert_eq!(layers.len(), 2);

        let mut rank_pattern = std::collections::BTreeMap::new();
        rank_pattern.insert("base_model.model.layers.0.self_attn.q_proj".to_string(), 4);
        rank_pattern.insert("base_model.model.layers.0.mlp.down_proj".to_string(), 12);

        let cfg_path = std::env::temp_dir().join("test_peft_adapter_config.json");
        write_peft_rank_pattern_config(&cfg_path, &rank_pattern, 16).expect("write peft config");
        let content = std::fs::read_to_string(&cfg_path).expect("read peft config");
        assert!(content.contains("\"rank_pattern\""));
        assert!(content.contains("\"alpha_pattern\""));
        assert!(content.contains("\"layers.0.self_attn.q_proj\": 4"));
        assert!(content.contains("\"layers.0.mlp.down_proj\": 12"));
        let _ = std::fs::remove_file(&cfg_path);
    }

    #[test]
    fn test_layer_block_weighted_ortho_svd_merge() {
        let mut rng = Rng64::new(9090);
        let d = 32;
        let r = 4;

        let l1_attn = LoraLayer {
            base_name: "layers.0.self_attn.q_proj".to_string(),
            a_key: "layers.0.self_attn.q_proj.lora_A.weight".to_string(),
            b_key: "layers.0.self_attn.q_proj.lora_B.weight".to_string(),
            a: random_matrix(r, d, &mut rng, DType::F32),
            b: random_matrix(d, r, &mut rng, DType::F32),
        };
        let l1_mlp = LoraLayer {
            base_name: "layers.0.mlp.up_proj".to_string(),
            a_key: "layers.0.mlp.up_proj.lora_A.weight".to_string(),
            b_key: "layers.0.mlp.up_proj.lora_B.weight".to_string(),
            a: random_matrix(r, d, &mut rng, DType::F32),
            b: random_matrix(d, r, &mut rng, DType::F32),
        };

        let l2_attn = LoraLayer {
            base_name: "layers.0.self_attn.q_proj".to_string(),
            a_key: "layers.0.self_attn.q_proj.lora_A.weight".to_string(),
            b_key: "layers.0.self_attn.q_proj.lora_B.weight".to_string(),
            a: random_matrix(r, d, &mut rng, DType::F32),
            b: random_matrix(d, r, &mut rng, DType::F32),
        };
        let l2_mlp = LoraLayer {
            base_name: "layers.0.mlp.up_proj".to_string(),
            a_key: "layers.0.mlp.up_proj.lora_A.weight".to_string(),
            b_key: "layers.0.mlp.up_proj.lora_B.weight".to_string(),
            a: random_matrix(r, d, &mut rng, DType::F32),
            b: random_matrix(d, r, &mut rng, DType::F32),
        };

        let rules = parse_layer_weights("self_attn=1.0:0.0,mlp=0.0:1.0").expect("parse rules");
        let merged = merge_adapters_with_block_weights(
            &[l1_attn.clone(), l1_mlp.clone()],
            &[l2_attn.clone(), l2_mlp.clone()],
            MergeStrategy::OrthoSvd {
                w1: 0.5,
                w2: 0.5,
                target_rank: r,
            },
            &rules,
        )
        .expect("block merge");

        let dw_attn_orig = matmul(&l1_attn.b, &l1_attn.a);
        let dw_attn_merged = matmul(&merged[0].b, &merged[0].a);
        let err_attn: f64 = dw_attn_orig
            .data
            .iter()
            .zip(dw_attn_merged.data.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f64>()
            .sqrt();
        assert!(err_attn < 1e-10, "attn should match adapter 1, err={}", err_attn);
    }

    #[test]
    fn test_disjoint_and_heterogeneous_rank_adapter_merge_with_preserve_norm() {
        let mut rng = Rng64::new(5150);
        let d = 48;

        // Adapter 1 has q_proj (r=4) and v_proj (r=4)
        let l1_q = LoraLayer {
            base_name: "layers.0.self_attn.q_proj".to_string(),
            a_key: "layers.0.self_attn.q_proj.lora_A.weight".to_string(),
            b_key: "layers.0.self_attn.q_proj.lora_B.weight".to_string(),
            a: random_matrix(4, d, &mut rng, DType::F32),
            b: random_matrix(d, 4, &mut rng, DType::F32),
        };
        let l1_v = LoraLayer {
            base_name: "layers.0.self_attn.v_proj".to_string(),
            a_key: "layers.0.self_attn.v_proj.lora_A.weight".to_string(),
            b_key: "layers.0.self_attn.v_proj.lora_B.weight".to_string(),
            a: random_matrix(4, d, &mut rng, DType::F32),
            b: random_matrix(d, 4, &mut rng, DType::F32),
        };

        // Adapter 2 has q_proj (r=8, heterogeneous rank!) and down_proj (r=8, disjoint layer!)
        let l2_q = LoraLayer {
            base_name: "layers.0.self_attn.q_proj".to_string(),
            a_key: "layers.0.self_attn.q_proj.lora_A.weight".to_string(),
            b_key: "layers.0.self_attn.q_proj.lora_B.weight".to_string(),
            a: random_matrix(8, d, &mut rng, DType::F32),
            b: random_matrix(d, 8, &mut rng, DType::F32),
        };
        let l2_down = LoraLayer {
            base_name: "layers.0.mlp.down_proj".to_string(),
            a_key: "layers.0.mlp.down_proj.lora_A.weight".to_string(),
            b_key: "layers.0.mlp.down_proj.lora_B.weight".to_string(),
            a: random_matrix(8, d, &mut rng, DType::F32),
            b: random_matrix(d, 8, &mut rng, DType::F32),
        };

        let s1_max = qr_core_singular_values(&l1_q.b, &l1_q.a)[0];
        let s2_max = qr_core_singular_values(&l2_q.b, &l2_q.a)[0];
        let bound = s1_max.max(s2_max);

        let mut merged = merge_adapters_with_options(
            &[l1_q, l1_v],
            &[l2_q, l2_down],
            MergeStrategy::OrthoSvd {
                w1: 2.0,
                w2: 2.0,
                target_rank: 6,
            },
            &[],
            true,
        )
        .expect("disjoint heterogeneous merge");

        assert_eq!(merged.len(), 3);
        let s_merged_q = qr_core_singular_values(&merged[0].b, &merged[0].a)[0];
        assert!(
            s_merged_q <= bound + 1e-9,
            "preserve_norm violated: {} > {}",
            s_merged_q,
            bound
        );

        let bytes_f32: usize = merged.iter().map(|l| l.raw_byte_size()).sum();
        cast_layers_dtype(&mut merged, DType::from_name("bf16").unwrap());
        let bytes_bf16: usize = merged.iter().map(|l| l.raw_byte_size()).sum();
        assert_eq!(bytes_bf16 * 2, bytes_f32);
    }

    #[test]
    fn test_alpha_scalar_safetensors_and_rslora_scale_baking() {
        let mut rng = Rng64::new(6060);
        let d = 32;
        let r = 16;
        let layer = LoraLayer {
            base_name: "lora_unet_down_blocks_0_attentions_0_proj_in".to_string(),
            a_key: "lora_unet_down_blocks_0_attentions_0_proj_in.lora_down.weight".to_string(),
            b_key: "lora_unet_down_blocks_0_attentions_0_proj_in.lora_up.weight".to_string(),
            a: random_matrix(r, d, &mut rng, DType::F32),
            b: random_matrix(d, r, &mut rng, DType::F32),
        };

        let orig_dw = matmul(&layer.b, &layer.a);
        let mut map = layers_to_tensor_map(&[layer]);
        map.insert(
            "lora_unet_down_blocks_0_attentions_0_proj_in.alpha".to_string(),
            Tensor2D {
                rows: 1,
                cols: 1,
                dtype: DType::F32,
                data: vec![8.0],
            },
        );

        let tmp_path = std::env::temp_dir().join("test_kohya_alpha.safetensors");
        write_safetensors(&tmp_path, &map).expect("write alpha safetensors");
        let loaded_map = read_safetensors(&tmp_path).expect("read alpha safetensors");
        let alphas = extract_lora_alphas(&loaded_map);
        assert_eq!(
            alphas.get("lora_unet_down_blocks_0_attentions_0_proj_in"),
            Some(&8.0)
        );

        // 1. Standard LoRA scale baking: scale = alpha / r = 8.0 / 16.0 = 0.5
        let mut layers_std = extract_lora_layers(&loaded_map).expect("extract layers");
        bake_adapter_scaling(&mut layers_std, &alphas, None, false);
        let std_dw = matmul(&layers_std[0].b, &layers_std[0].a);
        for (&orig, &baked) in orig_dw.data.iter().zip(std_dw.data.iter()) {
            assert!((baked - orig * 0.5).abs() < 1e-6);
        }

        // 2. rsLoRA scale baking: scale = alpha / sqrt(r) = 8.0 / 4.0 = 2.0
        let mut layers_rs = extract_lora_layers(&loaded_map).expect("extract layers");
        bake_adapter_scaling(&mut layers_rs, &alphas, None, true);
        let rs_dw = matmul(&layers_rs[0].b, &layers_rs[0].a);
        for (&orig, &baked) in orig_dw.data.iter().zip(rs_dw.data.iter()) {
            assert!((baked - orig * 2.0).abs() < 1e-6);
        }

        let mut out_map = layers_to_tensor_map(&layers_rs);
        insert_unit_alpha_tensors(&mut out_map, &layers_rs);
        assert_eq!(
            out_map["lora_unet_down_blocks_0_attentions_0_proj_in.alpha"].data[0],
            16.0
        );
        let _ = std::fs::remove_file(&tmp_path);
    }

    #[test]
    fn test_multi_adapter_3way_ortho_svd_merge() {
        let mut rng = Rng64::new(7070);
        let d = 48;
        let r = 4;

        let make_adapter = |rng: &mut Rng64| -> Vec<LoraLayer> {
            vec![LoraLayer {
                base_name: "model.layers.0.self_attn.q_proj".to_string(),
                a_key: "model.layers.0.self_attn.q_proj.lora_A.weight".to_string(),
                b_key: "model.layers.0.self_attn.q_proj.lora_B.weight".to_string(),
                a: random_matrix(r, d, rng, DType::F32),
                b: random_matrix(d, r, rng, DType::F32),
            }]
        };

        let a1 = make_adapter(&mut rng);
        let a2 = make_adapter(&mut rng);
        let a3 = make_adapter(&mut rng);

        let max_s1 = [
            qr_core_singular_values(&a1[0].b, &a1[0].a)[0],
            qr_core_singular_values(&a2[0].b, &a2[0].a)[0],
            qr_core_singular_values(&a3[0].b, &a3[0].a)[0],
        ]
        .into_iter()
        .fold(0.0_f64, f64::max);

        let merged = merge_multiple_adapters(
            &[a1, a2, a3],
            &[1.0, 1.0, 1.0],
            "ortho-svd",
            8,
            &[],
            true,
        )
        .expect("3-way ortho-svd merge");

        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0].rank(), 8);
        let merged_s1 = qr_core_singular_values(&merged[0].b, &merged[0].a)[0];
        assert!(merged_s1 <= max_s1 + 1e-9);
    }

    #[test]
    fn test_inspect_adapter_architecture_reverse_engineering() {
        let mut rng = Rng64::new(8181);
        let flux_layer = LoraLayer {
            base_name: "transformer.double_blocks.0.img_attn.proj".to_string(),
            a_key: "transformer.double_blocks.0.img_attn.proj.lora_down.weight".to_string(),
            b_key: "transformer.double_blocks.0.img_attn.proj.lora_up.weight".to_string(),
            a: random_matrix(4, 128, &mut rng, DType::BF16),
            b: random_matrix(128, 4, &mut rng, DType::BF16),
        };
        let flux_layer2 = LoraLayer {
            base_name: "transformer.single_blocks.0.linear1".to_string(),
            a_key: "transformer.single_blocks.0.linear1.lora_down.weight".to_string(),
            b_key: "transformer.single_blocks.0.linear1.lora_up.weight".to_string(),
            a: random_matrix(8, 128, &mut rng, DType::BF16),
            b: random_matrix(128, 8, &mut rng, DType::BF16),
        };
        let mut alphas = std::collections::BTreeMap::new();
        alphas.insert("transformer.double_blocks.0.img_attn.proj".to_string(), 4.0);

        let info = inspect_adapter_architecture(&[flux_layer, flux_layer2], &alphas);
        assert_eq!(info.naming_format, "Kohya-ss / ComfyUI");
        assert_eq!(info.inferred_architecture, "Flux.1 DiT (12B Flow Transformer)");
        assert_eq!(info.min_rank, 4);
        assert_eq!(info.max_rank, 8);
        assert!(info.is_heterogeneous_rank);
        assert_eq!(info.alpha_tensor_count, 1);
    }

    #[test]
    fn test_dora_magnitude_vector_roundtrip_and_key_format_conversion() {
        let mut rng = Rng64::new(9191);
        let d_out = 32;
        let d_in = 32;
        let r = 4;

        let mut layers = vec![LoraLayer {
            base_name: "base_model.model.layers.0.self_attn.q_proj".to_string(),
            a_key: "base_model.model.layers.0.self_attn.q_proj.lora_A.weight".to_string(),
            b_key: "base_model.model.layers.0.self_attn.q_proj.lora_B.weight".to_string(),
            a: random_matrix(r, d_in, &mut rng, DType::F32),
            b: random_matrix(d_out, r, &mut rng, DType::F32),
        }];

        let mut map = layers_to_tensor_map(&layers);
        let dora_vec = Tensor2D {
            rows: 1,
            cols: d_out,
            dtype: DType::F32,
            data: (0..d_out).map(|i| 1.0 + (i as f64) * 0.01).collect(),
        };
        map.insert(
            "base_model.model.layers.0.self_attn.q_proj.lora_magnitude_vector.weight".to_string(),
            dora_vec.clone(),
        );
        map.insert(
            "base_model.model.layers.0.self_attn.k_proj.magnitude".to_string(),
            dora_vec,
        );

        let tmp_path = std::env::temp_dir().join("test_dora_format.safetensors");
        write_safetensors(&tmp_path, &map).expect("write dora safetensors");
        let loaded = read_safetensors(&tmp_path).expect("read dora safetensors");
        let dora_scales = extract_dora_scales(&loaded);
        assert_eq!(dora_scales.len(), 2);
        assert!(dora_scales.contains_key("base_model.model.layers.0.self_attn.q_proj"));
        assert!(dora_scales.contains_key("base_model.model.layers.0.self_attn.k_proj"));

        // Convert key format from PEFT -> Kohya -> Diffusers
        convert_layers_key_format(&mut layers, "kohya").expect("convert to kohya");
        assert!(layers[0].a_key.ends_with(".lora_down.weight"));
        assert!(layers[0].b_key.ends_with(".lora_up.weight"));

        let mut out_map = layers_to_tensor_map(&layers);
        insert_dora_scales(&mut out_map, &dora_scales, Some("kohya"), None);
        assert!(out_map.contains_key("base_model.model.layers.0.self_attn.q_proj.dora_scale"));

        convert_layers_key_format(&mut layers, "diffusers").expect("convert to diffusers");
        assert!(layers[0].a_key.ends_with(".lora.down.weight"));
        assert!(layers[0].b_key.ends_with(".lora.up.weight"));

        let _ = std::fs::remove_file(&tmp_path);
    }

    #[test]
    fn test_extract_lora_from_dense_models() {
        let mut rng = Rng64::new(9292);
        let d_out = 48;
        let d_in = 48;
        let true_r = 4;

        let base_w = random_matrix(d_out, d_in, &mut rng, DType::F32);
        let true_b = random_matrix(d_out, true_r, &mut rng, DType::F32);
        let true_a = random_matrix(true_r, d_in, &mut rng, DType::F32);
        let true_dw = matmul(&true_b, &true_a);

        let mut tuned_w = base_w.clone();
        for i in 0..tuned_w.data.len() {
            tuned_w.data[i] += true_dw.data[i];
        }

        let mut base_map = std::collections::BTreeMap::new();
        let mut tuned_map = std::collections::BTreeMap::new();
        base_map.insert("model.layers.0.self_attn.q_proj.weight".to_string(), base_w);
        tuned_map.insert("model.layers.0.self_attn.q_proj.weight".to_string(), tuned_w);

        let res = extract_lora_from_dense_models(
            &base_map,
            &tuned_map,
            CompressionPolicy::FixedRank(true_r),
            "peft",
        )
        .expect("extract lora from dense models");

        assert_eq!(res.layers.len(), 1);
        assert_eq!(res.layers[0].rank(), true_r);
        assert!(res.metrics[0].relative_frobenius_error < 1e-10);

        let rec_dw = matmul(&res.layers[0].b, &res.layers[0].a);
        for (&orig, &rec) in true_dw.data.iter().zip(rec_dw.data.iter()) {
            assert!((orig - rec).abs() < 1e-9);
        }

        // Verify cross-prefix matching (e.g. ComfyUI/SRPO "model.diffusion_model.double_blocks..." vs Kohya "double_blocks...")
        let mut srpo_base_map = std::collections::BTreeMap::new();
        let mut kohya_tuned_map = std::collections::BTreeMap::new();
        srpo_base_map.insert(
            "model.diffusion_model.double_blocks.0.img_attn.proj.weight".to_string(),
            base_map["model.layers.0.self_attn.q_proj.weight"].clone(),
        );
        kohya_tuned_map.insert(
            "double_blocks.0.img_attn.proj.weight".to_string(),
            tuned_map["model.layers.0.self_attn.q_proj.weight"].clone(),
        );
        let res_prefix = extract_lora_from_dense_models(
            &srpo_base_map,
            &kohya_tuned_map,
            CompressionPolicy::FixedRank(true_r),
            "kohya",
        )
        .expect("extract lora across mismatched wrapper prefixes");
        assert_eq!(res_prefix.layers.len(), 1);
        assert_eq!(res_prefix.layers[0].base_name, "double_blocks.0.img_attn.proj");
        assert_eq!(res_prefix.layers[0].a_key, "double_blocks.0.img_attn.proj.lora_down.weight");
    }

    #[test]
    fn test_remote_hf_uri_parsing_siblings_resolution_and_cache() {
        use crate::remote::{
            build_curl_args, cache_path_for_hf, hf_api_model_url, hf_resolve_url,
            is_huggingface_url, parse_source_spec, resolve_cache_dir_from_tmpdir,
            resolve_source_path_with_cache, select_safetensors_from_hf_api_json, RemoteSpec,
        };

        // 1. hf://<org>/<repo> default repo spec
        let spec_repo = parse_source_spec("hf://XLabs-AI/flux-RealismLora").expect("parse hf repo");
        assert_eq!(
            spec_repo,
            RemoteSpec::HuggingFaceRepo {
                org: "XLabs-AI".to_string(),
                repo: "flux-RealismLora".to_string(),
                revision: "main".to_string(),
            }
        );
        assert_eq!(
            hf_api_model_url("XLabs-AI", "flux-RealismLora", "main"),
            "https://huggingface.co/api/models/XLabs-AI/flux-RealismLora"
        );

        // 2. hf://<org>/<repo>/<filename.safetensors> -> exact resolve/main URL
        let spec_file =
            parse_source_spec("hf://ostris/flux-lora/lora_v2.safetensors").expect("parse hf file");
        assert_eq!(
            spec_file,
            RemoteSpec::HuggingFaceFile {
                org: "ostris".to_string(),
                repo: "flux-lora".to_string(),
                revision: "main".to_string(),
                filename: "lora_v2.safetensors".to_string(),
            }
        );
        assert_eq!(
            hf_resolve_url("ostris", "flux-lora", "main", "lora_v2.safetensors"),
            "https://huggingface.co/ostris/flux-lora/resolve/main/lora_v2.safetensors"
        );

        // 3. Direct URL & HuggingFace /blob/ -> /resolve/ normalization
        let spec_blob = parse_source_spec(
            "https://huggingface.co/SalyyS1/demo-lora/blob/main/adapter_model.safetensors",
        )
        .expect("parse blob url");
        assert_eq!(
            spec_blob,
            RemoteSpec::DirectUrl {
                url: "https://huggingface.co/SalyyS1/demo-lora/resolve/main/adapter_model.safetensors"
                    .to_string()
            }
        );

        // 4. Reject path traversal in hf:// URIs
        assert!(parse_source_spec("hf://org/../etc/passwd").is_err());
        assert!(parse_source_spec("hf://only_org").is_err());

        // 5. Sibling discovery from HuggingFace /api/models/<org>/<repo> JSON
        let peft_api_json = r#"{
            "id": "org/peft-lora",
            "siblings": [
                {"rfilename": "README.md"},
                {"rfilename": "adapter_config.json"},
                {"rfilename": "adapter_model.safetensors"}
            ]
        }"#;
        assert_eq!(
            select_safetensors_from_hf_api_json(peft_api_json).as_deref(),
            Some("adapter_model.safetensors")
        );

        let kohya_api_json = r#"{
            "id": "XLabs-AI/flux-RealismLora",
            "siblings": [
                {"rfilename": ".gitattributes"},
                {"rfilename": "flux_realism_lora.safetensors"}
            ]
        }"#;
        assert_eq!(
            select_safetensors_from_hf_api_json(kohya_api_json).as_deref(),
            Some("flux_realism_lora.safetensors")
        );

        // 6. HF_TOKEN header scoping: attached to huggingface.co, never leaked to third-party hosts
        assert!(is_huggingface_url(
            "https://huggingface.co/org/repo/resolve/main/lora.safetensors"
        ));
        assert!(!is_huggingface_url("https://example.com/lora.safetensors"));

        let hf_args = build_curl_args(
            "https://huggingface.co/org/repo/resolve/main/lora.safetensors",
            None,
            Some("hf_secret_test_123"),
        );
        assert!(hf_args
            .iter()
            .any(|a| a == "Authorization: Bearer hf_secret_test_123"));

        let ext_args = build_curl_args(
            "https://cdn.example.com/lora.safetensors",
            None,
            Some("hf_secret_test_123"),
        );
        assert!(!ext_args.iter().any(|a| a.contains("Authorization")));

        // 7. Warm-cache zero-network resolution under ${TMPDIR:-/tmp}/loraspectra-cache/
        assert_eq!(
            resolve_cache_dir_from_tmpdir(None),
            std::path::PathBuf::from("/tmp/loraspectra-cache")
        );
        let test_cache = std::env::temp_dir().join("loraspectra_unit_cache_v060");
        let cached_file = cache_path_for_hf(
            &test_cache,
            "SalyyS1",
            "cached-lora",
            "main",
            "custom_lora.safetensors",
        );
        std::fs::create_dir_all(cached_file.parent().unwrap()).expect("create cache dir");
        std::fs::write(&cached_file, &[0u8; 32]).expect("write dummy cached safetensors");
        std::fs::write(
            cached_file.parent().unwrap().join(".resolved_target"),
            "custom_lora.safetensors",
        )
        .expect("write pointer");

        let resolved =
            resolve_source_path_with_cache("hf://SalyyS1/cached-lora", &test_cache, None)
                .expect("warm cache repo resolve without network");
        assert_eq!(resolved, cached_file);

        let resolved_explicit = resolve_source_path_with_cache(
            "hf://SalyyS1/cached-lora/custom_lora.safetensors",
            &test_cache,
            None,
        )
        .expect("warm cache explicit file resolve without network");
        assert_eq!(resolved_explicit, cached_file);

        let _ = std::fs::remove_dir_all(&test_cache);
    }

    #[test]
    fn test_interference_automated_merge_recommendation_modes_and_localization() {
        use crate::merge::analyze_adapter_interference;

        let mut rng = Rng64::new(6006);
        let d = 96;
        let r = 6;

        // Orthonormal basis pools for deterministic orthogonal vs colliding subspaces
        let (u_pool, _) = thin_qr_columns(&random_matrix(d, 3 * r, &mut rng, DType::F32));
        let (v_pool, _) = thin_qr_columns(&random_matrix(d, 3 * r, &mut rng, DType::F32));

        let build_from_modes = |name: &str, u_off: usize, v_off: usize, sign: f64| -> LoraLayer {
            let mut b = Tensor2D::zeros(d, r, DType::F32);
            let mut a = Tensor2D::zeros(r, d, DType::F32);
            for j in 0..r {
                for row in 0..d {
                    b.set(row, j, sign * u_pool[u_off + j][row] * 2.0);
                    a.set(j, row, v_pool[v_off + j][row] * 2.0);
                }
            }
            LoraLayer {
                base_name: name.to_string(),
                a_key: format!("{}.lora_A.weight", name),
                b_key: format!("{}.lora_B.weight", name),
                a,
                b,
            }
        };

        // Case 1: Orthogonal subspaces across both attn and mlp -> recommends --mode linear
        let ortho_a = vec![
            build_from_modes("model.layers.0.self_attn.q_proj", 0, 0, 1.0),
            build_from_modes("model.layers.0.mlp.down_proj", 0, 0, 1.0),
        ];
        let ortho_b = vec![
            build_from_modes("model.layers.0.self_attn.q_proj", r, r, 1.0),
            build_from_modes("model.layers.0.mlp.down_proj", r, r, 1.0),
        ];
        let sum_ortho = analyze_adapter_interference(&ortho_a, &ortho_b, "a.safetensors", "b.safetensors")
            .expect("orthogonal interference summary");
        assert_eq!(sum_ortho.recommendation.recommended_mode, "linear");
        assert_eq!(sum_ortho.recommendation.conflict_localization, "NONE");
        assert!(sum_ortho.recommendation.suggested_layer_weights.is_none());
        assert!(sum_ortho.recommendation.cli_command.contains("--mode linear"));

        // Case 2: High sign-opposing collision localized ONLY in attention, orthogonal in MLP
        // -> recommends --mode ortho-svd + --layer-weights attenuating attn!
        let loc_b = vec![
            build_from_modes("model.layers.0.self_attn.q_proj", 0, 0, -1.0),
            build_from_modes("model.layers.0.mlp.down_proj", r, r, 1.0),
        ];
        let sum_loc = analyze_adapter_interference(&ortho_a, &loc_b, "hf://org/code", "hf://org/math")
            .expect("localized conflict summary");
        assert_eq!(sum_loc.recommendation.recommended_mode, "ortho-svd");
        assert_eq!(sum_loc.recommendation.conflict_localization, "LOCALIZED_ATTN");
        let lw = sum_loc
            .recommendation
            .suggested_layer_weights
            .as_deref()
            .expect("expected suggested layer weights");
        assert!(lw.contains("attn=1.0:0.35"));
        assert!(lw.contains("mlp=1.0:1.0"));
        assert!(sum_loc.recommendation.cli_command.contains("--mode ortho-svd"));
        assert!(sum_loc.recommendation.cli_command.contains("--layer-weights"));

        // Case 3: Moderate overlap (shared U subspace, orthogonal V subspace -> cos_u = 1.0, cos_v = 0.0, frob_cos = 0.0)
        // Wait: max_principal_cos = 1.0 > 0.70 triggers ortho-svd unless cos_u ~ 0.55; let's construct cos_u = 0.55!
        let mut b_mod = Tensor2D::zeros(d, r, DType::F32);
        let mut a_mod = Tensor2D::zeros(r, d, DType::F32);
        let c = 0.55_f64;
        let s = (1.0 - c * c).sqrt();
        for j in 0..r {
            for row in 0..d {
                b_mod.set(
                    row,
                    j,
                    (c * u_pool[j][row] + s * u_pool[r + j][row]) * 2.0,
                );
                a_mod.set(j, row, v_pool[r + j][row] * 2.0);
            }
        }
        let mod_b = vec![
            LoraLayer {
                base_name: "model.layers.0.self_attn.q_proj".to_string(),
                a_key: "model.layers.0.self_attn.q_proj.lora_A.weight".to_string(),
                b_key: "model.layers.0.self_attn.q_proj.lora_B.weight".to_string(),
                a: a_mod.clone(),
                b: b_mod.clone(),
            },
            LoraLayer {
                base_name: "model.layers.0.mlp.down_proj".to_string(),
                a_key: "model.layers.0.mlp.down_proj.lora_A.weight".to_string(),
                b_key: "model.layers.0.mlp.down_proj.lora_B.weight".to_string(),
                a: a_mod,
                b: b_mod,
            },
        ];
        let sum_ties = analyze_adapter_interference(&ortho_a, &mod_b, "a.safetensors", "b.safetensors")
            .expect("moderate overlap summary");
        assert_eq!(sum_ties.recommendation.recommended_mode, "ties");
        assert!(sum_ties.recommendation.cli_command.contains("--mode ties"));
    }

    #[test]
    fn test_remote_url_caching_revisions_and_localized_mlp_conflict() {
        use crate::merge::analyze_adapter_interference;
        use crate::remote::{
            cache_path_for_url, hf_api_model_url, parse_source_spec,
            resolve_source_path_with_cache, select_safetensors_from_hf_api_json, RemoteSpec,
        };

        // 1. hf://<org>/<repo>@<revision>/<subpath>
        let spec_rev = parse_source_spec("hf://org/my-lora@v2.1/checkpoints/lora.safetensors")
            .expect("parse revision spec");
        assert_eq!(
            spec_rev,
            RemoteSpec::HuggingFaceFile {
                org: "org".to_string(),
                repo: "my-lora".to_string(),
                revision: "v2.1".to_string(),
                filename: "checkpoints/lora.safetensors".to_string(),
            }
        );
        assert_eq!(
            hf_api_model_url("org", "my-lora", "v2.1"),
            "https://huggingface.co/api/models/org/my-lora/revision/v2.1"
        );

        // 2. Prefer LoRA sibling over sharded base model weights
        let mixed_siblings = r#"{
            "siblings": [
                {"rfilename": "model-00001-of-00004.safetensors"},
                {"rfilename": "model-00002-of-00004.safetensors"},
                {"rfilename": "style_lora_bf16.safetensors"}
            ]
        }"#;
        assert_eq!(
            select_safetensors_from_hf_api_json(mixed_siblings).as_deref(),
            Some("style_lora_bf16.safetensors")
        );

        // 3. Direct URL cache resolution without network hit when cached
        let tmp_cache = std::env::temp_dir().join("loraspectra_url_cache_v060");
        let direct_url = "https://cdn.example.org/models/custom_adapter.safetensors?download=1";
        let expected_path = cache_path_for_url(&tmp_cache, direct_url);
        std::fs::create_dir_all(expected_path.parent().unwrap()).expect("mkdir url cache");
        std::fs::write(&expected_path, &[0u8; 32]).expect("seed url cache");

        let got = resolve_source_path_with_cache(direct_url, &tmp_cache, None)
            .expect("resolve cached direct url");
        assert_eq!(got, expected_path);
        let _ = std::fs::remove_dir_all(&tmp_cache);

        // 4. Localized MLP conflict -> recommends attenuating mlp in --layer-weights
        let mut rng = Rng64::new(7171);
        let d = 64;
        let r = 4;
        let (u_pool, _) = thin_qr_columns(&random_matrix(d, 3 * r, &mut rng, DType::F32));
        let (v_pool, _) = thin_qr_columns(&random_matrix(d, 3 * r, &mut rng, DType::F32));
        let make_l = |name: &str, off: usize, sign: f64| -> LoraLayer {
            let mut b = Tensor2D::zeros(d, r, DType::F32);
            let mut a = Tensor2D::zeros(r, d, DType::F32);
            for j in 0..r {
                for row in 0..d {
                    b.set(row, j, sign * u_pool[off + j][row] * 2.0);
                    a.set(j, row, v_pool[off + j][row] * 2.0);
                }
            }
            LoraLayer {
                base_name: name.to_string(),
                a_key: format!("{}.lora_down.weight", name),
                b_key: format!("{}.lora_up.weight", name),
                a,
                b,
            }
        };

        let a_layers = vec![
            make_l("lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q", 0, 1.0),
            make_l("lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_ff_net_0_proj", 0, 1.0),
        ];
        let b_layers = vec![
            make_l("lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q", r, 1.0),
            make_l("lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_ff_net_0_proj", 0, -1.0),
        ];
        let rep = analyze_adapter_interference(&a_layers, &b_layers, "style.safetensors", "char.safetensors")
            .expect("sdxl kohya interference");
        assert_eq!(rep.recommendation.conflict_localization, "LOCALIZED_MLP");
        let lw = rep.recommendation.suggested_layer_weights.unwrap();
        assert!(lw.contains("ff=1.0:0.35"), "unexpected lw: {}", lw);
    }

    #[test]
    fn test_integrity_detects_zero_norm_special_token_embedding_rows_and_dead_lora_columns() {
        use crate::audit::{audit_adapter, audit_adapter_integrity, render_audit_json};
        use crate::safetensors::{extract_dora_scales, extract_lora_alphas};

        let mut rng = Rng64::new(2027);
        let d_model = 32;
        let vocab = 64;
        let r = 8;

        // 1. Healthy q_proj LoRA layer
        let q_layer = LoraLayer {
            base_name: "base_model.model.model.layers.0.self_attn.q_proj".to_string(),
            a_key: "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight".to_string(),
            b_key: "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight".to_string(),
            a: random_matrix(r, d_model, &mut rng, DType::F32),
            b: random_matrix(d_model, r, &mut rng, DType::F32),
        };

        // 2. PEFT Embedding LoRA (embed_tokens.lora_embedding_A: r x vocab, lora_embedding_B: d_model x r)
        // Simulate 4 newly added <tool_call> special tokens at indices 60..64 that never received gradients,
        // plus 2 dead columns in lora_embedding_B.
        let mut emb_a = random_matrix(r, vocab, &mut rng, DType::F32);
        for tok in 60..vocab {
            for row in 0..r {
                emb_a.set(row, tok, 0.0);
            }
        }
        let mut emb_b = random_matrix(d_model, r, &mut rng, DType::F32);
        for row in 0..d_model {
            emb_b.set(row, 6, 0.0);
            emb_b.set(row, 7, 0.0);
        }

        // 3. Dense saved lm_head (modules_to_save) with 4 trailing zero-norm special-token rows
        let mut dense_lm_head = random_matrix(vocab, d_model, &mut rng, DType::F32);
        for tok in 60..vocab {
            for col in 0..d_model {
                dense_lm_head.set(tok, col, 0.0);
            }
        }

        let mut tensors = layers_to_tensor_map(&[q_layer]);
        tensors.insert(
            "base_model.model.model.embed_tokens.lora_embedding_A".to_string(),
            emb_a,
        );
        tensors.insert(
            "base_model.model.model.embed_tokens.lora_embedding_B".to_string(),
            emb_b,
        );
        tensors.insert(
            "base_model.model.lm_head.modules_to_save.default.weight".to_string(),
            dense_lm_head,
        );

        let layers = extract_lora_layers(&tensors).expect("extract layers with lora_embedding_A/B");
        assert_eq!(layers.len(), 2);
        let alphas = extract_lora_alphas(&tensors);
        let dora = extract_dora_scales(&tensors);

        let summary = audit_adapter(&layers);
        let integrity = audit_adapter_integrity(&layers, &tensors, &alphas, &dora, None);

        assert_eq!(integrity.status, "ISSUES_DETECTED");
        assert_eq!(integrity.embedding_entries.len(), 2);

        let dense_entry = integrity
            .embedding_entries
            .iter()
            .find(|e| e.kind == "dense_tensor")
            .expect("dense lm_head entry");
        assert_eq!(dense_entry.zero_norm_rows, 4);
        assert_eq!(dense_entry.trailing_zero_rows, 4);

        let lora_emb_entry = integrity
            .embedding_entries
            .iter()
            .find(|e| e.kind == "lora_embedding")
            .expect("lora_embedding entry");
        assert_eq!(lora_emb_entry.zero_norm_rows, 4);
        assert_eq!(lora_emb_entry.trailing_zero_rows, 4);
        assert_eq!(lora_emb_entry.dead_lora_b_cols, 2);

        assert!(integrity
            .issues
            .iter()
            .any(|i| i.code == "ZERO_NORM_EMBEDDING_ROWS" && i.message.contains("<tool_call>")));
        assert!(integrity
            .issues
            .iter()
            .any(|i| i.code == "DEAD_EMBEDDING_LORA_COLUMNS"));

        let json_out = render_audit_json("qwen_coder_sft.safetensors", &summary, &integrity);
        assert!(json_out.contains("\"integrity_status\": \"ISSUES_DETECTED\""));
        assert!(json_out.contains("\"ZERO_NORM_EMBEDDING_ROWS\""));
        assert!(json_out.contains("\"trailing_zero_rows\": 4"));
    }

    #[test]
    fn test_integrity_detects_wte_and_lm_head_lora_zero_rows_and_dead_layers() {
        use crate::audit::audit_adapter_integrity;

        let mut rng = Rng64::new(3031);
        let d = 24;
        let vocab = 40;
        let r = 4;

        // GPT-2 / StarCoder style wte + lm_head LoRA where lm_head has 3 trailing dead rows
        // and v_proj is completely dead (lora_B = 0).
        let mut lm_b = random_matrix(vocab, r, &mut rng, DType::F32);
        for tok in 37..vocab {
            for col in 0..r {
                lm_b.set(tok, col, 0.0);
            }
        }
        let lm_layer = LoraLayer {
            base_name: "transformer.lm_head".to_string(),
            a_key: "transformer.lm_head.lora_A.weight".to_string(),
            b_key: "transformer.lm_head.lora_B.weight".to_string(),
            a: random_matrix(r, d, &mut rng, DType::F32),
            b: lm_b,
        };
        let dead_v = LoraLayer {
            base_name: "transformer.h.0.attn.v_proj".to_string(),
            a_key: "transformer.h.0.attn.v_proj.lora_A.weight".to_string(),
            b_key: "transformer.h.0.attn.v_proj.lora_B.weight".to_string(),
            a: random_matrix(r, d, &mut rng, DType::F32),
            b: Tensor2D::zeros(d, r, DType::F32),
        };
        let mut tensors = layers_to_tensor_map(&[lm_layer.clone(), dead_v.clone()]);
        let mut wte = random_matrix(vocab, d, &mut rng, DType::F32);
        for col in 0..d {
            wte.set(39, col, 0.0);
        }
        tensors.insert("transformer.wte.weight".to_string(), wte);

        let rep = audit_adapter_integrity(
            &[lm_layer, dead_v],
            &tensors,
            &std::collections::BTreeMap::new(),
            &std::collections::BTreeMap::new(),
            None,
        );
        assert_eq!(rep.dead_lora_layers, 1);
        assert!(rep.issues.iter().any(|i| i.code == "DEAD_LORA_LAYER"));
        assert_eq!(rep.embedding_entries.len(), 2);
        assert!(rep
            .embedding_entries
            .iter()
            .any(|e| e.name == "transformer.wte.weight" && e.trailing_zero_rows == 1));
        assert!(rep
            .embedding_entries
            .iter()
            .any(|e| e.name == "transformer.lm_head" && e.trailing_zero_rows == 3));
    }

    #[test]
    fn test_integrity_detects_alpha_and_dora_scaling_anomalies() {
        use crate::audit::audit_adapter_integrity;
        use crate::safetensors::parse_peft_adapter_config;

        let mut rng = Rng64::new(4041);
        let d = 32;
        let r = 16;

        let l1 = LoraLayer {
            base_name: "base_model.model.layers.0.self_attn.q_proj".to_string(),
            a_key: "base_model.model.layers.0.self_attn.q_proj.lora_A.weight".to_string(),
            b_key: "base_model.model.layers.0.self_attn.q_proj.lora_B.weight".to_string(),
            a: random_matrix(r, d, &mut rng, DType::F32),
            b: random_matrix(d, r, &mut rng, DType::F32),
        };
        let l2 = LoraLayer {
            base_name: "base_model.model.layers.0.self_attn.k_proj".to_string(),
            a_key: "base_model.model.layers.0.self_attn.k_proj.lora_A.weight".to_string(),
            b_key: "base_model.model.layers.0.self_attn.k_proj.lora_B.weight".to_string(),
            a: random_matrix(r, d, &mut rng, DType::F32),
            b: random_matrix(d, r, &mut rng, DType::F32),
        };
        let layers = vec![l1, l2];
        let tensors = layers_to_tensor_map(&layers);
        let empty_alphas = std::collections::BTreeMap::new();
        let empty_dora = std::collections::BTreeMap::new();

        // Case 1: Missing lora_alpha in adapter_config.json when .alpha tensors are absent
        let cfg_missing =
            parse_peft_adapter_config(r#"{"peft_type": "LORA", "r": 16, "use_dora": true}"#)
                .expect("parse config missing lora_alpha");
        let rep_missing = audit_adapter_integrity(
            &layers,
            &tensors,
            &empty_alphas,
            &empty_dora,
            Some(&cfg_missing),
        );
        assert_eq!(rep_missing.alpha_summary.status, "ANOMALY_DETECTED");
        assert!(rep_missing
            .issues
            .iter()
            .any(|i| i.code == "MISSING_LORA_ALPHA"));
        assert!(rep_missing
            .issues
            .iter()
            .any(|i| i.code == "MISSING_DORA_VECTORS"));

        // Case 2: Partial .alpha scalar tensors across layers + extreme underscale (< 0.1)
        let mut partial_alphas = std::collections::BTreeMap::new();
        partial_alphas.insert(
            "base_model.model.layers.0.self_attn.q_proj".to_string(),
            0.5, // alpha/r = 0.5 / 16 = 0.03125 < 0.1
        );
        let rep_partial =
            audit_adapter_integrity(&layers, &tensors, &partial_alphas, &empty_dora, None);
        assert!(rep_partial
            .issues
            .iter()
            .any(|i| i.code == "MISMATCHED_ALPHA_TENSORS"));
        assert!(rep_partial
            .issues
            .iter()
            .any(|i| i.code == "EXTREME_ALPHA_UNDERSCALE"));

        // Case 3: Extreme overscale (alpha/r > 16.0) via adapter_config.json
        let cfg_overscale = parse_peft_adapter_config(
            r#"{"peft_type": "LORA", "r": 16, "lora_alpha": 512, "alpha_pattern": {"layers.0.self_attn.k_proj": 1024}}"#,
        )
        .expect("parse overscale config");
        let rep_over = audit_adapter_integrity(
            &layers,
            &tensors,
            &empty_alphas,
            &empty_dora,
            Some(&cfg_overscale),
        );
        assert!(rep_over
            .issues
            .iter()
            .any(|i| i.code == "EXTREME_ALPHA_OVERSCALE"));
        assert!((rep_over.alpha_summary.max_alpha_r_ratio.unwrap() - 64.0).abs() < 1e-6);

        // Case 4: Healthy scaling (alpha = 32, r = 16 -> alpha/r = 2.0)
        let cfg_healthy =
            parse_peft_adapter_config(r#"{"peft_type": "LORA", "r": 16, "lora_alpha": 32}"#)
                .expect("parse healthy config");
        let rep_ok = audit_adapter_integrity(
            &layers,
            &tensors,
            &empty_alphas,
            &empty_dora,
            Some(&cfg_healthy),
        );
        assert_eq!(rep_ok.status, "HEALTHY");
        assert_eq!(rep_ok.alpha_summary.status, "HEALTHY");
        assert!(rep_ok.issues.is_empty());
    }

    #[test]
    fn test_huggingface_model_card_markdown_generator_and_backlink_badge() {
        use crate::audit::{audit_adapter, audit_adapter_integrity};
        use crate::card::generate_model_card;
        use crate::safetensors::parse_peft_adapter_config;

        let mut rng = Rng64::new(5051);
        let d = 64;
        let r = 16;

        // High-utilization bottleneck layer + rank-collapsed over-parameterized layer
        let full_q = LoraLayer {
            base_name: "model.layers.0.self_attn.q_proj".to_string(),
            a_key: "model.layers.0.self_attn.q_proj.lora_A.weight".to_string(),
            b_key: "model.layers.0.self_attn.q_proj.lora_B.weight".to_string(),
            a: random_matrix(r, d, &mut rng, DType::BF16),
            b: random_matrix(d, r, &mut rng, DType::BF16),
        };

        let u_vec = random_matrix(d, 1, &mut rng, DType::BF16);
        let v_vec = random_matrix(1, d, &mut rng, DType::BF16);
        let mut b_col = Tensor2D::zeros(d, r, DType::BF16);
        let mut a_col = Tensor2D::zeros(r, d, DType::BF16);
        for idx in 0..d {
            b_col.set(idx, 0, u_vec.get(idx, 0) * 4.0);
            a_col.set(0, idx, v_vec.get(0, idx) * 4.0);
        }
        let collapsed_down = LoraLayer {
            base_name: "model.layers.0.mlp.down_proj".to_string(),
            a_key: "model.layers.0.mlp.down_proj.lora_A.weight".to_string(),
            b_key: "model.layers.0.mlp.down_proj.lora_B.weight".to_string(),
            a: a_col,
            b: b_col,
        };

        let layers = vec![full_q, collapsed_down];
        let tensors = layers_to_tensor_map(&layers);
        let alphas = std::collections::BTreeMap::new();
        let dora = std::collections::BTreeMap::new();
        let cfg = parse_peft_adapter_config(r#"{"peft_type": "LORA", "r": 16, "lora_alpha": 32}"#)
            .expect("parse peft config");

        let arch = inspect_adapter_architecture(&layers, &alphas);
        let summary = audit_adapter(&layers);
        let integrity = audit_adapter_integrity(&layers, &tensors, &alphas, &dora, Some(&cfg));

        let md = generate_model_card("hf://SalyyS1/demo-qwen-lora", &arch, &summary, &integrity);
        assert!(md.contains("https://img.shields.io/badge/LoRASpectra-99%25_Rank_r%3D"));
        assert!(md.contains("](https://github.com/SalyyS1/loraspectra)"));
        assert!(md.contains("| **Total LoRA Parameters** |"));
        assert!(md.contains("| **Mean Energy Utilization (99% Frobenius)** |"));
        assert!(md.contains("| **Optimal Eckart-Young Rank (99% Energy)** |"));
        assert!(md.contains("| **DoRA / Alpha & SFT Health** | HEALTHY (alpha/r=2.00; Standard LoRA (no DoRA vectors)) |"));
        assert!(md.contains("### Top Bottleneck & Over-Parameterized Target Modules"));
        assert!(md.contains("`model.layers.0.self_attn.q_proj`"));
        assert!(md.contains("`model.layers.0.mlp.down_proj`"));
        assert!(md.contains("`RANK_COLLAPSED`"));
        assert!(!md.contains('\u{2014}'), "model card must not contain em-dashes");
    }

    #[test]
    fn test_diff_adapters_exact_frobenius_and_subspace_overlap() {
        let mut rng = Rng64::new(42);
        let d_out = 64;
        let d_in = 64;
        let rank = 8;

        // Base layer A
        let mut a = Tensor2D::zeros(rank, d_in, DType::F32);
        let mut b = Tensor2D::zeros(d_out, rank, DType::F32);
        for x in a.data.iter_mut() {
            *x = rng.next_gaussian();
        }
        for x in b.data.iter_mut() {
            *x = rng.next_gaussian();
        }

        let layer_a = LoraLayer {
            base_name: "model.layers.0.self_attn.q_proj".to_string(),
            a_key: "model.layers.0.self_attn.q_proj.lora_A.weight".to_string(),
            b_key: "model.layers.0.self_attn.q_proj.lora_B.weight".to_string(),
            a: a.clone(),
            b: b.clone(),
        };

        // Case 1: Identical adapter -> relative drift = 0, subspace similarity = 1.0
        let mut alphas = BTreeMap::new();
        alphas.insert("model.layers.0.self_attn.q_proj".to_string(), 16.0);

        let layers_a = vec![layer_a.clone()];
        let layers_b = vec![layer_a.clone()];

        let diff = crate::diff::diff_adapters(&layers_a, &alphas, &layers_b, &alphas);
        assert_eq!(diff.shared_layers.len(), 1);
        let rep = &diff.shared_layers[0];
        assert!(rep.relative_drift < 1e-10, "identical drift should be 0, got {}", rep.relative_drift);
        assert!((rep.subspace_similarity - 1.0).abs() < 1e-5, "identical subspace similarity should be 1.0, got {}", rep.subspace_similarity);
        assert_eq!(rep.rank_delta, 0);

        // Case 2: Inverted adapter -> relative drift = 2.0, subspace similarity = 1.0
        let mut b_inv = b.clone();
        for x in b_inv.data.iter_mut() {
            *x = -*x;
        }
        let layer_inv = LoraLayer {
            b: b_inv,
            ..layer_a.clone()
        };
        let layers_inv = vec![layer_inv];
        let diff_inv = crate::diff::diff_adapters(&layers_a, &alphas, &layers_inv, &alphas);
        let rep_inv = &diff_inv.shared_layers[0];
        assert!((rep_inv.relative_drift - 2.0).abs() < 1e-5, "inverted relative drift should be 2.0, got {}", rep_inv.relative_drift);
        assert!((rep_inv.subspace_similarity - 1.0).abs() < 1e-5);

        // Case 3: Added and removed layers
        let layer_c = LoraLayer {
            base_name: "model.layers.0.mlp.down_proj".to_string(),
            a_key: "model.layers.0.mlp.down_proj.lora_A.weight".to_string(),
            b_key: "model.layers.0.mlp.down_proj.lora_B.weight".to_string(),
            a: a.clone(),
            b: b.clone(),
        };
        let layers_mixed_a = vec![layer_a.clone()];
        let layers_mixed_b = vec![layer_c.clone()];
        let diff_mixed = crate::diff::diff_adapters(&layers_mixed_a, &alphas, &layers_mixed_b, &alphas);
        assert_eq!(diff_mixed.shared_layers.len(), 0);
        assert_eq!(diff_mixed.removed_layers, vec!["model.layers.0.self_attn.q_proj"]);
        assert_eq!(diff_mixed.added_layers, vec!["model.layers.0.mlp.down_proj"]);

        // Render json and table
        let json = crate::diff::render_diff_json(&diff);
        assert!(json.contains("\"total_shared_layers\": 1"));
        assert!(json.contains("\"subspace_similarity\": 1.0"));
        assert!(!json.contains('\u{2014}'));

        let table = crate::diff::render_diff_table(&diff);
        assert!(table.contains("model.layers.0.self_attn.q_proj"));
        assert!(table.contains("RelDrift"));
        assert!(!table.contains('\u{2014}'));
    }

    #[test]
    fn test_sarif_v210_schema_generation_and_rule_mapping() {
        let d = 32;
        let r = 8;
        let zero_b = Tensor2D::zeros(d, r, DType::F32);
        let rand_a = Tensor2D::zeros(r, d, DType::F32);

        let dead_layer = LoraLayer {
            base_name: "model.layers.0.self_attn.q_proj".to_string(),
            a_key: "model.layers.0.self_attn.q_proj.lora_A.weight".to_string(),
            b_key: "model.layers.0.self_attn.q_proj.lora_B.weight".to_string(),
            a: rand_a,
            b: zero_b,
        };

        let layers = vec![dead_layer];
        let summary = audit_adapter(&layers);

        let mut tensors = BTreeMap::new();
        tensors.insert(layers[0].a_key.clone(), layers[0].a.clone());
        tensors.insert(layers[0].b_key.clone(), layers[0].b.clone());

        // Add an embedding tensor with trailing zero rows
        let mut embed = Tensor2D::zeros(16, d, DType::F32);
        for row in 0..8 {
            for col in 0..d {
                embed.data[row * d + col] = 0.5;
            }
        }
        tensors.insert("model.embed_tokens.weight".to_string(), embed);

        let alphas = BTreeMap::new();
        let dora = BTreeMap::new();
        let integrity = audit_adapter_integrity(&layers, &tensors, &alphas, &dora, None);

        let sarif_str = crate::sarif::build_audit_sarif(
            Some(&summary),
            Some(&integrity),
            "adapter_model.safetensors",
            "0.8.0",
        );

        assert!(sarif_str.contains("\"$schema\": \"https://json.schemastore.org/sarif-2.1.0.json\""));
        assert!(sarif_str.contains("\"version\": \"2.1.0\""));
        assert!(sarif_str.contains("\"name\": \"loraspectra\""));
        assert!(sarif_str.contains("\"version\": \"0.8.0\""));
        assert!(sarif_str.contains("\"ruleId\": \"LS001\""));
        assert!(sarif_str.contains("\"ruleId\": \"LS002\""));
        assert!(sarif_str.contains("\"level\": \"error\""));
        assert!(sarif_str.contains("\"uri\": \"adapter_model.safetensors\""));
        assert!(!sarif_str.contains('\u{2014}'));
    }

    #[test]
    fn test_multi_adapter_merge_check_orthogonal_three_way() {
        let d_out = 24;
        let d_in = 24;
        let r = 4;

        // Construct 3 mutually orthogonal rank-4 adapters occupying disjoint coordinate slices
        let mut inputs = Vec::new();
        for idx in 0..3 {
            let mut a = Tensor2D::zeros(r, d_in, DType::F32);
            let mut b = Tensor2D::zeros(d_out, r, DType::F32);
            for k in 0..r {
                let col = idx * r + k;
                let row = idx * r + k;
                a.set(k, col, 1.0 + 0.2 * (k as f64));
                b.set(row, k, 1.5 - 0.1 * (idx as f64));
            }
            let layer = LoraLayer {
                base_name: "model.layers.0.self_attn.q_proj".to_string(),
                a_key: "model.layers.0.self_attn.q_proj.lora_A.weight".to_string(),
                b_key: "model.layers.0.self_attn.q_proj.lora_B.weight".to_string(),
                a,
                b,
            };
            inputs.push(crate::merge_check::AdapterMergeInput {
                name: format!("adapter_{}.safetensors", idx),
                layers: vec![layer],
                alphas: BTreeMap::new(),
                peft_config: None,
                weight: 1.0,
            });
        }

        let summary = crate::merge_check::analyze_multi_adapter_merge(&inputs).expect("merge check");
        assert_eq!(summary.shared_layers.len(), 1);
        let rep = &summary.shared_layers[0];
        assert!(rep.subspace_collision_ratio < 1e-9);
        assert!((rep.orthogonal_inflation_ratio - 1.0).abs() < 1e-9);
        assert_eq!(rep.union_rank_99, 12);
        assert_eq!(summary.pairwise.len(), 3);
        assert_eq!(summary.advisor.risk_level, "LOW");
        assert_eq!(summary.advisor.recommended_mode, "linear");

        // Verify exact match against dense sum Frobenius & spectral norm
        let mut dense_sum = Tensor2D::zeros(d_out, d_in, DType::F32);
        for inp in &inputs {
            let prod = matmul(&inp.layers[0].b, &inp.layers[0].a);
            for i in 0..dense_sum.data.len() {
                dense_sum.data[i] += prod.data[i];
            }
        }
        let dense_frob = dense_sum.frobenius_norm_sq().sqrt();
        let dense_s1 = jacobi_svd(&dense_sum).sigma[0];
        assert!((rep.linear_merged_norm - dense_frob).abs() < 1e-9);
        assert!((rep.merged_spectral_norm - dense_s1).abs() < 1e-9);
    }

    #[test]
    fn test_multi_adapter_merge_check_collinear_inflation_and_alpha_scaling() {
        let mut rng = Rng64::new(9001);
        let d_out = 20;
        let d_in = 20;
        let r = 4;
        let base_a = random_matrix(r, d_in, &mut rng, DType::F32);
        let base_b = random_matrix(d_out, r, &mut rng, DType::F32);

        let mut cfg = crate::safetensors::PeftAdapterConfig::default();
        cfg.lora_alpha = Some(16.0); // scale = 16 / 4 = 4.0

        let mut inputs = Vec::new();
        for idx in 0..3 {
            let layer = LoraLayer {
                base_name: "model.layers.0.mlp.up_proj".to_string(),
                a_key: "model.layers.0.mlp.up_proj.lora_A.weight".to_string(),
                b_key: "model.layers.0.mlp.up_proj.lora_B.weight".to_string(),
                a: base_a.clone(),
                b: base_b.clone(),
            };
            inputs.push(crate::merge_check::AdapterMergeInput {
                name: format!("collinear_{}.safetensors", idx),
                layers: vec![layer],
                alphas: BTreeMap::new(),
                peft_config: Some(cfg.clone()),
                weight: 1.0,
            });
        }

        let summary = crate::merge_check::analyze_multi_adapter_merge(&inputs).expect("merge check");
        let rep = &summary.shared_layers[0];
        assert!((rep.subspace_collision_ratio - 1.0).abs() < 1e-6);
        assert!((rep.peak_inflation_ratio - 3.0).abs() < 1e-6);
        assert!((rep.spectral_inflation_ratio - 3.0).abs() < 1e-6);
        assert_eq!(rep.status, "NORM_INFLATION");
        assert!(summary.advisor.preserve_norm);
        assert!(summary.advisor.suggested_weights[0] < 0.40);
    }

    #[test]
    fn test_multi_adapter_merge_check_destructive_sign_cancellation() {
        let mut rng = Rng64::new(9002);
        let d_out = 18;
        let d_in = 18;
        let r = 4;
        let base_a = random_matrix(r, d_in, &mut rng, DType::F32);
        let base_b = random_matrix(d_out, r, &mut rng, DType::F32);
        let mut neg_b = base_b.clone();
        for v in &mut neg_b.data {
            *v = -*v;
        }
        let third_a = random_matrix(r, d_in, &mut rng, DType::F32);
        let third_b = random_matrix(d_out, r, &mut rng, DType::F32);

        let mk = |name: &str, a: Tensor2D, b: Tensor2D| crate::merge_check::AdapterMergeInput {
            name: name.to_string(),
            layers: vec![LoraLayer {
                base_name: "model.layers.1.self_attn.v_proj".to_string(),
                a_key: "model.layers.1.self_attn.v_proj.lora_A.weight".to_string(),
                b_key: "model.layers.1.self_attn.v_proj.lora_B.weight".to_string(),
                a,
                b,
            }],
            alphas: BTreeMap::new(),
            peft_config: None,
            weight: 1.0,
        };

        let inputs = vec![
            mk("task_pos.safetensors", base_a.clone(), base_b),
            mk("task_neg.safetensors", base_a, neg_b),
            mk("task_aux.safetensors", third_a, third_b),
        ];

        let summary = crate::merge_check::analyze_multi_adapter_merge(&inputs).expect("merge check");
        let rep = &summary.shared_layers[0];
        assert!((rep.min_pair_frobenius_cosine - (-1.0)).abs() < 1e-6);
        assert_eq!(rep.worst_pair_indices, (0, 1));
        assert_eq!(rep.status, "DESTRUCTIVE_CANCELLATION");
        assert_eq!(summary.destructive_layer_count, 1);
        assert_eq!(summary.advisor.recommended_mode, "ties");
        assert_eq!(summary.advisor.risk_level, "HIGH");
    }

    #[test]
    fn test_multi_adapter_merge_check_overcomplete_rank_and_disjoint_modules() {
        let mut rng = Rng64::new(9003);
        let d_out = 12;
        let d_in = 12;
        let r = 5; // 4 adapters * 5 = 20 > 12 (overcomplete stacked rank)

        let mut inputs = Vec::new();
        let mut dense_sum = Tensor2D::zeros(d_out, d_in, DType::F32);
        let weights = [0.7, 0.5, 0.9, 0.4];

        for (idx, &w) in weights.iter().enumerate() {
            let a = random_matrix(r, d_in, &mut rng, DType::F32);
            let b = random_matrix(d_out, r, &mut rng, DType::F32);
            let prod = matmul(&b, &a);
            for k in 0..dense_sum.data.len() {
                dense_sum.data[k] += w * prod.data[k];
            }
            let mut layers = vec![LoraLayer {
                base_name: "model.layers.0.self_attn.o_proj".to_string(),
                a_key: "model.layers.0.self_attn.o_proj.lora_A.weight".to_string(),
                b_key: "model.layers.0.self_attn.o_proj.lora_B.weight".to_string(),
                a,
                b,
            }];
            if idx == 3 {
                layers.push(LoraLayer {
                    base_name: "model.layers.0.mlp.down_proj".to_string(),
                    a_key: "model.layers.0.mlp.down_proj.lora_A.weight".to_string(),
                    b_key: "model.layers.0.mlp.down_proj.lora_B.weight".to_string(),
                    a: random_matrix(r, d_in, &mut rng, DType::F32),
                    b: random_matrix(d_out, r, &mut rng, DType::F32),
                });
            }
            inputs.push(crate::merge_check::AdapterMergeInput {
                name: format!("over_{}.safetensors", idx),
                layers,
                alphas: BTreeMap::new(),
                peft_config: None,
                weight: w,
            });
        }

        let summary = crate::merge_check::analyze_multi_adapter_merge(&inputs).expect("merge check");
        assert_eq!(summary.shared_layers.len(), 1);
        assert_eq!(summary.single_adapter_layers.len(), 1);

        let rep = &summary.shared_layers[0];
        let dense_frob = dense_sum.frobenius_norm_sq().sqrt();
        let dense_s1 = jacobi_svd(&dense_sum).sigma[0];
        assert!(
            (rep.linear_merged_norm - dense_frob).abs() < 1e-7,
            "frob mismatch: {} vs {}",
            rep.linear_merged_norm,
            dense_frob
        );
        assert!(
            (rep.merged_spectral_norm - dense_s1).abs() < 1e-7,
            "s1 mismatch: {} vs {}",
            rep.merged_spectral_norm,
            dense_s1
        );

        let json_out = crate::merge_check::render_merge_check_json(&summary);
        let table_out = crate::merge_check::render_merge_check_table(&summary);
        assert!(json_out.contains("\"single_adapter_layer_count\": 1"));
        assert!(!json_out.contains('\u{2014}'));
        assert!(!table_out.contains('\u{2014}'));
    }

    #[test]
    fn test_merge_check_incompatible_shapes_and_sarif_export() {
        let mut rng = Rng64::new(9004);
        let inp_a = crate::merge_check::AdapterMergeInput {
            name: "llama8b_lora.safetensors".to_string(),
            layers: vec![LoraLayer {
                base_name: "model.layers.0.self_attn.q_proj".to_string(),
                a_key: "model.layers.0.self_attn.q_proj.lora_A.weight".to_string(),
                b_key: "model.layers.0.self_attn.q_proj.lora_B.weight".to_string(),
                a: random_matrix(4, 16, &mut rng, DType::F32),
                b: random_matrix(16, 4, &mut rng, DType::F32),
            }],
            alphas: BTreeMap::new(),
            peft_config: None,
            weight: 1.0,
        };
        let inp_b = crate::merge_check::AdapterMergeInput {
            name: "qwen7b_lora.safetensors".to_string(),
            layers: vec![LoraLayer {
                base_name: "model.layers.0.self_attn.q_proj".to_string(),
                a_key: "model.layers.0.self_attn.q_proj.lora_A.weight".to_string(),
                b_key: "model.layers.0.self_attn.q_proj.lora_B.weight".to_string(),
                a: random_matrix(4, 24, &mut rng, DType::F32),
                b: random_matrix(24, 4, &mut rng, DType::F32),
            }],
            alphas: BTreeMap::new(),
            peft_config: None,
            weight: 1.0,
        };

        let summary =
            crate::merge_check::analyze_multi_adapter_merge(&[inp_a, inp_b]).expect("merge check");
        assert_eq!(summary.incompatible_shapes.len(), 1);
        assert_eq!(summary.advisor.risk_level, "INCOMPATIBLE");

        let sarif = crate::sarif::build_merge_check_sarif(&summary, "0.9.0");
        assert!(sarif.contains("\"ruleId\": \"LS018\""));
        assert!(sarif.contains("\"version\": \"0.9.0\""));
        assert!(!sarif.contains('\u{2014}'));
    }

    #[test]
    fn test_budget_optimal_rank_allocator_beats_uniform_fixed_rank_at_identical_bytes() {
        let mut rng = Rng64::new(10001);
        let d = 64;
        let r = 16;

        // Build 2 rank-2 collapsed attention layers + 2 rank-10 MLP layers
        let mut layers = Vec::new();
        for idx in 0..2 {
            let u2 = random_matrix(d, 2, &mut rng, DType::F32);
            let v2 = random_matrix(2, d, &mut rng, DType::F32);
            let mut b = Tensor2D::zeros(d, r, DType::F32);
            let mut a = Tensor2D::zeros(r, d, DType::F32);
            for col in 0..2 {
                for row in 0..d {
                    b.set(row, col, u2.get(row, col) * 2.0);
                    a.set(col, row, v2.get(col, row) * 2.0);
                }
            }
            let name = format!("base_model.model.model.layers.{}.self_attn.q_proj", idx);
            layers.push(LoraLayer {
                a_key: format!("{}.lora_A.weight", name),
                b_key: format!("{}.lora_B.weight", name),
                base_name: name,
                a,
                b,
            });
        }
        for idx in 0..2 {
            let u10 = random_matrix(d, 10, &mut rng, DType::F32);
            let v10 = random_matrix(10, d, &mut rng, DType::F32);
            let mut b = Tensor2D::zeros(d, r, DType::F32);
            let mut a = Tensor2D::zeros(r, d, DType::F32);
            for col in 0..10 {
                for row in 0..d {
                    b.set(row, col, u10.get(row, col) * 3.0);
                    a.set(col, row, v10.get(col, row) * 3.0);
                }
            }
            let name = format!("base_model.model.model.layers.{}.mlp.down_proj", idx);
            layers.push(LoraLayer {
                a_key: format!("{}.lora_A.weight", name),
                b_key: format!("{}.lora_B.weight", name),
                base_name: name,
                a,
                b,
            });
        }

        let res_fixed = crate::compress::compress_adapter(&layers, CompressionPolicy::FixedRank(6));
        let res_budget = crate::compress::compress_adapter(
            &layers,
            CompressionPolicy::TargetBudget {
                target: crate::compress::BudgetTarget::TargetMeanRank(6.0),
                min_rank: 1,
                max_rank: 16,
            },
        );

        assert!(
            res_budget.total_compressed_bytes <= res_fixed.total_compressed_bytes,
            "budget bytes {} exceeded fixed bytes {}",
            res_budget.total_compressed_bytes,
            res_fixed.total_compressed_bytes
        );
        assert!(
            res_budget.global_retained_energy > res_fixed.global_retained_energy + 0.15,
            "expected budget allocator to strictly beat uniform rank: budget={} vs fixed={}",
            res_budget.global_retained_energy,
            res_fixed.global_retained_energy
        );
        assert!(res_budget.global_retained_energy > 0.999999);
        assert_eq!(res_budget.metrics[0].compressed_rank, 2);
        assert_eq!(res_budget.metrics[1].compressed_rank, 2);
        assert_eq!(res_budget.metrics[2].compressed_rank, 10);
        assert_eq!(res_budget.metrics[3].compressed_rank, 10);
    }

    #[test]
    fn test_budget_optimal_allocator_respects_exact_max_bytes_and_dtype_cost() {
        assert_eq!(crate::compress::parse_byte_budget("512").unwrap(), 512);
        assert_eq!(crate::compress::parse_byte_budget("4K").unwrap(), 4096);
        assert_eq!(
            crate::compress::parse_byte_budget("1.5MiB").unwrap(),
            1_572_864
        );
        assert!(crate::compress::parse_byte_budget("0").is_err());
        assert!(crate::compress::parse_byte_budget("-8K").is_err());
        assert!(crate::compress::parse_byte_budget("12TB").is_err());

        let mut rng = Rng64::new(10002);
        let narrow_k = LoraLayer {
            base_name: "model.layers.0.self_attn.k_proj".to_string(),
            a_key: "model.layers.0.self_attn.k_proj.lora_A.weight".to_string(),
            b_key: "model.layers.0.self_attn.k_proj.lora_B.weight".to_string(),
            a: random_matrix(16, 64, &mut rng, DType::F32),
            b: random_matrix(16, 16, &mut rng, DType::F32),
        };
        let wide_up = LoraLayer {
            base_name: "model.layers.0.mlp.up_proj".to_string(),
            a_key: "model.layers.0.mlp.up_proj.lora_A.weight".to_string(),
            b_key: "model.layers.0.mlp.up_proj.lora_B.weight".to_string(),
            a: random_matrix(16, 64, &mut rng, DType::F32),
            b: random_matrix(192, 16, &mut rng, DType::F32),
        };

        // In BF16 (2 bytes/elem): narrow_k step = (64+16)*2 = 160B, wide_up step = (64+192)*2 = 512B.
        let max_bytes = 2_400usize;
        let mut res = crate::compress::compress_adapter_with_options(
            &[narrow_k, wide_up],
            None,
            CompressionPolicy::TargetBudget {
                target: crate::compress::BudgetTarget::MaxBytes(max_bytes),
                min_rank: 2,
                max_rank: 12,
            },
            Some(DType::BF16),
        );
        cast_layers_dtype(&mut res.layers, DType::BF16);
        let bf16_bytes: usize = res.layers.iter().map(|l| l.raw_byte_size()).sum();
        assert!(
            bf16_bytes <= max_bytes,
            "BF16 payload {} exceeded max_bytes {}",
            bf16_bytes,
            max_bytes
        );
        for m in &res.metrics {
            assert!(m.compressed_rank >= 2 && m.compressed_rank <= 12);
        }
    }

    #[test]
    fn test_peft_scale_preserving_alpha_pattern_standard_and_rslora_roundtrip() {
        let mut rng = Rng64::new(10003);
        let d = 32;
        let r = 8;

        // Exact rank-3 layer and exact rank-5 layer stored in rank-8 tensors
        let make_exact_rank_layer = |name: &str, true_k: usize, rng: &mut Rng64| -> LoraLayer {
            let u = random_matrix(d, true_k, rng, DType::F32);
            let v = random_matrix(true_k, d, rng, DType::F32);
            let mut b = Tensor2D::zeros(d, r, DType::F32);
            let mut a = Tensor2D::zeros(r, d, DType::F32);
            for col in 0..true_k {
                for row in 0..d {
                    b.set(row, col, u.get(row, col));
                    a.set(col, row, v.get(col, row));
                }
            }
            LoraLayer {
                base_name: name.to_string(),
                a_key: format!("{}.lora_A.weight", name),
                b_key: format!("{}.lora_B.weight", name),
                a,
                b,
            }
        };

        let l0 = make_exact_rank_layer("base_model.model.model.layers.0.self_attn.q_proj", 3, &mut rng);
        let l1 = make_exact_rank_layer("base_model.model.model.layers.0.mlp.down_proj", 5, &mut rng);
        let layers = vec![l0, l1];

        for use_rslora in [false, true] {
            let mut src_cfg = crate::safetensors::PeftAdapterConfig::default();
            src_cfg.r = Some(r);
            src_cfg.lora_alpha = Some(24.0);
            src_cfg.use_rslora = use_rslora;
            src_cfg.base_model_name_or_path = Some("Qwen/Qwen2.5-7B".to_string());
            src_cfg.task_type = Some("CAUSAL_LM".to_string());
            src_cfg.target_modules = vec!["down_proj".to_string(), "q_proj".to_string()];
            src_cfg.modules_to_save = vec!["lm_head".to_string()];

            let alphas = BTreeMap::new();
            let mut pre_scales = Vec::new();
            for l in &layers {
                let (_, s, _) = crate::safetensors::resolve_layer_scaling_params(
                    l,
                    &alphas,
                    None,
                    Some(&src_cfg),
                    false,
                );
                pre_scales.push(s);
            }

            let res = crate::compress::compress_adapter_with_options(
                &layers,
                Some(&pre_scales),
                CompressionPolicy::AdaptiveEnergy {
                    target_energy: 0.9999,
                    min_rank: 1,
                    max_rank: 8,
                },
                None,
            );
            assert_eq!(res.layers[0].rank(), 3);
            assert_eq!(res.layers[1].rank(), 5);

            let alpha_pattern = crate::safetensors::compute_peft_alpha_pattern(
                &res.layers,
                &pre_scales,
                false,
                use_rslora,
            );
            let inferred = crate::safetensors::infer_peft_target_modules(&res.layers);
            let json = crate::safetensors::build_peft_adapter_config_json(
                &res.rank_pattern,
                &alpha_pattern,
                5,
                use_rslora,
                false,
                &inferred,
                Some(&src_cfg),
            );
            let parsed = crate::safetensors::parse_peft_adapter_config(&json).expect("parse emitted config");
            assert_eq!(
                parsed.base_model_name_or_path.as_deref(),
                Some("Qwen/Qwen2.5-7B")
            );
            assert_eq!(parsed.task_type.as_deref(), Some("CAUSAL_LM"));
            assert_eq!(parsed.use_rslora, use_rslora);
            assert_eq!(parsed.modules_to_save, vec!["lm_head".to_string()]);
            assert_eq!(
                parsed.rank_pattern.get("model.layers.0.self_attn.q_proj"),
                Some(&3)
            );
            assert_eq!(
                parsed.rank_pattern.get("model.layers.0.mlp.down_proj"),
                Some(&5)
            );

            // Verify exact forward-pass equivalence: s_orig * (B @ A) == s_new * (B_k @ A_k)
            for (orig_l, comp_l) in layers.iter().zip(res.layers.iter()) {
                let (_, s_orig, _) = crate::safetensors::resolve_layer_scaling_params(
                    orig_l,
                    &alphas,
                    None,
                    Some(&src_cfg),
                    false,
                );
                let (_, s_new, _) = crate::safetensors::resolve_layer_scaling_params(
                    comp_l,
                    &alphas,
                    None,
                    Some(&parsed),
                    false,
                );
                assert!(
                    (s_orig - s_new).abs() < 1e-6,
                    "scale mismatch (rslora={}): orig={} vs new={}",
                    use_rslora,
                    s_orig,
                    s_new
                );
                let dw_orig = matmul(&orig_l.b, &orig_l.a);
                let dw_comp = matmul(&comp_l.b, &comp_l.a);
                for (&x, &y) in dw_orig.data.iter().zip(dw_comp.data.iter()) {
                    assert!((s_orig * x - s_new * y).abs() < 1e-6);
                }
            }
        }
    }

    #[test]
    fn test_peft_pattern_dot_boundary_and_string_target_modules() {
        let mut pat = BTreeMap::new();
        pat.insert("layers.1.self_attn.q_proj".to_string(), 16usize);
        pat.insert("q_proj".to_string(), 4usize);

        assert_eq!(
            crate::safetensors::match_peft_pattern_value(
                &pat,
                "base_model.model.model.layers.1.self_attn.q_proj"
            ),
            Some(16)
        );
        assert_eq!(
            crate::safetensors::match_peft_pattern_value(
                &pat,
                "base_model.model.model.layers.10.self_attn.q_proj"
            ),
            Some(4)
        );

        let mut single_pat = BTreeMap::new();
        single_pat.insert("layers.1".to_string(), 32usize);
        assert_eq!(
            crate::safetensors::match_peft_pattern_value(
                &single_pat,
                "model.layers.10.self_attn.q_proj"
            ),
            None
        );

        let raw_cfg = r#"{
          "peft_type": "LORA",
          "r": 16,
          "lora_alpha": 32,
          "target_modules": "all-linear",
          "base_model_name_or_path": null
        }"#;
        let parsed = crate::safetensors::parse_peft_adapter_config(raw_cfg).expect("parse all-linear");
        assert_eq!(parsed.target_modules_str.as_deref(), Some("all-linear"));
        assert_eq!(parsed.base_model_name_or_path, None);

        let mut rp = BTreeMap::new();
        rp.insert("model.layers.0.q_proj".to_string(), 8usize);
        let mut ap = BTreeMap::new();
        ap.insert("model.layers.0.q_proj".to_string(), 16.0);
        let out_json = crate::safetensors::build_peft_adapter_config_json(
            &rp,
            &ap,
            8,
            false,
            false,
            &[],
            Some(&parsed),
        );
        assert!(out_json.contains("\"target_modules\": \"all-linear\""));
    }

    #[test]
    fn test_dense_model_extraction_with_global_budget_allocator() {
        let mut rng = Rng64::new(10005);
        let d = 24;
        let mut base = BTreeMap::new();
        let mut tuned = BTreeMap::new();

        let w0 = random_matrix(d, d, &mut rng, DType::F32);
        let w1 = random_matrix(d, d, &mut rng, DType::F32);
        base.insert("model.layers.0.q_proj.weight".to_string(), w0.clone());
        base.insert("model.layers.0.down_proj.weight".to_string(), w1.clone());

        // Layer 0 gets rank-1 delta; Layer 1 gets rank-6 delta
        let u1 = random_matrix(d, 1, &mut rng, DType::F32);
        let v1 = random_matrix(1, d, &mut rng, DType::F32);
        let d0 = matmul(&u1, &v1);
        let mut t0 = w0;
        for (t, &dv) in t0.data.iter_mut().zip(d0.data.iter()) {
            *t += dv;
        }

        let u6 = random_matrix(d, 6, &mut rng, DType::F32);
        let v6 = random_matrix(6, d, &mut rng, DType::F32);
        let d1 = matmul(&u6, &v6);
        let mut t1 = w1;
        for (t, &dv) in t1.data.iter_mut().zip(d1.data.iter()) {
            *t += dv * 2.0;
        }

        tuned.insert("model.layers.0.q_proj.weight".to_string(), t0);
        tuned.insert("model.layers.0.down_proj.weight".to_string(), t1);

        // Each F32 rank step on 24x24 costs (24+24)*4 = 192 bytes.
        // Budget for 7 total rank steps = 7 * 192 = 1344 bytes -> exact (1, 6) allocation!
        let res = extract_lora_from_dense_models(
            &base,
            &tuned,
            CompressionPolicy::TargetBudget {
                target: crate::compress::BudgetTarget::MaxBytes(1344),
                min_rank: 1,
                max_rank: 12,
            },
            "peft",
        )
        .expect("extract budget lora");

        assert!(res.total_compressed_bytes <= 1344);
        assert_eq!(res.rank_pattern.get("model.layers.0.q_proj"), Some(&1));
        assert_eq!(res.rank_pattern.get("model.layers.0.down_proj"), Some(&6));
        assert!(res.global_retained_energy > 0.999999);
    }

    #[test]
    fn test_dora_1d_safetensors_roundtrip_filter_and_weighted_merge() {
        let mut rng = Rng64::new(11001);
        let d_out = 24;
        let d_in = 16;
        let r = 4;

        let q_layer = LoraLayer {
            base_name: "base_model.model.layers.0.self_attn.q_proj".to_string(),
            a_key: "base_model.model.layers.0.self_attn.q_proj.lora_A.weight".to_string(),
            b_key: "base_model.model.layers.0.self_attn.q_proj.lora_B.weight".to_string(),
            a: random_matrix(r, d_in, &mut rng, DType::F32),
            b: random_matrix(d_out, r, &mut rng, DType::F32),
        };
        let k_layer = LoraLayer {
            base_name: "base_model.model.layers.0.self_attn.k_proj".to_string(),
            a_key: "base_model.model.layers.0.self_attn.k_proj.lora_A.weight".to_string(),
            b_key: "base_model.model.layers.0.self_attn.k_proj.lora_B.weight".to_string(),
            a: random_matrix(r, d_in, &mut rng, DType::F32),
            b: random_matrix(d_out, r, &mut rng, DType::F32),
        };

        // True 1D [d_out] tensors represented with rows=d_out, cols=1
        let dora_q = Tensor2D {
            rows: d_out,
            cols: 1,
            dtype: DType::F32,
            data: (0..d_out).map(|i| 1.0 + 0.05 * (i as f64)).collect(),
        };
        let dora_k = Tensor2D {
            rows: d_out,
            cols: 1,
            dtype: DType::F32,
            data: (0..d_out).map(|i| 2.0 + 0.02 * (i as f64)).collect(),
        };

        let mut map = layers_to_tensor_map(&[q_layer.clone(), k_layer]);
        map.insert(
            "base_model.model.layers.0.self_attn.q_proj.lora_magnitude_vector.weight".to_string(),
            dora_q.clone(),
        );
        map.insert(
            "base_model.model.layers.0.self_attn.k_proj.magnitude".to_string(),
            dora_k,
        );

        let tmp = std::env::temp_dir().join("test_dora_1d_v110.safetensors");
        write_safetensors(&tmp, &map).expect("write 1d dora safetensors");
        let raw_bytes = std::fs::read(&tmp).expect("read raw bytes");
        let header_len = u64::from_le_bytes(raw_bytes[0..8].try_into().unwrap()) as usize;
        let header_str = std::str::from_utf8(&raw_bytes[8..8 + header_len]).unwrap();
        assert!(
            header_str.contains(
                "\"base_model.model.layers.0.self_attn.q_proj.lora_magnitude_vector.weight\":{\"dtype\":\"F32\",\"shape\":[24]"
            ),
            "expected 1D shape [24] in header, got {}",
            header_str
        );

        let loaded = read_safetensors(&tmp).expect("reload 1d dora safetensors");
        let _ = std::fs::remove_file(&tmp);
        let extracted = extract_dora_scales(&loaded);
        assert_eq!(extracted.len(), 2);

        // Filtering against retained [q_layer] drops k_proj orphan DoRA vector
        let filtered =
            crate::safetensors::filter_dora_scales_for_layers(&extracted, std::slice::from_ref(&q_layer));
        assert_eq!(filtered.len(), 1);
        assert!(filtered.contains_key("base_model.model.layers.0.self_attn.q_proj"));

        // Weighted merge preserves scale invariance on identical vectors and convex-combines distinct vectors
        let mut d2 = BTreeMap::new();
        let dora_q_shifted = Tensor2D {
            rows: d_out,
            cols: 1,
            dtype: DType::F32,
            data: dora_q.data.iter().map(|&v| v + 1.0).collect(),
        };
        d2.insert(
            "base_model.model.layers.0.self_attn.q_proj".to_string(),
            (
                "base_model.model.layers.0.self_attn.q_proj.lora_magnitude_vector".to_string(),
                dora_q_shifted,
            ),
        );
        let merged_dora = crate::safetensors::merge_dora_scales(&[filtered, d2], &[3.0, 1.0]);
        let m_vec = &merged_dora["base_model.model.layers.0.self_attn.q_proj"].1;
        for (idx, (&orig, &mv)) in dora_q.data.iter().zip(m_vec.data.iter()).enumerate() {
            assert!(
                (mv - (orig + 0.25)).abs() < 1e-6,
                "channel {} expected {}, got {}",
                idx,
                orig + 0.25,
                mv
            );
        }
    }

    #[test]
    fn test_dora_audit_row_norms_ls019_ls020_sarif_diagnostics() {
        let mut rng = Rng64::new(11002);
        let d_out = 20;
        let d_in = 16;
        let r = 4;

        let healthy_layer = LoraLayer {
            base_name: "model.layers.0.self_attn.q_proj".to_string(),
            a_key: "model.layers.0.self_attn.q_proj.lora_A.weight".to_string(),
            b_key: "model.layers.0.self_attn.q_proj.lora_B.weight".to_string(),
            a: random_matrix(r, d_in, &mut rng, DType::F32),
            b: random_matrix(d_out, r, &mut rng, DType::F32),
        };

        // Verify O(r^2 d) Gram-matrix row norms match exact dense B*A row Euclidean norms
        let fast_norms = crate::audit::compute_layer_row_directional_norms(&healthy_layer);
        let dense = matmul(&healthy_layer.b, &healthy_layer.a);
        for row in 0..d_out {
            let slice = &dense.data[row * d_in..(row + 1) * d_in];
            let dense_norm = slice.iter().map(|&x| x * x).sum::<f64>().sqrt();
            assert!(
                (fast_norms[row] - dense_norm).abs() < 1e-11,
                "row {} norm mismatch: fast={} vs dense={}",
                row,
                fast_norms[row],
                dense_norm
            );
        }

        // Layer 2: dead directional B*A (75% zero rows) with diverging DoRA magnitude -> LS020
        let mut stag_b = random_matrix(d_out, r, &mut rng, DType::F32);
        for row in 0..15 {
            for col in 0..r {
                stag_b.set(row, col, 0.0);
            }
        }
        let stagnant_layer = LoraLayer {
            base_name: "model.layers.0.self_attn.k_proj".to_string(),
            a_key: "model.layers.0.self_attn.k_proj.lora_A.weight".to_string(),
            b_key: "model.layers.0.self_attn.k_proj.lora_B.weight".to_string(),
            a: random_matrix(r, d_in, &mut rng, DType::F32),
            b: stag_b,
        };

        // Layer 3: invalid collapsed zero channel in DoRA magnitude -> LS019
        let collapsed_mag_layer = LoraLayer {
            base_name: "model.layers.0.mlp.up_proj".to_string(),
            a_key: "model.layers.0.mlp.up_proj.lora_A.weight".to_string(),
            b_key: "model.layers.0.mlp.up_proj.lora_B.weight".to_string(),
            a: random_matrix(r, d_in, &mut rng, DType::F32),
            b: random_matrix(d_out, r, &mut rng, DType::F32),
        };

        let layers = vec![healthy_layer, stagnant_layer, collapsed_mag_layer];
        let all_tensors = layers_to_tensor_map(&layers);
        let mut dora_scales = BTreeMap::new();
        // Extreme 100x channel dispersion on q_proj -> LS020 (DORA_EXTREME_DISPERSION)
        let mut disp_data = vec![1.0; d_out];
        disp_data[0] = 100.0;
        dora_scales.insert(
            "model.layers.0.self_attn.q_proj".to_string(),
            (
                "model.layers.0.self_attn.q_proj.lora_magnitude_vector".to_string(),
                Tensor2D {
                    rows: d_out,
                    cols: 1,
                    dtype: DType::F32,
                    data: disp_data,
                },
            ),
        );
        // Stagnant directional layer k_proj -> LS020 (DORA_DIRECTIONAL_STAGNATION)
        dora_scales.insert(
            "model.layers.0.self_attn.k_proj".to_string(),
            (
                "model.layers.0.self_attn.k_proj.lora_magnitude_vector".to_string(),
                Tensor2D {
                    rows: d_out,
                    cols: 1,
                    dtype: DType::F32,
                    data: (0..d_out).map(|i| 0.8 + 0.05 * (i as f64)).collect(),
                },
            ),
        );
        // Zero channel on up_proj -> LS019 (INVALID_DORA_MAGNITUDE)
        let mut zero_ch = vec![1.0; d_out];
        zero_ch[3] = 0.0;
        dora_scales.insert(
            "model.layers.0.mlp.up_proj".to_string(),
            (
                "model.layers.0.mlp.up_proj.lora_magnitude_vector".to_string(),
                Tensor2D {
                    rows: d_out,
                    cols: 1,
                    dtype: DType::F32,
                    data: zero_ch,
                },
            ),
        );

        let summary = audit_adapter(&layers);
        let integrity =
            audit_adapter_integrity(&layers, &all_tensors, &BTreeMap::new(), &dora_scales, None);
        assert_eq!(integrity.dora_diagnostics.len(), 3);
        assert!(integrity.issues.iter().any(|i| i.code == "DORA_EXTREME_DISPERSION"));
        assert!(integrity
            .issues
            .iter()
            .any(|i| i.code == "DORA_DIRECTIONAL_STAGNATION"));
        assert!(integrity
            .issues
            .iter()
            .any(|i| i.code == "INVALID_DORA_MAGNITUDE"));

        let audit_json = crate::audit::render_audit_json("dora.safetensors", &summary, &integrity);
        assert!(audit_json.contains("\"dora_diagnostics\": ["));
        assert!(!audit_json.contains('\u{2014}'));

        let sarif =
            crate::sarif::build_audit_sarif(Some(&summary), Some(&integrity), "dora.safetensors", "1.1.0");
        assert!(sarif.contains("\"ruleId\": \"LS019\""));
        assert!(sarif.contains("\"ruleId\": \"LS020\""));
        assert!(sarif.contains("\"version\": \"1.1.0\""));
        assert!(!sarif.contains('\u{2014}'));
    }

    #[test]
    fn test_diff_adapters_with_dora_directional_vs_magnitude_coupling() {
        let mut rng = Rng64::new(11003);
        let d_out = 18;
        let d_in = 16;
        let r = 4;

        let a1 = random_matrix(r, d_in, &mut rng, DType::F32);
        let b1 = random_matrix(d_out, r, &mut rng, DType::F32);
        // Adapter B scales each row k of B by (1 + 0.1 * k) so directional row delta grows monotonically with k
        let mut b2 = b1.clone();
        for k in 0..d_out {
            let mult = 1.0 + 0.1 * (k as f64);
            for col in 0..r {
                b2.set(k, col, b1.get(k, col) * mult);
            }
        }
        let l1 = LoraLayer {
            base_name: "model.layers.0.self_attn.v_proj".to_string(),
            a_key: "model.layers.0.self_attn.v_proj.lora_A.weight".to_string(),
            b_key: "model.layers.0.self_attn.v_proj.lora_B.weight".to_string(),
            a: a1.clone(),
            b: b1,
        };
        let l2 = LoraLayer {
            base_name: "model.layers.0.self_attn.v_proj".to_string(),
            a_key: "model.layers.0.self_attn.v_proj.lora_A.weight".to_string(),
            b_key: "model.layers.0.self_attn.v_proj.lora_B.weight".to_string(),
            a: a1,
            b: b2,
        };

        // Construct DoRA magnitude vectors where |m_B[k] - m_A[k]| decreases with k (negative coupling, as in DoRA ICML 2024)
        let mag_a = vec![2.0; d_out];
        let mut mag_b = Vec::with_capacity(d_out);
        for k in 0..d_out {
            let dense_row_delta = {
                let dw1 = matmul(&l1.b, &l1.a);
                let dw2 = matmul(&l2.b, &l2.a);
                let s1 = &dw1.data[k * d_in..(k + 1) * d_in];
                let s2 = &dw2.data[k * d_in..(k + 1) * d_in];
                s1.iter()
                    .zip(s2.iter())
                    .map(|(x, y)| (y - x) * (y - x))
                    .sum::<f64>()
                    .sqrt()
            };
            mag_b.push(2.0 + (2.0 - dense_row_delta).max(0.05));
        }

        let mut dora_a = BTreeMap::new();
        dora_a.insert(
            l1.base_name.clone(),
            (
                format!("{}.lora_magnitude_vector", l1.base_name),
                Tensor2D {
                    rows: d_out,
                    cols: 1,
                    dtype: DType::F32,
                    data: mag_a.clone(),
                },
            ),
        );
        let mut dora_b = BTreeMap::new();
        dora_b.insert(
            l2.base_name.clone(),
            (
                format!("{}.lora_magnitude_vector", l2.base_name),
                Tensor2D {
                    rows: d_out,
                    cols: 1,
                    dtype: DType::F32,
                    data: mag_b.clone(),
                },
            ),
        );

        let summary = crate::diff::diff_adapters_with_dora(
            &[l1],
            &BTreeMap::new(),
            &dora_a,
            &[l2],
            &BTreeMap::new(),
            &dora_b,
        );
        assert_eq!(summary.dora_shared_layers, 1);
        let rep = &summary.shared_layers[0];
        let expected_l2 = mag_a
            .iter()
            .zip(mag_b.iter())
            .map(|(a, b)| (b - a) * (b - a))
            .sum::<f64>()
            .sqrt();
        assert!((rep.dora_l2_drift.unwrap() - expected_l2).abs() < 1e-11);
        assert!(
            rep.dora_mag_dir_coupling.unwrap() < -0.95,
            "expected strong negative coupling, got {}",
            rep.dora_mag_dir_coupling.unwrap()
        );

        let diff_json = crate::diff::render_diff_json(&summary);
        let diff_table = crate::diff::render_diff_table(&summary);
        assert!(diff_json.contains("\"dora_shared_layers\": 1"));
        assert!(diff_table.contains("DoRA Magnitude Drift:"));
        assert!(!diff_json.contains('\u{2014}'));
        assert!(!diff_table.contains('\u{2014}'));
    }

    #[test]
    fn test_rslora_rank_stabilized_scaling_audit_diff_and_sarif_ls021() {
        let mut rng = Rng64::new(12001);
        let d_out = 24;
        let d_in = 24;

        // 1. High-rank rsLoRA (r=256, lora_alpha=16 -> alpha/r=0.0625, but alpha/sqrt(r)=1.0)
        let hi_layer = LoraLayer {
            base_name: "model.layers.0.self_attn.q_proj".to_string(),
            a_key: "model.layers.0.self_attn.q_proj.lora_A.weight".to_string(),
            b_key: "model.layers.0.self_attn.q_proj.lora_B.weight".to_string(),
            a: random_matrix(256, d_in, &mut rng, DType::F32),
            b: random_matrix(d_out, 256, &mut rng, DType::F32),
        };
        let hi_tensors = layers_to_tensor_map(std::slice::from_ref(&hi_layer));
        let hi_cfg = crate::safetensors::parse_peft_adapter_config(
            r#"{"peft_type":"LORA","r":256,"lora_alpha":16,"use_rslora":true}"#,
        )
        .expect("parse rslora config");
        let hi_int = audit_adapter_integrity(
            std::slice::from_ref(&hi_layer),
            &hi_tensors,
            &BTreeMap::new(),
            &BTreeMap::new(),
            Some(&hi_cfg),
        );
        assert_eq!(hi_int.status, "HEALTHY", "issues: {:?}", hi_int.issues);
        assert!((hi_int.alpha_summary.min_effective_scale.unwrap() - 1.0).abs() < 1e-9);
        assert!((hi_int.alpha_summary.min_alpha_r_ratio.unwrap() - 0.0625).abs() < 1e-9);

        let arch = inspect_adapter_architecture(std::slice::from_ref(&hi_layer), &BTreeMap::new());
        let sum = audit_adapter(std::slice::from_ref(&hi_layer));
        let card = crate::card::generate_model_card("rslora.safetensors", &arch, &sum, &hi_int);
        assert!(card.contains("alpha/sqrt(r)=1.00 (rsLoRA)"));
        assert!(!card.contains('\u{2014}'));

        // 2. Naive alpha=2*r with use_rslora=true at r=100 -> alpha/sqrt(r) = 20.0 (> 16.0) + .alpha ambiguity
        let over_layer = LoraLayer {
            base_name: "model.layers.0.mlp.up_proj".to_string(),
            a_key: "model.layers.0.mlp.up_proj.lora_A.weight".to_string(),
            b_key: "model.layers.0.mlp.up_proj.lora_B.weight".to_string(),
            a: random_matrix(100, d_in, &mut rng, DType::F32),
            b: random_matrix(d_out, 100, &mut rng, DType::F32),
        };
        let over_tensors = layers_to_tensor_map(std::slice::from_ref(&over_layer));
        let mut over_alphas = BTreeMap::new();
        over_alphas.insert(over_layer.base_name.clone(), 200.0);
        let over_cfg = crate::safetensors::parse_peft_adapter_config(
            r#"{"peft_type":"LORA","r":100,"lora_alpha":200,"use_rslora":true}"#,
        )
        .expect("parse over rslora");
        let over_int = audit_adapter_integrity(
            std::slice::from_ref(&over_layer),
            &over_tensors,
            &over_alphas,
            &BTreeMap::new(),
            Some(&over_cfg),
        );
        assert!(over_int
            .issues
            .iter()
            .any(|i| i.code == "EXTREME_ALPHA_OVERSCALE" && i.message.contains("alpha/sqrt(r)")));
        assert!(over_int
            .issues
            .iter()
            .any(|i| i.code == "RSLORA_ALPHA_TENSOR_AMBIGUITY"));
        let sarif = crate::sarif::build_audit_sarif(None, Some(&over_int), "over.safetensors", "1.2.0");
        assert!(sarif.contains("\"ruleId\": \"LS021\""));
        assert!(sarif.contains("\"ruleId\": \"LS010\""));

        // 3. diff_adapters_full: rsLoRA (r=16, alpha=8 -> 8/4=2.0) vs standard LoRA (r=16, alpha=32 -> 32/16=2.0)
        let eq_layer = LoraLayer {
            base_name: "model.layers.0.self_attn.k_proj".to_string(),
            a_key: "model.layers.0.self_attn.k_proj.lora_A.weight".to_string(),
            b_key: "model.layers.0.self_attn.k_proj.lora_B.weight".to_string(),
            a: random_matrix(16, d_in, &mut rng, DType::F32),
            b: random_matrix(d_out, 16, &mut rng, DType::F32),
        };
        let cfg_rs = crate::safetensors::parse_peft_adapter_config(
            r#"{"r":16,"lora_alpha":8,"use_rslora":true}"#,
        )
        .unwrap();
        let cfg_std = crate::safetensors::parse_peft_adapter_config(
            r#"{"r":16,"lora_alpha":32,"use_rslora":false}"#,
        )
        .unwrap();
        let diff_sum = crate::diff::diff_adapters_full(
            std::slice::from_ref(&eq_layer),
            &BTreeMap::new(),
            &BTreeMap::new(),
            Some(&cfg_rs),
            None,
            std::slice::from_ref(&eq_layer),
            &BTreeMap::new(),
            &BTreeMap::new(),
            Some(&cfg_std),
            None,
        );
        assert!(diff_sum.use_rslora_a);
        assert!(!diff_sum.use_rslora_b);
        assert!(diff_sum.shared_layers[0].frobenius_delta < 1e-12);
        assert!(diff_sum.shared_layers[0].relative_drift < 1e-12);
        assert!((diff_sum.shared_layers[0].scale_a - 2.0).abs() < 1e-12);
        assert!((diff_sum.shared_layers[0].scale_b - 2.0).abs() < 1e-12);
    }

    #[test]
    fn test_pissa_residual_adapter_audit_diagnostics_and_sarif_ls022() {
        let mut rng = Rng64::new(12002);
        let d_out = 20;
        let d_in = 16;
        let r = 4;

        // 1. Unconverted PiSSA without pissa_init reference weights and non-unit scale (lora_alpha=8, r=4 -> 2.0)
        let active_l1 = LoraLayer {
            base_name: "base_model.model.layers.0.self_attn.q_proj".to_string(),
            a_key: "base_model.model.layers.0.self_attn.q_proj.lora_A.default.weight".to_string(),
            b_key: "base_model.model.layers.0.self_attn.q_proj.lora_B.default.weight".to_string(),
            a: random_matrix(r, d_in, &mut rng, DType::F32),
            b: random_matrix(d_out, r, &mut rng, DType::F32),
        };
        let pissa_cfg = crate::safetensors::parse_peft_adapter_config(
            r#"{"peft_type":"LORA","r":4,"lora_alpha":8,"init_lora_weights":"pissa_niter_4"}"#,
        )
        .expect("parse pissa config");
        assert!(pissa_cfg.is_pissa());

        let t_unconv = layers_to_tensor_map(std::slice::from_ref(&active_l1));
        let rep_unconv = audit_adapter_integrity(
            std::slice::from_ref(&active_l1),
            &t_unconv,
            &BTreeMap::new(),
            &BTreeMap::new(),
            Some(&pissa_cfg),
        );
        assert!(rep_unconv
            .issues
            .iter()
            .any(|i| i.code == "PISSA_UNCONVERTED_RESIDUAL_HAZARD"));
        assert!(rep_unconv
            .issues
            .iter()
            .any(|i| i.code == "PISSA_NON_UNIT_SCALING"));

        // 2. Multi-adapter safetensors containing both default active LoRA and pissa_init reference weights
        let active_l2 = LoraLayer {
            base_name: "base_model.model.layers.0.self_attn.v_proj".to_string(),
            a_key: "base_model.model.layers.0.self_attn.v_proj.lora_A.default.weight".to_string(),
            b_key: "base_model.model.layers.0.self_attn.v_proj.lora_B.default.weight".to_string(),
            a: random_matrix(r, d_in, &mut rng, DType::F32),
            b: random_matrix(d_out, r, &mut rng, DType::F32),
        };
        // Perturb l1 away from init, keep l2 identical to init (zero net task update!)
        let mut init_l1_b = active_l1.b.clone();
        for v in &mut init_l1_b.data {
            *v *= 0.75;
        }
        let mut multi_map = layers_to_tensor_map(&[active_l1, active_l2.clone()]);
        multi_map.insert(
            "base_model.model.layers.0.self_attn.q_proj.lora_A.pissa_init.weight".to_string(),
            t_unconv["base_model.model.layers.0.self_attn.q_proj.lora_A.default.weight"].clone(),
        );
        multi_map.insert(
            "base_model.model.layers.0.self_attn.q_proj.lora_B.pissa_init.weight".to_string(),
            init_l1_b,
        );
        multi_map.insert(
            "base_model.model.layers.0.self_attn.v_proj.lora_A.pissa_init.weight".to_string(),
            active_l2.a.clone(),
        );
        multi_map.insert(
            "base_model.model.layers.0.self_attn.v_proj.lora_B.pissa_init.weight".to_string(),
            active_l2.b.clone(),
        );

        let extracted_active = extract_lora_layers(&multi_map).expect("active layers");
        let extracted_pissa =
            crate::safetensors::extract_pissa_init_layers(&multi_map).expect("pissa_init layers");
        assert_eq!(extracted_active.len(), 2);
        assert_eq!(extracted_pissa.len(), 2);

        let unit_pissa_cfg = crate::safetensors::parse_peft_adapter_config(
            r#"{"peft_type":"LORA","r":4,"lora_alpha":4,"init_lora_weights":"pissa"}"#,
        )
        .unwrap();
        let rep_bundled = audit_adapter_integrity(
            &extracted_active,
            &multi_map,
            &BTreeMap::new(),
            &BTreeMap::new(),
            Some(&unit_pissa_cfg),
        );
        assert_eq!(rep_bundled.pissa_diagnostics.len(), 2);
        assert!(rep_bundled
            .issues
            .iter()
            .any(|i| i.code == "PISSA_ZERO_TASK_UPDATE"));
        assert!(!rep_bundled
            .issues
            .iter()
            .any(|i| i.code == "PISSA_UNCONVERTED_RESIDUAL_HAZARD"));

        let sarif =
            crate::sarif::build_audit_sarif(None, Some(&rep_bundled), "pissa.safetensors", "1.2.0");
        assert!(sarif.contains("\"ruleId\": \"LS022\""));
        let json_out = crate::audit::render_audit_json(
            "pissa.safetensors",
            &audit_adapter(&extracted_active),
            &rep_bundled,
        );
        assert!(json_out.contains("\"pissa_diagnostics\": ["));
        assert!(!json_out.contains('\u{2014}'));
    }

    #[test]
    fn test_diff_adapters_full_normalizes_pissa_residual_against_converted_lora() {
        let mut rng = Rng64::new(12003);
        let d_out = 22;
        let d_in = 18;
        let r = 4;

        let pissa_init = LoraLayer {
            base_name: "model.layers.0.mlp.down_proj".to_string(),
            a_key: "model.layers.0.mlp.down_proj.lora_A.pissa_init.weight".to_string(),
            b_key: "model.layers.0.mlp.down_proj.lora_B.pissa_init.weight".to_string(),
            a: random_matrix(r, d_in, &mut rng, DType::F32),
            b: random_matrix(d_out, r, &mut rng, DType::F32),
        };
        let active_pissa = LoraLayer {
            base_name: "model.layers.0.mlp.down_proj".to_string(),
            a_key: "model.layers.0.mlp.down_proj.lora_A.default.weight".to_string(),
            b_key: "model.layers.0.mlp.down_proj.lora_B.default.weight".to_string(),
            a: random_matrix(r, d_in, &mut rng, DType::F32),
            b: random_matrix(d_out, r, &mut rng, DType::F32),
        };

        // PEFT convert_pissa_to_lora produces rank-2r standard LoRA equal to B*A - B_init*A_init
        let converted_lora = crate::safetensors::subtract_pissa_residual_layer(
            &active_pissa,
            1.0,
            &pissa_init,
            1.0,
        )
        .expect("subtract pissa residual");
        assert_eq!(converted_lora.rank(), 2 * r);

        let mut pissa_map_a = BTreeMap::new();
        pissa_map_a.insert(pissa_init.base_name.clone(), pissa_init);

        let cfg_pissa = crate::safetensors::parse_peft_adapter_config(
            r#"{"r":4,"lora_alpha":4,"init_lora_weights":"pissa"}"#,
        )
        .unwrap();
        let cfg_conv = crate::safetensors::parse_peft_adapter_config(
            r#"{"r":8,"lora_alpha":8,"init_lora_weights":true}"#,
        )
        .unwrap();

        let summary = crate::diff::diff_adapters_full(
            &[active_pissa],
            &BTreeMap::new(),
            &BTreeMap::new(),
            Some(&cfg_pissa),
            Some(&pissa_map_a),
            &[converted_lora],
            &BTreeMap::new(),
            &BTreeMap::new(),
            Some(&cfg_conv),
            None,
        );
        assert_eq!(summary.pissa_layers_normalized, 1);
        let rep = &summary.shared_layers[0];
        assert_eq!(rep.rank_a, 4);
        assert_eq!(rep.rank_b, 8);
        assert!(rep.pissa_residual_subtracted);
        assert!(
            rep.frobenius_delta < 1e-12,
            "expected zero net base-model delta between pissa+init and converted rank-2r LoRA, got {}",
            rep.frobenius_delta
        );
        assert!(rep.relative_drift < 1e-12);
    }

    #[test]
    fn test_quantization_sqnr_ordering_and_exact_low_rank_difference_norms() {
        use crate::quant::{
            analyze_layer_quantization, exact_low_rank_difference_norms,
            quantize_dequantize_tensor, QuantScheme, NF4_CODEBOOK,
        };

        // Verify NF4 codebook monotonicity, [-1, 1] endpoints, and exact zero at index 7
        assert_eq!(NF4_CODEBOOK[0], -1.0);
        assert_eq!(NF4_CODEBOOK[7], 0.0);
        assert_eq!(NF4_CODEBOOK[15], 1.0);
        for w in NF4_CODEBOOK.windows(2) {
            assert!(w[0] < w[1], "NF4 codebook must be strictly increasing");
        }

        let mut rng = Rng64::new(13001);
        let d_out = 96;
        let d_in = 128;
        let r = 8;
        let layer = LoraLayer {
            base_name: "model.layers.0.self_attn.q_proj".to_string(),
            a_key: "model.layers.0.self_attn.q_proj.lora_A.weight".to_string(),
            b_key: "model.layers.0.self_attn.q_proj.lora_B.weight".to_string(),
            a: random_matrix(r, d_in, &mut rng, DType::F32),
            b: random_matrix(d_out, r, &mut rng, DType::F32),
        };

        // Verify exact_low_rank_difference_norms matches full dense SVD to machine precision
        let a_nf4 = quantize_dequantize_tensor(&layer.a, QuantScheme::Nf4, 64);
        let b_nf4 = quantize_dequantize_tensor(&layer.b, QuantScheme::Nf4, 64);
        let scale = 2.0;
        let (spec_fast, frob_fast) =
            exact_low_rank_difference_norms(&layer.b, &layer.a, scale, &b_nf4, &a_nf4, scale);

        let dw_orig = matmul(&layer.b, &layer.a);
        let dw_q = matmul(&b_nf4, &a_nf4);
        let mut dense_diff = Tensor2D::zeros(d_out, d_in, DType::F32);
        for i in 0..dense_diff.data.len() {
            dense_diff.data[i] = scale * (dw_orig.data[i] - dw_q.data[i]);
        }
        let dense_frob = dense_diff.frobenius_norm_sq().sqrt();
        let dense_spec = jacobi_svd(&dense_diff).sigma[0];

        assert!(
            (frob_fast - dense_frob).abs() < 1e-11,
            "Frobenius diff mismatch: fast={} vs dense={}",
            frob_fast,
            dense_frob
        );
        assert!(
            (spec_fast - dense_spec).abs() < 1e-11,
            "Spectral diff mismatch: fast={} vs dense={}",
            spec_fast,
            dense_spec
        );

        // On Gaussian weights, information-theoretically optimal NF4 outperforms FP4 and INT4, while INT8 dominates 4-bit
        let rep_int8 = analyze_layer_quantization(&layer, scale, QuantScheme::Int8, 64);
        let rep_nf4 = analyze_layer_quantization(&layer, scale, QuantScheme::Nf4, 64);
        let rep_fp4 = analyze_layer_quantization(&layer, scale, QuantScheme::Fp4, 64);
        let rep_int4 = analyze_layer_quantization(&layer, scale, QuantScheme::Int4, 64);

        assert!(
            rep_int8.sqnr_db > 35.0 && rep_int8.sqnr_db > rep_nf4.sqnr_db + 15.0,
            "INT8 SQNR ({:.2} dB) must far exceed NF4 ({:.2} dB)",
            rep_int8.sqnr_db,
            rep_nf4.sqnr_db
        );
        assert!(
            rep_nf4.sqnr_db > rep_fp4.sqnr_db,
            "NF4 SQNR ({:.2} dB) should exceed FP4 ({:.2} dB) on Gaussian weights",
            rep_nf4.sqnr_db,
            rep_fp4.sqnr_db
        );
        assert!(
            rep_nf4.sqnr_db > rep_int4.sqnr_db,
            "NF4 SQNR ({:.2} dB) should exceed symmetric INT4 ({:.2} dB) on Gaussian weights",
            rep_nf4.sqnr_db,
            rep_int4.sqnr_db
        );
    }

    #[test]
    fn test_imbalanced_factor_quantization_loss_and_svd_rebalancing_ls023() {
        use crate::quant::{analyze_adapter_quantization, QuantScheme};

        let mut rng = Rng64::new(13002);
        let d_out = 64;
        let d_in = 64;
        let r = 8;

        // Construct a layer with severe channel outlier in A (100x spike on rank-0 col-0) while B has opposite compensation
        let mut a = random_matrix(r, d_in, &mut rng, DType::F32);
        let mut b = random_matrix(d_out, r, &mut rng, DType::F32);
        // Scale rank-0 row of A up by 80x and rank-0 col of B down by 80x so B*A is unchanged,
        // creating severe blockwise quantization dynamic-range starvation in A!
        for c in 0..d_in {
            a.set(0, c, a.get(0, c) * 0.02);
        }
        a.set(0, 0, 25.0);
        for row in 0..d_out {
            b.set(row, 0, b.get(row, 0) * 0.02);
        }

        let outlier_layer = LoraLayer {
            base_name: "model.layers.0.mlp.up_proj".to_string(),
            a_key: "model.layers.0.mlp.up_proj.lora_A.weight".to_string(),
            b_key: "model.layers.0.mlp.up_proj.lora_B.weight".to_string(),
            a,
            b,
        };

        // Also construct an extreme single-spike matrix that destroys 4-bit INT4 precision (>25% error)
        let mut a_spike = random_matrix(r, d_in, &mut rng, DType::F32);
        let b_spike = random_matrix(d_out, r, &mut rng, DType::F32);
        a_spike.set(0, 0, 500.0); // dwarfs all other 63 weights in block 0 so they round to 0
        for r_idx in 1..r {
            a_spike.set(r_idx, 0, 500.0);
        }
        let severe_layer = LoraLayer {
            base_name: "model.layers.0.mlp.down_proj".to_string(),
            a_key: "model.layers.0.mlp.down_proj.lora_A.weight".to_string(),
            b_key: "model.layers.0.mlp.down_proj.lora_B.weight".to_string(),
            a: a_spike,
            b: b_spike,
        };

        let (q_summary, q_issues) = analyze_adapter_quantization(
            &[outlier_layer.clone(), severe_layer.clone()],
            &[1.0, 1.0],
            QuantScheme::Int4,
            64,
        );
        assert_eq!(q_summary.layers.len(), 2);
        assert!(
            q_summary.layers[0].balanced_relative_frobenius_error
                < q_summary.layers[0].relative_frobenius_error,
            "SVD rebalancing should reduce quantization error on imbalanced outlier factors ({} vs {})",
            q_summary.layers[0].balanced_relative_frobenius_error,
            q_summary.layers[0].relative_frobenius_error
        );
        assert!(
            !q_issues.is_empty(),
            "Expected LS023 quantization reconstruction warning/critical issue"
        );

        let layers = vec![outlier_layer, severe_layer];
        let tensors = layers_to_tensor_map(&layers);
        let mut integrity = audit_adapter_integrity(
            &layers,
            &tensors,
            &BTreeMap::new(),
            &BTreeMap::new(),
            None,
        );
        integrity.quant_summary = Some(q_summary);
        integrity.issues.extend(q_issues);
        integrity.status = "ISSUES_DETECTED";

        let sarif = crate::sarif::build_audit_sarif(
            None,
            Some(&integrity),
            "qlora_adapter.safetensors",
            "1.3.0",
        );
        assert!(sarif.contains("\"ruleId\": \"LS023\""));
        let json = crate::audit::render_audit_json(
            "qlora_adapter.safetensors",
            &audit_adapter(&layers),
            &integrity,
        );
        assert!(json.contains("\"quantization_diagnostics\": {"));
        assert!(json.contains("\"scheme\": \"INT4\""));
        assert!(!json.contains('\u{2014}'));
    }

    #[test]
    fn test_safetensors_bnb_nf4_double_quant_and_int8_scb_dequantization_and_metadata_audit() {
        use crate::quant::NF4_CODEBOOK;

        // 1. Build a packed 4-bit bitsandbytes__nf4 lora_A (shape 4 x 16 = 64 elements -> 32 packed U8 bytes)
        //    with double quantization (U8 absmax + nested_absmax + nested_quant_map)
        let rows_a = 4;
        let cols_a = 16;
        let mut expected_a = vec![0.0_f64; rows_a * cols_a];
        let mut packed_u8 = vec![0.0_f64; (rows_a * cols_a) / 2];
        let blocksize = 16; // 4 blocks of 16 elements
        let block_scales = [0.5_f64, 1.25, 0.75, 2.0];

        for i in 0..packed_u8.len() {
            let idx_hi = (2 * i) % 16;
            let idx_lo = (2 * i + 5) % 16;
            let byte = ((idx_hi << 4) | idx_lo) as u8;
            packed_u8[i] = byte as f64;
            let s_hi = block_scales[(2 * i) / blocksize];
            let s_lo = block_scales[(2 * i + 1) / blocksize];
            expected_a[2 * i] = NF4_CODEBOOK[idx_hi] * s_hi;
            expected_a[2 * i + 1] = NF4_CODEBOOK[idx_lo] * s_lo;
        }

        // Double quantization metadata: nested_quant_map[k] = k / 100.0, nested_absmax = [1.0], nested_offset = 0.0
        // Then block_scales [0.50, 1.25, 0.75, 2.00] correspond exactly to U8 indices [50, 125, 75, 200]!
        let mut nested_qmap = vec![0.0_f64; 256];
        for k in 0..256 {
            nested_qmap[k] = (k as f64) / 100.0;
        }
        let u8_absmax = vec![50.0, 125.0, 75.0, 200.0];
        let state_json = br#"{"quant_type":"nf4","blocksize":16,"shape":[4,16],"dtype":"float16"}"#;
        let state_u8: Vec<f64> = state_json.iter().map(|&b| b as f64).collect();

        // 2. Build an LLM.int8() row-wise SCB quantized lora_B (shape 12 x 4)
        let rows_b = 12;
        let cols_b = 4;
        let mut i8_b_vals = vec![0.0_f64; rows_b * cols_b];
        let mut scb_vals = vec![0.0_f64; rows_b];
        let mut expected_b = vec![0.0_f64; rows_b * cols_b];
        for r in 0..rows_b {
            let row_scb = 0.4 + 0.1 * (r as f64);
            scb_vals[r] = row_scb;
            for c in 0..cols_b {
                let q = ((r as i32 * 11 - c as i32 * 23) % 100) as f64;
                i8_b_vals[r * cols_b + c] = q;
                expected_b[r * cols_b + c] = q * (row_scb / 127.0);
            }
        }

        let mut map = BTreeMap::new();
        map.insert(
            "model.layers.0.self_attn.q_proj.lora_A.weight".to_string(),
            Tensor2D {
                rows: packed_u8.len(),
                cols: 1,
                dtype: DType::U8,
                data: packed_u8,
            },
        );
        map.insert(
            "model.layers.0.self_attn.q_proj.lora_A.weight.absmax".to_string(),
            Tensor2D {
                rows: 4,
                cols: 1,
                dtype: DType::U8,
                data: u8_absmax,
            },
        );
        map.insert(
            "model.layers.0.self_attn.q_proj.lora_A.weight.nested_absmax".to_string(),
            Tensor2D {
                rows: 1,
                cols: 1,
                dtype: DType::F32,
                data: vec![1.0],
            },
        );
        map.insert(
            "model.layers.0.self_attn.q_proj.lora_A.weight.nested_quant_map".to_string(),
            Tensor2D {
                rows: 256,
                cols: 1,
                dtype: DType::F32,
                data: nested_qmap,
            },
        );
        map.insert(
            "model.layers.0.self_attn.q_proj.lora_A.weight.quant_state.bitsandbytes__nf4".to_string(),
            Tensor2D {
                rows: state_u8.len(),
                cols: 1,
                dtype: DType::U8,
                data: state_u8,
            },
        );
        map.insert(
            "model.layers.0.self_attn.q_proj.lora_B.weight".to_string(),
            Tensor2D {
                rows: rows_b,
                cols: cols_b,
                dtype: DType::I8,
                data: i8_b_vals,
            },
        );
        map.insert(
            "model.layers.0.self_attn.q_proj.lora_B.SCB".to_string(),
            Tensor2D {
                rows: rows_b,
                cols: 1,
                dtype: DType::F32,
                data: scb_vals,
            },
        );

        // Roundtrip through real binary .safetensors file!
        let tmp_path = std::env::temp_dir().join("test_bnb_nf4_i8_lora.safetensors");
        write_safetensors(&tmp_path, &map).expect("write quantized safetensors");
        let reloaded = read_safetensors(&tmp_path).expect("read quantized safetensors");
        let _ = std::fs::remove_file(&tmp_path);

        let layers = extract_lora_layers(&reloaded).expect("dequantize and extract LoRA layers");
        assert_eq!(layers.len(), 1);
        assert_eq!(layers[0].rank(), 4);
        assert_eq!(layers[0].d_in(), 16);
        assert_eq!(layers[0].d_out(), 12);

        for (got, exp) in layers[0].a.data.iter().zip(expected_a.iter()) {
            assert!((got - exp).abs() < 1e-6, "NF4 dequant mismatch: {} vs {}", got, exp);
        }
        for (got, exp) in layers[0].b.data.iter().zip(expected_b.iter()) {
            assert!((got - exp).abs() < 1e-6, "I8 SCB dequant mismatch: {} vs {}", got, exp);
        }

        // Healthy quantized metadata should pass integrity audit with zero issues
        let healthy_rep = audit_adapter_integrity(
            &layers,
            &reloaded,
            &BTreeMap::new(),
            &BTreeMap::new(),
            None,
        );
        assert_eq!(healthy_rep.status, "HEALTHY", "unexpected issues: {:?}", healthy_rep.issues);

        // Corrupting SCB with 0.0 and quant_map with wrong length triggers LS023 metadata issues
        let mut corrupt_map = reloaded.clone();
        corrupt_map
            .get_mut("model.layers.0.self_attn.q_proj.lora_B.SCB")
            .unwrap()
            .data[0] = 0.0;
        corrupt_map.insert(
            "model.layers.0.self_attn.q_proj.lora_A.weight.quant_map".to_string(),
            Tensor2D {
                rows: 8,
                cols: 1,
                dtype: DType::F32,
                data: vec![0.1; 8],
            },
        );
        let corrupt_rep = audit_adapter_integrity(
            &layers,
            &corrupt_map,
            &BTreeMap::new(),
            &BTreeMap::new(),
            None,
        );
        assert!(corrupt_rep
            .issues
            .iter()
            .any(|i| i.code == "QUANT_ZERO_OR_NAN_ABSMAX"));
        assert!(corrupt_rep
            .issues
            .iter()
            .any(|i| i.code == "QUANT_CORRUPT_METADATA"));
    }

    #[test]
    fn test_svd_truncation_error_bounds_weyl_certificate_and_error_bound_compression_ls024() {
        use crate::audit::{audit_truncation_error_bounds, compute_truncation_bounds};
        use crate::linalg::{qr_core_svd_truncated, svd_to_lora_factors};
        use crate::quant::{compute_combined_truncation_quant_certificate, QuantScheme};

        // Construct a rank-6 layer with controlled singular spectrum [10.0, 5.0, 2.0, 1.995, 0.5, 0.1]
        // Note the nearly-degenerate singular doublet at k=3: sigma_3 = 2.0, sigma_4 = 1.995 (local gap = 0.25%)!
        let mut rng = Rng64::new(13004);
        let d_out = 48;
        let d_in = 48;
        let r = 6;
        let base_b = random_matrix(d_out, r, &mut rng, DType::F32);
        let base_a = random_matrix(r, d_in, &mut rng, DType::F32);
        let mut svd = qr_core_svd_truncated(&base_b, &base_a, r);
        svd.sigma = vec![10.0, 5.0, 2.0, 1.995, 0.5, 0.1];
        let (b, a) = svd_to_lora_factors(&svd, r, DType::F32);
        let layer = LoraLayer {
            base_name: "model.layers.0.self_attn.o_proj".to_string(),
            a_key: "model.layers.0.self_attn.o_proj.lora_A.weight".to_string(),
            b_key: "model.layers.0.self_attn.o_proj.lora_B.weight".to_string(),
            a,
            b,
        };

        // 1. Verify exact Eckart-Young spectral/Frobenius, Mirsky nuclear, and Wedin gap at k=3
        let bounds_k3 = compute_truncation_bounds(&svd.sigma, 3);
        let expected_frob_err = (1.995_f64.powi(2) + 0.5_f64.powi(2) + 0.1_f64.powi(2)).sqrt();
        let expected_nuc_err = 1.995 + 0.5 + 0.1;
        assert!((bounds_k3.spectral_error - 1.995).abs() < 1e-12);
        assert!((bounds_k3.relative_spectral_error - 0.1995).abs() < 1e-12);
        assert!((bounds_k3.frobenius_error - expected_frob_err).abs() < 1e-12);
        assert!((bounds_k3.nuclear_error - expected_nuc_err).abs() < 1e-12);
        assert!((bounds_k3.wedin_spectral_gap - (0.005 / 10.0)).abs() < 1e-12);

        // 2. Verify Weyl lower bound & triangle inequality sandwich certificate under NF4 quantization
        let cert =
            compute_combined_truncation_quant_certificate(&layer, 1.0, 3, QuantScheme::Nf4, 64);
        assert!(
            cert.exact_total_spectral_error + 1e-12 >= cert.weyl_spectral_lower_bound,
            "Weyl lower bound violated: exact={} < weyl={}",
            cert.exact_total_spectral_error,
            cert.weyl_spectral_lower_bound
        );
        assert!(
            cert.exact_total_spectral_error <= cert.triangle_spectral_upper_bound + 1e-12,
            "Triangle spectral upper bound violated: exact={} > upper={}",
            cert.exact_total_spectral_error,
            cert.triangle_spectral_upper_bound
        );
        assert!(
            cert.exact_total_frobenius_error <= cert.triangle_frobenius_upper_bound + 1e-12,
            "Triangle Frobenius upper bound violated: exact={} > upper={}",
            cert.exact_total_frobenius_error,
            cert.triangle_frobenius_upper_bound
        );

        // 3. Verify CompressionPolicy::ErrorBound selects smallest rank meeting spectral <= 0.06 (sigma_{k+1}/10 <= 0.06 -> k=4 where sigma_5=0.5 -> 0.05)
        let (_comp_layer, metric) = compress_layer(
            &layer,
            CompressionPolicy::ErrorBound {
                max_frobenius_error: Some(0.06),
                max_spectral_error: Some(0.06),
                min_rank: 1,
                max_rank: 6,
            },
        );
        assert_eq!(metric.compressed_rank, 4);
        assert!(metric.relative_spectral_error <= 0.06);
        assert!(metric.relative_frobenius_error <= 0.06);
        assert!(metric.relative_nuclear_error > 0.0);

        // 4. Verify LS024 audit issues at k=3 with max_error_tol=0.10 (triggers both SVD_TRUNCATION_BOUND_EXCEEDED and SVD_DEGENERATE_SPECTRAL_GAP)
        let summary = audit_adapter(std::slice::from_ref(&layer));
        let trunc_issues = audit_truncation_error_bounds(&summary.layers, Some(3), Some(0.10));
        assert!(trunc_issues
            .iter()
            .any(|i| i.code == "SVD_TRUNCATION_BOUND_EXCEEDED"));
        assert!(trunc_issues
            .iter()
            .any(|i| i.code == "SVD_DEGENERATE_SPECTRAL_GAP"));

        let tensors = layers_to_tensor_map(std::slice::from_ref(&layer));
        let mut integrity = audit_adapter_integrity(
            std::slice::from_ref(&layer),
            &tensors,
            &BTreeMap::new(),
            &BTreeMap::new(),
            None,
        );
        integrity.issues.extend(trunc_issues);
        integrity.status = "ISSUES_DETECTED";

        let sarif =
            crate::sarif::build_audit_sarif(Some(&summary), Some(&integrity), "trunc.safetensors", "1.4.0");
        assert!(sarif.contains("\"ruleId\": \"LS024\""));
    }

    // ---- Subspace Drift Tests ----------------------------------------------------------------

    #[test]
    fn test_subspace_drift_stagnant_detection() {
        use crate::diff::{compute_layer_subspace_drift, SubspaceDriftClass};

        let mut rng = Rng64::new(5001);
        let d = 64;
        let r = 8;

        // Two identical layers -> zero principal angle -> STAGNANT
        let a = random_matrix(r, d, &mut rng, DType::F32);
        let b = random_matrix(d, r, &mut rng, DType::F32);

        let layer_a = LoraLayer {
            base_name: "model.layers.0.q_proj".to_string(),
            a_key: "model.layers.0.q_proj.lora_A.weight".to_string(),
            b_key: "model.layers.0.q_proj.lora_B.weight".to_string(),
            a: a.clone(),
            b: b.clone(),
        };
        let layer_b = layer_a.clone();

        let report = compute_layer_subspace_drift(&layer_a, 1.0, &layer_b, 1.0).unwrap();
        assert_eq!(report.classification, SubspaceDriftClass::Stagnant);
        assert!(report.mean_angle_deg < 1.0, "Identical layers should have near-zero angle, got {}", report.mean_angle_deg);
        assert!(report.issues.contains(&"LS027"));
        assert!(!report.issues.contains(&"LS028"));
    }

    #[test]
    fn test_subspace_drift_healthy_detection() {
        use crate::diff::{compute_layer_subspace_drift, SubspaceDriftClass};

        let mut rng_a = Rng64::new(5002);
        let mut rng_b = Rng64::new(5003);
        let d = 128;
        let r = 8;

        // Two random independent adapters -> substantial principal angles -> likely HEALTHY or HEAVY
        let layer_a = LoraLayer {
            base_name: "model.layers.0.q_proj".to_string(),
            a_key: "model.layers.0.q_proj.lora_A.weight".to_string(),
            b_key: "model.layers.0.q_proj.lora_B.weight".to_string(),
            a: random_matrix(r, d, &mut rng_a, DType::F32),
            b: random_matrix(d, r, &mut rng_a, DType::F32),
        };
        let layer_b = LoraLayer {
            base_name: "model.layers.0.q_proj".to_string(),
            a_key: "model.layers.0.q_proj.lora_A.weight".to_string(),
            b_key: "model.layers.0.q_proj.lora_B.weight".to_string(),
            a: random_matrix(r, d, &mut rng_b, DType::F32),
            b: random_matrix(d, r, &mut rng_b, DType::F32),
        };

        let report = compute_layer_subspace_drift(&layer_a, 1.0, &layer_b, 1.0).unwrap();
        // For random d=128, r=8: expected principal angles ~arcsin(sqrt(r/d)) ~ arcsin(0.25) ~ 14.5 deg
        // Could be healthy or heavy depending on seed; just verify it's not stagnant or zero
        assert!(report.mean_angle_deg > 2.0, "Random adapters should have non-zero principal angles, got {}", report.mean_angle_deg);
        assert_eq!(report.rank_a, r);
        assert_eq!(report.rank_b, r);
    }

    #[test]
    fn test_subspace_drift_catastrophic_detection() {
        use crate::diff::{compute_layer_subspace_drift, SubspaceDriftClass};
        use crate::linalg::thin_qr_columns;

        let mut rng = Rng64::new(5004);
        let d = 64;
        let r = 4;

        // Build two completely orthogonal rank-4 subspaces using QR
        let raw = random_matrix(d, 2 * r, &mut rng, DType::F32);
        let (all_cols, _) = thin_qr_columns(&raw);

        // Adapter A uses columns 0..r
        let mut b_a = Tensor2D::zeros(d, r, DType::F32);
        let mut a_a = Tensor2D::zeros(r, d, DType::F32);
        for j in 0..r {
            for row in 0..d {
                b_a.set(row, j, all_cols[j][row]);
                a_a.set(j, row, all_cols[j][row]);
            }
        }

        // Adapter B uses columns r..2r (orthogonal to A's space)
        let mut b_b = Tensor2D::zeros(d, r, DType::F32);
        let mut a_b = Tensor2D::zeros(r, d, DType::F32);
        for j in 0..r {
            for row in 0..d {
                b_b.set(row, j, all_cols[r + j][row]);
                a_b.set(j, row, all_cols[r + j][row]);
            }
        }

        let layer_a = LoraLayer {
            base_name: "model.layers.0.q_proj".to_string(),
            a_key: "model.layers.0.q_proj.lora_A.weight".to_string(),
            b_key: "model.layers.0.q_proj.lora_B.weight".to_string(),
            a: a_a, b: b_a,
        };
        let layer_b = LoraLayer {
            base_name: "model.layers.0.q_proj".to_string(),
            a_key: "model.layers.0.q_proj.lora_A.weight".to_string(),
            b_key: "model.layers.0.q_proj.lora_B.weight".to_string(),
            a: a_b, b: b_b,
        };

        let report = compute_layer_subspace_drift(&layer_a, 1.0, &layer_b, 1.0).unwrap();
        // Orthogonal subspaces -> principal angles = 90 deg -> CATASTROPHIC_DRIFT
        assert_eq!(report.classification, SubspaceDriftClass::CatastrophicDrift,
            "Orthogonal subspaces should trigger catastrophic drift, got {} deg", report.mean_angle_deg);
        assert!(report.mean_angle_deg > 60.0, "Expected >60 deg, got {}", report.mean_angle_deg);
        assert!(report.issues.contains(&"LS028"));
    }

    #[test]
    fn test_subspace_drift_dimension_mismatch_returns_none() {
        use crate::diff::compute_layer_subspace_drift;

        let mut rng = Rng64::new(5005);
        let layer_a = LoraLayer {
            base_name: "model.q_proj".to_string(),
            a_key: "model.q_proj.lora_A.weight".to_string(),
            b_key: "model.q_proj.lora_B.weight".to_string(),
            a: random_matrix(4, 32, &mut rng, DType::F32),
            b: random_matrix(32, 4, &mut rng, DType::F32),
        };
        let layer_b = LoraLayer {
            base_name: "model.q_proj".to_string(),
            a_key: "model.q_proj.lora_A.weight".to_string(),
            b_key: "model.q_proj.lora_B.weight".to_string(),
            a: random_matrix(4, 64, &mut rng, DType::F32), // different d_in
            b: random_matrix(64, 4, &mut rng, DType::F32),
        };
        assert!(compute_layer_subspace_drift(&layer_a, 1.0, &layer_b, 1.0).is_none());
    }

    #[test]
    fn test_subspace_drift_summary_aggregation() {
        use crate::diff::{analyze_subspace_drift, SubspaceDriftClass};
        use std::collections::BTreeMap;

        let mut rng1 = Rng64::new(5006);
        let mut rng2 = Rng64::new(5007);
        let d = 64;
        let r = 4;

        let make_layer = |name: &str, rng: &mut Rng64| LoraLayer {
            base_name: name.to_string(),
            a_key: format!("{}.lora_A.weight", name),
            b_key: format!("{}.lora_B.weight", name),
            a: random_matrix(r, d, rng, DType::F32),
            b: random_matrix(d, r, rng, DType::F32),
        };

        // Adapter A: two layers
        let layers_a = vec![
            make_layer("model.layers.0.q_proj", &mut rng1),
            make_layer("model.layers.1.q_proj", &mut rng1),
        ];
        // Adapter B: same keys but different weights
        let layers_b = vec![
            make_layer("model.layers.0.q_proj", &mut rng2),
            make_layer("model.layers.1.q_proj", &mut rng2),
        ];

        let alphas: BTreeMap<String, f64> = BTreeMap::new();
        let summary = analyze_subspace_drift(&layers_a, &alphas, &layers_b, &alphas);
        assert_eq!(summary.total_shared_layers, 2);
        assert_eq!(summary.stagnant_layers + summary.healthy_layers + summary.heavy_drift_layers + summary.catastrophic_drift_layers, 2);
    }

    #[test]
    fn test_subspace_drift_json_render() {
        use crate::diff::{analyze_subspace_drift, render_subspace_drift_json};
        use std::collections::BTreeMap;

        let mut rng1 = Rng64::new(5008);
        let mut rng2 = Rng64::new(5009);

        let layers_a = vec![LoraLayer {
            base_name: "model.q_proj".to_string(),
            a_key: "model.q_proj.lora_A.weight".to_string(),
            b_key: "model.q_proj.lora_B.weight".to_string(),
            a: random_matrix(4, 32, &mut rng1, DType::F32),
            b: random_matrix(32, 4, &mut rng1, DType::F32),
        }];
        let layers_b = vec![LoraLayer {
            base_name: "model.q_proj".to_string(),
            a_key: "model.q_proj.lora_A.weight".to_string(),
            b_key: "model.q_proj.lora_B.weight".to_string(),
            a: random_matrix(4, 32, &mut rng2, DType::F32),
            b: random_matrix(32, 4, &mut rng2, DType::F32),
        }];

        let alphas: BTreeMap<String, f64> = BTreeMap::new();
        let summary = analyze_subspace_drift(&layers_a, &alphas, &layers_b, &alphas);
        let json = render_subspace_drift_json(&summary);
        assert!(json.contains("\"total_shared_layers\""));
        assert!(json.contains("\"mean_angle_deg\""));
        assert!(json.contains("\"classification\""));
    }
}

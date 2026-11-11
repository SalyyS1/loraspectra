pub mod audit;
pub mod compress;
pub mod linalg;
pub mod merge;
pub mod safetensors;

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
    use crate::audit::{audit_layer, LayerHealth};
    use crate::compress::{compress_layer, CompressionPolicy};
    use crate::linalg::{
        jacobi_svd, matmul, qr_core_singular_values, thin_qr_columns,
    };
    use crate::merge::{
        analyze_layer_interference, merge_adapters_with_block_weights, merge_adapters_with_options,
        merge_layer, merge_multiple_adapters, parse_layer_weights, MergeStrategy,
    };
    use crate::safetensors::{
        bake_adapter_scaling, bf16_to_f32, cast_layers_dtype, extract_lora_alphas,
        extract_lora_layers, f16_to_f32, f32_to_bf16, f32_to_f16, insert_unit_alpha_tensors,
        inspect_adapter_architecture, layers_to_tensor_map, read_safetensors,
        write_peft_rank_pattern_config, write_safetensors, DType, LoraLayer, Tensor2D,
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
}

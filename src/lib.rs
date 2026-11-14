pub mod audit;
pub mod card;
pub mod compress;
pub mod diff;
pub mod linalg;
pub mod merge;
pub mod remote;
pub mod safetensors;
pub mod sarif;

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
}

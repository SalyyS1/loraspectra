use crate::audit::{AdapterAuditSummary, AdapterIntegrityReport, IssueSeverity, LayerHealth};
use crate::knots::KnotsSummary;
use crate::lora_ga::LoraGaSummary;
use crate::merge_check::MultiMergeCheckSummary;
use crate::ortho_proj::OrthoProjectSummary;
use crate::ties_dare::TiesDareComposeSummary;

pub struct SarifRuleDefinition {
    pub id: &'static str,
    pub name: &'static str,
    pub short_description: &'static str,
    pub full_description: &'static str,
    pub default_level: &'static str,
}

pub const LORASPECTRA_SARIF_RULES: &[SarifRuleDefinition] = &[
    SarifRuleDefinition {
        id: "LS001",
        name: "DeadLoraLayer",
        short_description: "LoRA layer has zero or near-zero Frobenius norm",
        full_description: "The LoRA adapter layer has an uninitialized or dead weight matrix with zero/near-zero Frobenius norm, indicating frozen gradients or failed training updates.",
        default_level: "error",
    },
    SarifRuleDefinition {
        id: "LS002",
        name: "ZeroNormEmbeddingRows",
        short_description: "Saved token embedding has zero-norm rows",
        full_description: "Saved token embeddings or unembedding matrices contain all-zero rows. Newly added special tokens (e.g. tool calling or chat markers) never received gradients during fine-tuning.",
        default_level: "error",
    },
    SarifRuleDefinition {
        id: "LS003",
        name: "DeadLoraBCols",
        short_description: "LoRA B matrix contains all-zero columns",
        full_description: "Columns of the lora_B projection matrix are zeroed out, effectively reducing the active parameter capacity below nominal rank.",
        default_level: "warning",
    },
    SarifRuleDefinition {
        id: "LS004",
        name: "DeadLoraARows",
        short_description: "LoRA A matrix contains all-zero rows",
        full_description: "Rows of the lora_A projection matrix are zeroed out, disconnecting input channels from adapter rank subspaces.",
        default_level: "warning",
    },
    SarifRuleDefinition {
        id: "LS005",
        name: "MissingConfigAlpha",
        short_description: "adapter_config.json is missing lora_alpha",
        full_description: "PEFT configuration lacks lora_alpha definition while .alpha tensors are absent from checkpoint weights, risking undefined inference scaling.",
        default_level: "warning",
    },
    SarifRuleDefinition {
        id: "LS006",
        name: "MissingAlphaTensors",
        short_description: "Partial or missing .alpha tensors across adapter layers",
        full_description: "Some layers specify explicit .alpha scalar tensors while others omit them, causing inconsistent scaling during multi-layer merging.",
        default_level: "warning",
    },
    SarifRuleDefinition {
        id: "LS007",
        name: "MismatchedAlphaConfig",
        short_description: "Conflicting scaling between .alpha tensors and adapter_config.json",
        full_description: "Values in .alpha weight tensors contradict the lora_alpha or alpha_pattern entries in adapter_config.json.",
        default_level: "warning",
    },
    SarifRuleDefinition {
        id: "LS008",
        name: "NonPositiveAlpha",
        short_description: "Non-positive or NaN alpha scaling factor",
        full_description: "Layer alpha factor is non-positive or NaN, causing adapter updates to be inverted, zeroed, or corrupted.",
        default_level: "error",
    },
    SarifRuleDefinition {
        id: "LS009",
        name: "ExtremeAlphaUnderscale",
        short_description: "Extreme alpha/r ratio under-scaling adapter updates",
        full_description: "The alpha/r scaling ratio is under 0.1, severely dampening fine-tuned updates during inference or base model merging.",
        default_level: "warning",
    },
    SarifRuleDefinition {
        id: "LS010",
        name: "ExtremeAlphaOverscale",
        short_description: "Extreme alpha/r ratio risking weight norm explosion",
        full_description: "The alpha/r scaling ratio exceeds 16.0, risking logit saturation and weight magnitude divergence when merged.",
        default_level: "warning",
    },
    SarifRuleDefinition {
        id: "LS011",
        name: "MismatchedAlphaScaling",
        short_description: "Wide unpatterned alpha/r spread across adapter layers",
        full_description: "Layers display a wide alpha/r spread exceeding 8x disparity without explicit pattern definitions, suggesting mixed checkpoints.",
        default_level: "warning",
    },
    SarifRuleDefinition {
        id: "LS012",
        name: "MissingDoraVectors",
        short_description: "Declared DoRA adapter is missing magnitude vectors",
        full_description: "adapter_config.json sets use_dora=true but no lora_magnitude_vector tensors are present in safetensors.",
        default_level: "warning",
    },
    SarifRuleDefinition {
        id: "LS013",
        name: "RankCollapsed",
        short_description: "LoRA layer rank has collapsed",
        full_description: "The von Neumann entropy effective rank of the layer is collapsed below 2.0 or 85%+ energy is concentrated in a single singular value.",
        default_level: "error",
    },
    SarifRuleDefinition {
        id: "LS014",
        name: "OverParameterized",
        short_description: "LoRA layer is over-parameterized and compressible",
        full_description: "The layer requires 55% or fewer nominal rank modes to retain 99% spectral energy, indicating excessive rank allocation.",
        default_level: "note",
    },
    SarifRuleDefinition {
        id: "LS015",
        name: "MultiAdapterSubspaceCollision",
        short_description: "High cross-adapter singular subspace collision prior to merging",
        full_description: "Multiple LoRA adapters project onto overlapping singular subspaces, risking mode interference unless merged with orthogonalized SVD.",
        default_level: "warning",
    },
    SarifRuleDefinition {
        id: "LS016",
        name: "MergeNormInflation",
        short_description: "Multi-adapter linear merge inflates Frobenius or spectral norm",
        full_description: "Summing weighted adapter updates inflates layer Frobenius or operator norm well above constituent adapters, risking logit saturation without --preserve-norm.",
        default_level: "warning",
    },
    SarifRuleDefinition {
        id: "LS017",
        name: "DestructiveSignCancellation",
        short_description: "Opposing adapter updates cause destructive sign cancellation",
        full_description: "Shared LoRA layers exhibit negative Frobenius cosine similarity, causing linear merging to cancel task-specific updates.",
        default_level: "error",
    },
    SarifRuleDefinition {
        id: "LS018",
        name: "IncompatibleMergeShape",
        short_description: "Incompatible LoRA layer dimensions across adapters",
        full_description: "Shared module names map to mismatched (d_out, d_in) tensor shapes across adapters, preventing safe weight merging.",
        default_level: "error",
    },
    SarifRuleDefinition {
        id: "LS019",
        name: "DoraMagnitudeCollapseOrMismatch",
        short_description: "DoRA magnitude vector has invalid channels, shape mismatch, or partial coverage",
        full_description: "A DoRA lora_magnitude_vector tensor has length mismatched with lora_B d_out, contains zero/negative/non-finite channel scales, or covers only a subset of adapted linear layers.",
        default_level: "error",
    },
    SarifRuleDefinition {
        id: "LS020",
        name: "DoraMagnitudeDirectionalAnomaly",
        short_description: "DoRA magnitude vector exhibits extreme channel dispersion or directional stagnation",
        full_description: "DoRA output channel scales span an extreme >50x dynamic range (cv > 1.5) or diverge while the low-rank directional update B*A remains frozen across output rows.",
        default_level: "warning",
    },
    SarifRuleDefinition {
        id: "LS021",
        name: "RsLoraScalingAnomaly",
        short_description: "rsLoRA rank-stabilized scaling factor ambiguity or mismatch",
        full_description: "adapter_config.json enables rank-stabilized LoRA (use_rslora=true, alpha/sqrt(r)) alongside embedded .alpha scalar tensors that standalone safetensors loaders divide by r instead of sqrt(r).",
        default_level: "warning",
    },
    SarifRuleDefinition {
        id: "LS022",
        name: "PissaResidualAdapterHazard",
        short_description: "PiSSA residual-adapter conversion, scaling, or principal update hazard",
        full_description: "An unconverted PiSSA adapter (init_lora_weights=\"pissa*\") lacks pissa_init reference weights or unit principal scaling (risking doubled principal singular components on unmodified W_0), has mismatched/partial pissa_init tensors, or shows zero net task update relative to B_init*A_init.",
        default_level: "warning",
    },
    SarifRuleDefinition {
        id: "LS023",
        name: "QuantizationReconstructionHazard",
        short_description: "Low-bit NF4/FP4/INT4/INT8 quantization distortion or corrupt scale metadata",
        full_description: "Blockwise NF4/FP4/INT4/INT8 quantization introduces high spectral or Frobenius reconstruction distortion due to outlier clipping or imbalanced (B, A) factors, or quantized safetensors scale/codebook metadata (absmax, SCB, quant_map, quant_state) is invalid.",
        default_level: "warning",
    },
    SarifRuleDefinition {
        id: "LS024",
        name: "SvdTruncationErrorBoundViolation",
        short_description: "SVD rank truncation violates Eckart-Young spectral/Frobenius bounds or Wedin gap stability",
        full_description: "Truncating the LoRA singular value spectrum at the target or denoised rank exceeds prescribed Eckart-Young spectral/Frobenius/nuclear error bounds or bisects a nearly-degenerate singular doublet with near-zero Wedin gap.",
        default_level: "warning",
    },
    SarifRuleDefinition {
        id: "LS025",
        name: "VeraScalingLayerFailure",
        short_description: "VeRA scaling vector pair has non-finite, all-zero, or dead entries",
        full_description: "A VeRA (Vector-based Random Matrix Adaptation) layer's lambda_b or lambda_d scaling vector contains NaN/Inf entries, or is effectively all-zero across rank or output channels, making the layer's weight update dead during inference.",
        default_level: "error",
    },
    SarifRuleDefinition {
        id: "LS026",
        name: "VeraScalingDispersionAnomaly",
        short_description: "VeRA scaling vectors exhibit extreme output channel or rank dispersion",
        full_description: "VeRA lambda_b or lambda_d vectors contain negative entries that flip adapter update signs, or display extreme coefficient of variation (>2.0) indicating heavily imbalanced scaling across output channels or rank dimensions.",
        default_level: "warning",
    },
    SarifRuleDefinition {
        id: "LS027",
        name: "SubspaceTrainingStagnation",
        short_description: "LoRA singular subspace has not rotated between checkpoints (stagnant training)",
        full_description: "The mean principal angle between right/left singular vector subspaces across two checkpoints is below 5 degrees, indicating gradient flow has stalled and adapter weights are not updating meaningfully.",
        default_level: "warning",
    },
    SarifRuleDefinition {
        id: "LS028",
        name: "CatastrophicSubspaceDrift",
        short_description: "LoRA singular subspace has rotated catastrophically between checkpoints",
        full_description: "The mean principal angle between right/left singular vector subspaces across two checkpoints exceeds 60 degrees, indicating the adapter's learned low-rank subspace has been completely overwritten, risking catastrophic forgetting of earlier fine-tuning signal.",
        default_level: "error",
    },
    SarifRuleDefinition {
        id: "LS029",
        name: "RedundantCrossLayerAlignment",
        short_description: "Adjacent LoRA layers exhibit near-identical input subspaces",
        full_description: "Grassmannian distance between consecutive adapted layers is below pi/12 (mean angle < 15 degrees), indicating redundant subspace capacity across adjacent blocks.",
        default_level: "warning",
    },
    SarifRuleDefinition {
        id: "LS030",
        name: "MaximallyDisjointCrossLayer",
        short_description: "Adjacent LoRA layers are nearly maximally disjoint on the Grassmann manifold",
        full_description: "Grassmannian distance between consecutive adapted layers exceeds pi/2.5, indicating abrupt subspace discontinuity across adjacent blocks.",
        default_level: "note",
    },
    SarifRuleDefinition {
        id: "LS031",
        name: "Fp8ScaleNanOrInf",
        short_description: "FP8 quantization scale factor is NaN or infinite",
        full_description: "An FP8 E4M3 or E5M2 scale tensor contains NaN or Inf entries, corrupting dequantized linear layer activations.",
        default_level: "error",
    },
    SarifRuleDefinition {
        id: "LS032",
        name: "Fp8ScaleNonPositive",
        short_description: "FP8 quantization scale factor is zero or negative",
        full_description: "An FP8 scale tensor contains zero or negative entries, zeroing out or sign-inverting quantized weights.",
        default_level: "error",
    },
    SarifRuleDefinition {
        id: "LS033",
        name: "Fp8ScaleOverflow",
        short_description: "FP8 scale factor causes weight codes to exceed format max finite bound",
        full_description: "Dividing layer weights by the FP8 scale factor exceeds the representable maximum finite code (448 for E4M3, 57344 for E5M2), causing saturation or NaN.",
        default_level: "warning",
    },
    SarifRuleDefinition {
        id: "LS034",
        name: "Fp8ScaleSubnormal",
        short_description: "FP8 scale factor falls in the subnormal denormalized range",
        full_description: "An FP8 scale factor lies below the format minimum normal value, risking mantissa precision loss during dequantization.",
        default_level: "note",
    },
    SarifRuleDefinition {
        id: "LS035",
        name: "TiesSignElectionConflict",
        short_description: "Severe cross-adapter sign conflict during TIES or DARE composition",
        full_description: "Shared active parameter coordinates across adapters exhibit high opposing-sign collision (>=35% overlap conflict or >=25% vetoed Frobenius energy), causing sign-election vetoes or linear cancellation.",
        default_level: "warning",
    },
    SarifRuleDefinition {
        id: "LS036",
        name: "DareDropSignalDistortion",
        short_description: "DARE Bernoulli drop-and-rescale caused severe signal wipeout or variance spike",
        full_description: "Stochastic DARE masking with 1/(1-p) rescaling either wiped out >=80% of concentrated sparse adapter Frobenius energy or inflated layer energy above 2.5x theoretical expectation.",
        default_level: "warning",
    },
    SarifRuleDefinition {
        id: "LS037",
        name: "MaskRankScatteringLoss",
        short_description: "Non-linear TIES/DARE masking scattered Frobenius energy beyond target LoRA rank",
        full_description: "Coordinate-wise magnitude trimming or Bernoulli masking raised the algebraic rank of the composed delta matrix, losing >15% of Frobenius energy when re-projected to target_rank.",
        default_level: "warning",
    },
    SarifRuleDefinition {
        id: "LS038",
        name: "ExcessiveTrimEnergyLoss",
        short_description: "Aggressive TIES magnitude trimming pruned over half of adapter Frobenius energy",
        full_description: "Top-k% TIES magnitude trimming retained <50% of pre-trim Frobenius energy on a dense weight distribution, discarding substantial fine-tuned signal.",
        default_level: "warning",
    },
    SarifRuleDefinition {
        id: "LS039",
        name: "LoraGaInsufficientGradientRank",
        short_description: "Calibration gradient rank is below 2*r required for full LoRA-GA disjoint alignment",
        full_description: "The calibration gradient or warm-up checkpoint has fewer than 2*r non-zero singular modes (or is zero/NaN), placing part of the disjoint A_0 subspace in the gradient nullspace.",
        default_level: "warning",
    },
    SarifRuleDefinition {
        id: "LS040",
        name: "LoraGaUncompensatedBaseDrift",
        short_description: "LoRA-GA non-zero B_0*A_0 initialization lacks base-weight residual compensation",
        full_description: "LoRA-GA initializes both B_0 and A_0 to non-zero singular subspaces; loading onto an unmodified W_0 without --emit-init or --out-base shifts t=0 forward logits.",
        default_level: "warning",
    },
    SarifRuleDefinition {
        id: "LS041",
        name: "OrthogonalProjectionSignalAnnihilation",
        short_description: "Orthogonal projection against anchor checkpoints removed >=70% of candidate Frobenius energy",
        full_description: "Projecting the candidate LoRA onto the orthogonal complement of protected anchor subspaces retained <30% of Frobenius energy, indicating severe feature overlap with protected tasks.",
        default_level: "warning",
    },
    SarifRuleDefinition {
        id: "LS042",
        name: "AnchorSubspaceSaturation",
        short_description: "Protected anchor subspaces saturate >=75% of ambient layer dimensions",
        full_description: "The union of protected anchor subspaces consumes >=75% of the ambient dimension or leaves fewer orthogonal dimensions than the candidate LoRA rank.",
        default_level: "warning",
    },
    SarifRuleDefinition {
        id: "LS043",
        name: "KnotsAlignedDestructiveCancellation",
        short_description: "Destructive task-vector cancellation in shared Knots singular frame",
        full_description: "Adapters projected into the shared Knots left-singular coordinate frame cancel >50% of combined Frobenius energy or exhibit pairwise Frobenius cosine below -0.25.",
        default_level: "warning",
    },
    SarifRuleDefinition {
        id: "LS044",
        name: "JointRankTruncationOverflow",
        short_description: "Joint multi-adapter subspace rank truncation discarded >20% of Frobenius energy",
        full_description: "Compressing the synthesized joint subspace down to target_rank retained <80% of post-alignment Frobenius energy; increase --rank to preserve multi-task capacity.",
        default_level: "warning",
    },
    SarifRuleDefinition {
        id: "LS045",
        name: "AdapterTaskNormDominance",
        short_description: "Single adapter dominates >=80% of pre-merge Frobenius energy on shared layer",
        full_description: "One participating adapter contributes >=80% of total pre-merge Frobenius energy on a shared layer, risking collapse of secondary task vectors without --normalize-tasks.",
        default_level: "warning",
    },
    SarifRuleDefinition {
        id: "LS046",
        name: "SingularSpectrumAnisotropySpike",
        short_description: "Merged singular spectrum exhibits severe rank-1 anisotropy spike (stable rank < 1.35)",
        full_description: "The combined task delta on a nominal rank >= 4 layer concentrates nearly all energy into its top singular value (stable rank < 1.35); consider --mode iso spectral flattening.",
        default_level: "warning",
    },
];

fn escape_json(s: &str) -> String {
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

fn map_issue_code_to_rule(code: &str) -> Option<(&'static str, &'static str)> {
    match code {
        "DEAD_LORA_LAYER" => Some(("LS001", "error")),
        "ZERO_NORM_EMBEDDING_ROWS" => Some(("LS002", "error")),
        "DEAD_LORA_B_COLS" | "DEAD_EMBEDDING_LORA_COLUMNS" => Some(("LS003", "warning")),
        "DEAD_LORA_A_ROWS" => Some(("LS004", "warning")),
        "MISSING_CONFIG_ALPHA" | "MISSING_LORA_ALPHA" => Some(("LS005", "warning")),
        "MISSING_ALPHA_TENSORS" | "MISMATCHED_ALPHA_TENSORS" => Some(("LS006", "warning")),
        "MISMATCHED_ALPHA_CONFIG" => Some(("LS007", "warning")),
        "NON_POSITIVE_ALPHA" => Some(("LS008", "error")),
        "EXTREME_ALPHA_UNDERSCALE" => Some(("LS009", "warning")),
        "EXTREME_ALPHA_OVERSCALE" => Some(("LS010", "warning")),
        "MISMATCHED_ALPHA_SCALING" => Some(("LS011", "warning")),
        "MISSING_DORA_VECTORS" => Some(("LS012", "warning")),
        "RANK_COLLAPSED" => Some(("LS013", "error")),
        "OVER_PARAMETERIZED" => Some(("LS014", "note")),
        "DORA_SHAPE_MISMATCH" | "INVALID_DORA_MAGNITUDE" => Some(("LS019", "error")),
        "PARTIAL_DORA_COVERAGE" => Some(("LS019", "warning")),
        "DORA_EXTREME_DISPERSION" | "DORA_DIRECTIONAL_STAGNATION" => Some(("LS020", "warning")),
        "RSLORA_ALPHA_TENSOR_AMBIGUITY" => Some(("LS021", "warning")),
        "PISSA_SHAPE_MISMATCH" => Some(("LS022", "error")),
        "PISSA_UNCONVERTED_RESIDUAL_HAZARD"
        | "PISSA_NON_UNIT_SCALING"
        | "PISSA_PARTIAL_INIT_COVERAGE"
        | "PISSA_ZERO_TASK_UPDATE" => Some(("LS022", "warning")),
        "QUANT_CORRUPT_METADATA" | "QUANT_ZERO_OR_NAN_ABSMAX" => Some(("LS023", "error")),
        "QUANT_RECONSTRUCTION_HIGH_ERROR" | "QUANT_IMBALANCED_FACTOR_LOSS" => {
            Some(("LS023", "warning"))
        }
        "SVD_TRUNCATION_BOUND_EXCEEDED" | "SVD_DEGENERATE_SPECTRAL_GAP" => {
            Some(("LS024", "warning"))
        }
        _ => None,
    }
}

pub fn build_audit_sarif(
    audit: Option<&AdapterAuditSummary>,
    integrity: Option<&AdapterIntegrityReport>,
    source_uri: &str,
    tool_version: &str,
) -> String {
    let mut results_json = Vec::new();

    // 1. Process spectral health findings from AdapterAuditSummary
    if let Some(summary) = audit {
        for l in &summary.layers {
            let (rule_id, level) = match l.health {
                LayerHealth::DeadLayer => ("LS001", "error"),
                LayerHealth::RankCollapsed => ("LS013", "error"),
                LayerHealth::OverParameterized => ("LS014", "note"),
                LayerHealth::Optimal => continue,
            };
            let msg = match l.health {
                LayerHealth::DeadLayer => format!(
                    "Layer '{}' has zero/near-zero Frobenius norm (spectral_norm={:.4e})",
                    l.base_name, l.spectral_norm
                ),
                LayerHealth::RankCollapsed => format!(
                    "Layer '{}' effective rank collapsed to {:.2} (nominal rank {}, 99% energy rank {})",
                    l.base_name, l.effective_rank, l.nominal_rank, l.rank_at_99_energy
                ),
                LayerHealth::OverParameterized => format!(
                    "Layer '{}' is over-parameterized (utilization {:.1}%, 99% energy retained at rank {} of {})",
                    l.base_name, l.utilization_ratio * 100.0, l.rank_at_99_energy, l.nominal_rank
                ),
                LayerHealth::Optimal => unreachable!(),
            };

            results_json.push(format!(
                r#"{{
        "ruleId": "{}",
        "level": "{}",
        "message": {{ "text": "{}" }},
        "locations": [
          {{
            "physicalLocation": {{
              "artifactLocation": {{ "uri": "{}" }}
            }},
            "logicalLocations": [
              {{
                "name": "{}",
                "kind": "module"
              }}
            ]
          }}
        ],
        "properties": {{
          "nominalRank": {},
          "effectiveRank": {:.4},
          "rankAt99Energy": {},
          "spectralNorm": {:.6}
        }}
      }}"#,
                rule_id,
                level,
                escape_json(&msg),
                escape_json(source_uri),
                escape_json(&l.base_name),
                l.nominal_rank,
                l.effective_rank,
                l.rank_at_99_energy,
                l.spectral_norm
            ));
        }
    }

    // 2. Process integrity issues from AdapterIntegrityReport
    if let Some(integ) = integrity {
        for issue in &integ.issues {
            let (rule_id, default_level) = map_issue_code_to_rule(issue.code)
                .unwrap_or(("LS001", "warning"));
            let level = match issue.severity {
                IssueSeverity::Critical => "error",
                IssueSeverity::Warning => "warning",
                IssueSeverity::Info => default_level,
            };

            results_json.push(format!(
                r#"{{
        "ruleId": "{}",
        "level": "{}",
        "message": {{ "text": "{}" }},
        "locations": [
          {{
            "physicalLocation": {{
              "artifactLocation": {{ "uri": "{}" }}
            }},
            "logicalLocations": [
              {{
                "name": "{}",
                "kind": "module"
              }}
            ]
          }}
        ],
        "properties": {{
          "issueCode": "{}",
          "severity": "{}"
        }}
      }}"#,
                rule_id,
                level,
                escape_json(&issue.message),
                escape_json(source_uri),
                escape_json(&issue.subject),
                escape_json(issue.code),
                issue.severity.as_str()
            ));
        }
    }

    let rules_rendered: Vec<String> = LORASPECTRA_SARIF_RULES
        .iter()
        .map(|r| {
            format!(
                r#"{{
            "id": "{}",
            "name": "{}",
            "shortDescription": {{ "text": "{}" }},
            "fullDescription": {{ "text": "{}" }},
            "defaultConfiguration": {{ "level": "{}" }},
            "helpUri": "https://github.com/SalyyS1/loraspectra#sarif-diagnostics"
          }}"#,
                r.id,
                r.name,
                escape_json(r.short_description),
                escape_json(r.full_description),
                r.default_level
            )
        })
        .collect();

    format!(
        r#"{{
  "$schema": "https://json.schemastore.org/sarif-2.1.0.json",
  "version": "2.1.0",
  "runs": [
    {{
      "tool": {{
        "driver": {{
          "name": "loraspectra",
          "version": "{}",
          "informationUri": "https://github.com/SalyyS1/loraspectra",
          "rules": [
            {}
          ]
        }}
      }},
      "results": [
        {}
      ]
    }}
  ]
}}
"#,
        escape_json(tool_version),
        rules_rendered.join(",\n            "),
        results_json.join(",\n        ")
    )
}

pub fn build_merge_check_sarif(
    summary: &MultiMergeCheckSummary,
    tool_version: &str,
) -> String {
    let mut results_json = Vec::new();
    let primary_uri = summary
        .adapter_names
        .first()
        .map(|s| s.as_str())
        .unwrap_or("merged_adapter.safetensors");

    for inc in &summary.incompatible_shapes {
        let msg = format!(
            "Layer '{}' has incompatible shapes: {} ({}x{}) vs {} ({}x{})",
            inc.base_name,
            inc.adapter_a,
            inc.shape_a.0,
            inc.shape_a.1,
            inc.adapter_b,
            inc.shape_b.0,
            inc.shape_b.1
        );
        results_json.push(format!(
            r#"{{
        "ruleId": "LS018",
        "level": "error",
        "message": {{ "text": "{}" }},
        "locations": [
          {{
            "physicalLocation": {{
              "artifactLocation": {{ "uri": "{}" }}
            }},
            "logicalLocations": [
              {{
                "name": "{}",
                "kind": "module"
              }}
            ]
          }}
        ]
      }}"#,
            escape_json(&msg),
            escape_json(&inc.adapter_b),
            escape_json(&inc.base_name)
        ));
    }

    for l in &summary.shared_layers {
        let (rule_id, level, msg) = match l.status {
            "DESTRUCTIVE_CANCELLATION" => (
                "LS017",
                "error",
                format!(
                    "Layer '{}' has destructive cross-adapter sign cancellation (min Frobenius cosine {:+.4}, orthogonal norm ratio {:.2}x)",
                    l.base_name, l.min_pair_frobenius_cosine, l.orthogonal_inflation_ratio
                ),
            ),
            "NORM_INFLATION" => (
                "LS016",
                "warning",
                format!(
                    "Layer '{}' inflates peak Frobenius norm by {:.2}x and spectral norm by {:.2}x across {} adapters",
                    l.base_name,
                    l.peak_inflation_ratio,
                    l.spectral_inflation_ratio,
                    l.adapter_indices.len()
                ),
            ),
            "HIGH_SUBSPACE_COLLISION" => (
                "LS015",
                "warning",
                format!(
                    "Layer '{}' exhibits {:.1}% multi-adapter singular subspace collision (union 99% rank {})",
                    l.base_name,
                    l.subspace_collision_ratio * 100.0,
                    l.union_rank_99
                ),
            ),
            _ => continue,
        };

        let worst_uri = summary
            .adapter_names
            .get(l.worst_pair_indices.1)
            .map(|s| s.as_str())
            .unwrap_or(primary_uri);

        results_json.push(format!(
            r#"{{
        "ruleId": "{}",
        "level": "{}",
        "message": {{ "text": "{}" }},
        "locations": [
          {{
            "physicalLocation": {{
              "artifactLocation": {{ "uri": "{}" }}
            }},
            "logicalLocations": [
              {{
                "name": "{}",
                "kind": "module"
              }}
            ]
          }}
        ],
        "properties": {{
          "adapterCount": {},
          "subspaceCollisionRatio": {:.6},
          "peakInflationRatio": {:.6},
          "spectralInflationRatio": {:.6},
          "minPairFrobeniusCosine": {:.6},
          "unionRank99": {}
        }}
      }}"#,
            rule_id,
            level,
            escape_json(&msg),
            escape_json(worst_uri),
            escape_json(&l.base_name),
            l.adapter_indices.len(),
            l.subspace_collision_ratio,
            l.peak_inflation_ratio,
            l.spectral_inflation_ratio,
            l.min_pair_frobenius_cosine,
            l.union_rank_99
        ));
    }

    let rules_rendered: Vec<String> = LORASPECTRA_SARIF_RULES
        .iter()
        .map(|r| {
            format!(
                r#"{{
            "id": "{}",
            "name": "{}",
            "shortDescription": {{ "text": "{}" }},
            "fullDescription": {{ "text": "{}" }},
            "defaultConfiguration": {{ "level": "{}" }},
            "helpUri": "https://github.com/SalyyS1/loraspectra#sarif-diagnostics"
          }}"#,
                r.id,
                r.name,
                escape_json(r.short_description),
                escape_json(r.full_description),
                r.default_level
            )
        })
        .collect();

    format!(
        r#"{{
  "$schema": "https://json.schemastore.org/sarif-2.1.0.json",
  "version": "2.1.0",
  "runs": [
    {{
      "tool": {{
        "driver": {{
          "name": "loraspectra",
          "version": "{}",
          "informationUri": "https://github.com/SalyyS1/loraspectra",
          "rules": [
            {}
          ]
        }}
      }},
      "results": [
        {}
      ]
    }}
  ]
}}
"#,
        escape_json(tool_version),
        rules_rendered.join(",\n            "),
        results_json.join(",\n        ")
    )
}

pub fn build_ties_dare_sarif(
    summary: &TiesDareComposeSummary,
    tool_version: &str,
) -> String {
    let mut results_json = Vec::new();

    for inc in &summary.incompatible_shapes {
        let msg = format!(
            "Layer '{}' has incompatible shapes: {} ({}x{}) vs {} ({}x{})",
            inc.base_name,
            inc.adapter_a,
            inc.shape_a.0,
            inc.shape_a.1,
            inc.adapter_b,
            inc.shape_b.0,
            inc.shape_b.1
        );
        results_json.push(format!(
            r#"{{
        "ruleId": "LS018",
        "level": "error",
        "message": {{ "text": "{}" }},
        "locations": [
          {{
            "physicalLocation": {{
              "artifactLocation": {{ "uri": "{}" }}
            }},
            "logicalLocations": [
              {{
                "name": "{}",
                "kind": "module"
              }}
            ]
          }}
        ]
      }}"#,
            escape_json(&msg),
            escape_json(&inc.adapter_b),
            escape_json(&inc.base_name)
        ));
    }

    for diag in &summary.diagnostics {
        results_json.push(format!(
            r#"{{
        "ruleId": "{}",
        "level": "{}",
        "message": {{ "text": "{}" }},
        "locations": [
          {{
            "physicalLocation": {{
              "artifactLocation": {{ "uri": "{}" }}
            }},
            "logicalLocations": [
              {{
                "name": "{}",
                "kind": "module"
              }}
            ]
          }}
        ],
        "properties": {{
          "mode": "{}",
          "signConflictRate": {:.6},
          "discardedConflictRatio": {:.6},
          "svdRetainedRatio": {:.6},
          "trimRetainedRatio": {:.6},
          "dareRescaleRatio": {:.6}
        }}
      }}"#,
            diag.rule_id,
            diag.severity.as_str(),
            escape_json(&diag.message),
            escape_json(&diag.artifact_uri),
            escape_json(&diag.layer_name),
            summary.mode.as_str(),
            diag.sign_conflict_rate,
            diag.discarded_conflict_ratio,
            diag.svd_retained_ratio,
            diag.trim_retained_ratio,
            diag.dare_rescale_ratio
        ));
    }

    let rules_rendered: Vec<String> = LORASPECTRA_SARIF_RULES
        .iter()
        .map(|r| {
            format!(
                r#"{{
            "id": "{}",
            "name": "{}",
            "shortDescription": {{ "text": "{}" }},
            "fullDescription": {{ "text": "{}" }},
            "defaultConfiguration": {{ "level": "{}" }},
            "helpUri": "https://github.com/SalyyS1/loraspectra#sarif-diagnostics"
          }}"#,
                r.id,
                r.name,
                escape_json(r.short_description),
                escape_json(r.full_description),
                r.default_level
            )
        })
        .collect();

    format!(
        r#"{{
  "$schema": "https://json.schemastore.org/sarif-2.1.0.json",
  "version": "2.1.0",
  "runs": [
    {{
      "tool": {{
        "driver": {{
          "name": "loraspectra",
          "version": "{}",
          "informationUri": "https://github.com/SalyyS1/loraspectra",
          "rules": [
            {}
          ]
        }}
      }},
      "results": [
        {}
      ]
    }}
  ]
}}
"#,
        escape_json(tool_version),
        rules_rendered.join(",\n            "),
        results_json.join(",\n        ")
    )
}

fn render_sarif_envelope(tool_version: &str, results_json: &[String]) -> String {
    let rules_rendered: Vec<String> = LORASPECTRA_SARIF_RULES
        .iter()
        .map(|r| {
            format!(
                r#"{{
            "id": "{}",
            "name": "{}",
            "shortDescription": {{ "text": "{}" }},
            "fullDescription": {{ "text": "{}" }},
            "defaultConfiguration": {{ "level": "{}" }},
            "helpUri": "https://github.com/SalyyS1/loraspectra#sarif-diagnostics"
          }}"#,
                r.id,
                r.name,
                escape_json(r.short_description),
                escape_json(r.full_description),
                r.default_level
            )
        })
        .collect();

    format!(
        r#"{{
  "$schema": "https://json.schemastore.org/sarif-2.1.0.json",
  "version": "2.1.0",
  "runs": [
    {{
      "tool": {{
        "driver": {{
          "name": "loraspectra",
          "version": "{}",
          "informationUri": "https://github.com/SalyyS1/loraspectra",
          "rules": [
            {}
          ]
        }}
      }},
      "results": [
        {}
      ]
    }}
  ]
}}
"#,
        escape_json(tool_version),
        rules_rendered.join(",\n            "),
        results_json.join(",\n        ")
    )
}

pub fn build_lora_ga_sarif(summary: &LoraGaSummary, tool_version: &str) -> String {
    let mut results_json = Vec::new();
    for diag in &summary.diagnostics {
        results_json.push(format!(
            r#"{{
        "ruleId": "{}",
        "level": "{}",
        "message": {{ "text": "{}" }},
        "locations": [
          {{
            "physicalLocation": {{
              "artifactLocation": {{ "uri": "{}" }}
            }},
            "logicalLocations": [
              {{
                "name": "{}",
                "kind": "module"
              }}
            ]
          }}
        ],
        "properties": {{
          "scaleMode": "{}",
          "nonzeroGradModes": {},
          "required2rModes": {},
          "initialPerturbationNorm": {:.8}
        }}
      }}"#,
            diag.rule_id,
            diag.severity.as_str(),
            escape_json(&diag.message),
            escape_json(&diag.artifact_uri),
            escape_json(&diag.layer_name),
            summary.scale_mode.as_str(),
            diag.nonzero_grad_modes,
            diag.required_2r_modes,
            diag.initial_perturbation_norm
        ));
    }
    render_sarif_envelope(tool_version, &results_json)
}

pub fn build_ortho_proj_sarif(summary: &OrthoProjectSummary, tool_version: &str) -> String {
    let mut results_json = Vec::new();

    for inc in &summary.incompatible_shapes {
        let msg = format!(
            "Layer '{}' has incompatible shapes: {} ({}x{}) vs {} ({}x{})",
            inc.base_name,
            inc.adapter_a,
            inc.shape_a.0,
            inc.shape_a.1,
            inc.adapter_b,
            inc.shape_b.0,
            inc.shape_b.1
        );
        results_json.push(format!(
            r#"{{
        "ruleId": "LS018",
        "level": "error",
        "message": {{ "text": "{}" }},
        "locations": [
          {{
            "physicalLocation": {{
              "artifactLocation": {{ "uri": "{}" }}
            }},
            "logicalLocations": [
              {{
                "name": "{}",
                "kind": "module"
              }}
            ]
          }}
        ]
      }}"#,
            escape_json(&msg),
            escape_json(&inc.adapter_a),
            escape_json(&inc.base_name)
        ));
    }

    for diag in &summary.diagnostics {
        results_json.push(format!(
            r#"{{
        "ruleId": "{}",
        "level": "{}",
        "message": {{ "text": "{}" }},
        "locations": [
          {{
            "physicalLocation": {{
              "artifactLocation": {{ "uri": "{}" }}
            }},
            "logicalLocations": [
              {{
                "name": "{}",
                "kind": "module"
              }}
            ]
          }}
        ],
        "properties": {{
          "domain": "{}",
          "retainedRatio": {:.6},
          "ambientSaturationRatio": {:.6}
        }}
      }}"#,
            diag.rule_id,
            diag.severity.as_str(),
            escape_json(&diag.message),
            escape_json(&diag.artifact_uri),
            escape_json(&diag.layer_name),
            summary.domain.as_str(),
            diag.retained_ratio,
            diag.ambient_saturation_ratio
        ));
    }

    render_sarif_envelope(tool_version, &results_json)
}

pub fn build_knots_sarif(summary: &KnotsSummary, tool_version: &str) -> String {
    let mut results_json = Vec::new();

    for inc in &summary.incompatible_shapes {
        let msg = format!(
            "Layer '{}' has incompatible shapes: {} ({}x{}) vs {} ({}x{})",
            inc.base_name,
            inc.adapter_a,
            inc.shape_a.0,
            inc.shape_a.1,
            inc.adapter_b,
            inc.shape_b.0,
            inc.shape_b.1
        );
        results_json.push(format!(
            r#"{{
        "ruleId": "LS018",
        "level": "error",
        "message": {{ "text": "{}" }},
        "locations": [
          {{
            "physicalLocation": {{
              "artifactLocation": {{ "uri": "{}" }}
            }},
            "logicalLocations": [
              {{
                "name": "{}",
                "kind": "module"
              }}
            ]
          }}
        ]
      }}"#,
            escape_json(&msg),
            escape_json(&inc.adapter_b),
            escape_json(&inc.base_name)
        ));
    }

    for diag in &summary.diagnostics {
        results_json.push(format!(
            r#"{{
        "ruleId": "{}",
        "level": "{}",
        "message": {{ "text": "{}" }},
        "locations": [
          {{
            "physicalLocation": {{
              "artifactLocation": {{ "uri": "{}" }}
            }},
            "logicalLocations": [
              {{
                "name": "{}",
                "kind": "module"
              }}
            ]
          }}
        ],
        "properties": {{
          "mode": "{}",
          "coherenceRatio": {:.6},
          "svdRetainedRatio": {:.6},
          "maxTaskEnergyShare": {:.6},
          "preStableRank": {:.6}
        }}
      }}"#,
            diag.rule_id,
            diag.severity.as_str(),
            escape_json(&diag.message),
            escape_json(&diag.artifact_uri),
            escape_json(&diag.layer_name),
            summary.mode.as_str(),
            diag.coherence_ratio,
            diag.svd_retained_ratio,
            diag.max_task_energy_share,
            diag.pre_stable_rank
        ));
    }

    render_sarif_envelope(tool_version, &results_json)
}

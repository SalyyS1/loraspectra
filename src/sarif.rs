use crate::audit::{AdapterAuditSummary, AdapterIntegrityReport, IssueSeverity, LayerHealth};

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
        "DEAD_LORA_B_COLS" => Some(("LS003", "warning")),
        "DEAD_LORA_A_ROWS" => Some(("LS004", "warning")),
        "MISSING_CONFIG_ALPHA" => Some(("LS005", "warning")),
        "MISSING_ALPHA_TENSORS" => Some(("LS006", "warning")),
        "MISMATCHED_ALPHA_CONFIG" => Some(("LS007", "warning")),
        "NON_POSITIVE_ALPHA" => Some(("LS008", "error")),
        "EXTREME_ALPHA_UNDERSCALE" => Some(("LS009", "warning")),
        "EXTREME_ALPHA_OVERSCALE" => Some(("LS010", "warning")),
        "MISMATCHED_ALPHA_SCALING" => Some(("LS011", "warning")),
        "MISSING_DORA_VECTORS" => Some(("LS012", "warning")),
        "RANK_COLLAPSED" => Some(("LS013", "error")),
        "OVER_PARAMETERIZED" => Some(("LS014", "note")),
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

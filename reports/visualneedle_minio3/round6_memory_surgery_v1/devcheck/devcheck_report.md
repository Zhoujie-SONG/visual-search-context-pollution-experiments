# Round-6 Visual Memory Surgery Devcheck Report

- Episodes: 20
- Unique samples: 20
- Judge pending: 0
- Pending judge rows are excluded from final-accuracy denominators, never counted as wrong.

## Primary Prefix-Hit

| Arm | N | Answer rate | Final accuracy | Accuracy among answered | Natural stop | Mean additional crops |
|---|---:|---:|---:|---:|---:|---:|
| A_full | 13 | 0.00% | 0.00% | NA | 0.00% | 5.00 |

## Strong-Evidence Sensitivity

GT coverage >=10% is a geometric-hit criterion, not a guarantee of sufficient answer evidence.

| Subgroup | Arm | N | Answer rate | Final accuracy | Accuracy among answered | Natural stop | Mean additional crops |
|---|---|---:|---:|---:|---:|---:|---:|
| primary_prefix_hit_all | A_full | 13 | 0.00% | 0.00% | NA | 0.00% | 5.00 |
| primary_prefix_hit_all | B_oracle_top2 | 0 | NA | NA | NA | NA | 0.00 |
| primary_prefix_hit_all | C_recent_top2 | 0 | NA | NA | NA | NA | 0.00 |
| primary_prefix_hit_all | D_force_answer_r6 | 0 | NA | NA | NA | NA | 0.00 |
| primary_prefix_hit_all | E_random_top2 | 0 | NA | NA | NA | NA | 0.00 |
| max_prefix_gt_coverage_ge_0.50 | A_full | 11 | 0.00% | 0.00% | NA | 0.00% | 5.00 |
| max_prefix_gt_coverage_ge_0.50 | B_oracle_top2 | 0 | NA | NA | NA | NA | 0.00 |
| max_prefix_gt_coverage_ge_0.50 | C_recent_top2 | 0 | NA | NA | NA | NA | 0.00 |
| max_prefix_gt_coverage_ge_0.50 | D_force_answer_r6 | 0 | NA | NA | NA | NA | 0.00 |
| max_prefix_gt_coverage_ge_0.50 | E_random_top2 | 0 | NA | NA | NA | NA | 0.00 |
| max_prefix_gt_coverage_ge_0.999999 | A_full | 10 | 0.00% | 0.00% | NA | 0.00% | 5.00 |
| max_prefix_gt_coverage_ge_0.999999 | B_oracle_top2 | 0 | NA | NA | NA | NA | 0.00 |
| max_prefix_gt_coverage_ge_0.999999 | C_recent_top2 | 0 | NA | NA | NA | NA | 0.00 |
| max_prefix_gt_coverage_ge_0.999999 | D_force_answer_r6 | 0 | NA | NA | NA | NA | 0.00 |
| max_prefix_gt_coverage_ge_0.999999 | E_random_top2 | 0 | NA | NA | NA | NA | 0.00 |

## Oracle Selection Audit

- Highly redundant oracle pairs: 0/0 (0.00%)
- Both retained crops have approximately 100% coverage: 0/0 (0.00%)
- Both 100% coverage and high overlap: 0/0 (0.00%)

## Intervention Boundary

The intervention removes historical visual observation images but intentionally preserves the assistant's textual reasoning trace.
B > A supports a causal effect of visual observation memory on continuation behavior.
B approximately A does not by itself show that all context pollution is absent, because prior visual information may persist in assistant reasoning text.

## Interpretation

This stage checks reconstruction only; no causal interpretation is made.
This descriptive label is based on raw point-estimate ordering only. It is not a statistical decision rule.
Statistical interpretation must use paired bootstrap confidence intervals and McNemar tests.

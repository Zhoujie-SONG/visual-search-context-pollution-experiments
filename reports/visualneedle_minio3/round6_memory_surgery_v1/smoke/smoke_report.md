# Round-6 Visual Memory Surgery Smoke Report

- Episodes: 50
- Unique samples: 10
- Judge pending: 0
- Pending judge rows are excluded from final-accuracy denominators, never counted as wrong.

## Primary Prefix-Hit

| Arm | N | Answer rate | Final accuracy | Accuracy among answered | Natural stop | Mean additional crops |
|---|---:|---:|---:|---:|---:|---:|
| A_full | 10 | 0.00% | 0.00% | NA | 0.00% | 5.00 |
| B_oracle_top2 | 10 | 0.00% | 0.00% | NA | 0.00% | 3.30 |
| C_recent_top2 | 10 | 0.00% | 0.00% | NA | 0.00% | 3.80 |
| D_force_answer_r6 | 10 | 0.00% | 0.00% | NA | 0.00% | 0.00 |
| E_random_top2 | 10 | 0.00% | 0.00% | NA | 0.00% | 3.10 |

## Strong-Evidence Sensitivity

GT coverage >=10% is a geometric-hit criterion, not a guarantee of sufficient answer evidence.

| Subgroup | Arm | N | Answer rate | Final accuracy | Accuracy among answered | Natural stop | Mean additional crops |
|---|---|---:|---:|---:|---:|---:|---:|
| primary_prefix_hit_all | A_full | 10 | 0.00% | 0.00% | NA | 0.00% | 5.00 |
| primary_prefix_hit_all | B_oracle_top2 | 10 | 0.00% | 0.00% | NA | 0.00% | 3.30 |
| primary_prefix_hit_all | C_recent_top2 | 10 | 0.00% | 0.00% | NA | 0.00% | 3.80 |
| primary_prefix_hit_all | D_force_answer_r6 | 10 | 0.00% | 0.00% | NA | 0.00% | 0.00 |
| primary_prefix_hit_all | E_random_top2 | 10 | 0.00% | 0.00% | NA | 0.00% | 3.10 |
| max_prefix_gt_coverage_ge_0.50 | A_full | 10 | 0.00% | 0.00% | NA | 0.00% | 5.00 |
| max_prefix_gt_coverage_ge_0.50 | B_oracle_top2 | 10 | 0.00% | 0.00% | NA | 0.00% | 3.30 |
| max_prefix_gt_coverage_ge_0.50 | C_recent_top2 | 10 | 0.00% | 0.00% | NA | 0.00% | 3.80 |
| max_prefix_gt_coverage_ge_0.50 | D_force_answer_r6 | 10 | 0.00% | 0.00% | NA | 0.00% | 0.00 |
| max_prefix_gt_coverage_ge_0.50 | E_random_top2 | 10 | 0.00% | 0.00% | NA | 0.00% | 3.10 |
| max_prefix_gt_coverage_ge_0.999999 | A_full | 9 | 0.00% | 0.00% | NA | 0.00% | 5.00 |
| max_prefix_gt_coverage_ge_0.999999 | B_oracle_top2 | 9 | 0.00% | 0.00% | NA | 0.00% | 3.44 |
| max_prefix_gt_coverage_ge_0.999999 | C_recent_top2 | 9 | 0.00% | 0.00% | NA | 0.00% | 3.67 |
| max_prefix_gt_coverage_ge_0.999999 | D_force_answer_r6 | 9 | 0.00% | 0.00% | NA | 0.00% | 0.00 |
| max_prefix_gt_coverage_ge_0.999999 | E_random_top2 | 9 | 0.00% | 0.00% | NA | 0.00% | 2.89 |

## Oracle Selection Audit

- Highly redundant oracle pairs: 0/10 (0.00%)
- Both retained crops have approximately 100% coverage: 5/10 (50.00%)
- Both 100% coverage and high overlap: 0/10 (0.00%)

## Intervention Boundary

The intervention removes historical visual observation images but intentionally preserves the assistant's textual reasoning trace.
B > A supports a causal effect of visual observation memory on continuation behavior.
B approximately A does not by itself show that all context pollution is absent, because prior visual information may persist in assistant reasoning text.

## Interpretation

DESCRIPTIVE_MIXED_OR_INCONCLUSIVE: observed ordering does not match a preregistered case cleanly.
Approximate-equality tolerance used by the descriptive decision rule: 0.03 absolute accuracy.
This descriptive label is based on raw point-estimate ordering only. It is not a statistical decision rule.
Statistical interpretation must use paired bootstrap confidence intervals and McNemar tests.

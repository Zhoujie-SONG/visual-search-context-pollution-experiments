# Round-6 Direct-Answer Memory Diagnostic

- Primary N: 88
- Generations: 352

## Main Results

| Arm | Strict answer | Lenient answer | Strict final | Lenient final | Strict acc/answered | Lenient acc/answered | Grounding violation |
|---|---:|---:|---:|---:|---:|---:|---:|
| F_full_direct | 3.41% | 3.41% | 0.00% | 0.00% | NA | NA | 40.91% |
| O_oracle_top2_direct | 4.55% | 4.55% | 0.00% | 0.00% | NA | NA | 39.77% |
| R_recent_top2_direct | 3.41% | 3.41% | 1.16% | 1.16% | 100.00% | 100.00% | 42.05% |
| X_random_top2_direct | 1.14% | 1.14% | 0.00% | 0.00% | NA | NA | 38.64% |

## Paired Comparisons

- O_oracle_top2_direct - F_full_direct, strict: delta=0.00%, 95% CI [0.00%, 0.00%], McNemar p=1, accuracy rescue=0.
- O_oracle_top2_direct - F_full_direct, lenient: delta=0.00%, 95% CI [0.00%, 0.00%], McNemar p=1, accuracy rescue=0.
- O_oracle_top2_direct - R_recent_top2_direct, strict: delta=-1.20%, 95% CI [-3.61%, 0.00%], McNemar p=1, accuracy rescue=0.
- O_oracle_top2_direct - R_recent_top2_direct, lenient: delta=-1.20%, 95% CI [-3.61%, 0.00%], McNemar p=1, accuracy rescue=0.
- O_oracle_top2_direct - X_random_top2_direct, strict: delta=0.00%, 95% CI [0.00%, 0.00%], McNemar p=1, accuracy rescue=0.
- O_oracle_top2_direct - X_random_top2_direct, lenient: delta=0.00%, 95% CI [0.00%, 0.00%], McNemar p=1, accuracy rescue=0.

## Judge

- Required or completed: 10
- Pending: 10
- Model: `gemini-3.8-flash`

## Interpretation

Final interpretation is valid only when judge pending is zero. Use strict results as primary and lenient results as a diagnostic.

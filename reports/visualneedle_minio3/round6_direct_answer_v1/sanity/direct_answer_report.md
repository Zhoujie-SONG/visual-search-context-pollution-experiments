# Round-6 Direct-Answer Memory Diagnostic

- Primary N: 2
- Generations: 8

## Main Results

| Arm | Strict answer | Lenient answer | Strict final | Lenient final | Strict acc/answered | Lenient acc/answered | Grounding violation |
|---|---:|---:|---:|---:|---:|---:|---:|
| F_full_direct | 0.00% | 0.00% | 0.00% | 0.00% | NA | NA | 100.00% |
| O_oracle_top2_direct | 0.00% | 0.00% | 0.00% | 0.00% | NA | NA | 100.00% |
| R_recent_top2_direct | 0.00% | 0.00% | 0.00% | 0.00% | NA | NA | 100.00% |
| X_random_top2_direct | 0.00% | 0.00% | 0.00% | 0.00% | NA | NA | 100.00% |

## Paired Comparisons

- O_oracle_top2_direct - F_full_direct, strict: delta=0.00%, 95% CI [0.00%, 0.00%], McNemar p=1, accuracy rescue=0.
- O_oracle_top2_direct - F_full_direct, lenient: delta=0.00%, 95% CI [0.00%, 0.00%], McNemar p=1, accuracy rescue=0.
- O_oracle_top2_direct - R_recent_top2_direct, strict: delta=0.00%, 95% CI [0.00%, 0.00%], McNemar p=1, accuracy rescue=0.
- O_oracle_top2_direct - R_recent_top2_direct, lenient: delta=0.00%, 95% CI [0.00%, 0.00%], McNemar p=1, accuracy rescue=0.
- O_oracle_top2_direct - X_random_top2_direct, strict: delta=0.00%, 95% CI [0.00%, 0.00%], McNemar p=1, accuracy rescue=0.
- O_oracle_top2_direct - X_random_top2_direct, lenient: delta=0.00%, 95% CI [0.00%, 0.00%], McNemar p=1, accuracy rescue=0.

## Judge

- Required or completed: 0
- Pending: 0
- Model: `gemini-3.8-flash`

## Interpretation

Final interpretation is valid only when judge pending is zero. Use strict results as primary and lenient results as a diagnostic.

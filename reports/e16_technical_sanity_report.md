# E1.6 Technical Sanity Checks

## Inference Configuration
- max_new_tokens: `8`
- generation parameters: `model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)`
- do_sample enabled: `False`
- decoding: greedy / deterministic decoding because do_sample=False and no beams are set
- stopping criteria: none explicit in run_context_pollution_eval.py
- output truncation handling: generated ids are sliced after input_len and decoded; no explicit truncation flag is logged
- finish_reason logged in E1: no

## Prompt And Image Consistency
- System prompt: no separate system prompt is used in `run_context_pollution_eval.py`.
- User prompt and answer instruction are produced by the same `strict_prompt()` for all conditions.
- Option shuffle is deterministic per sample and seed, independent of condition.
- All E1 conditions use contact-sheet mode by default.
- Contact sheet labels are consistently formatted as `Original image` and `Evidence-N`.
- Contact sheet resolution changes with number of evidence crops, by design: original-only uses 1100x760; crop conditions use original panel plus evidence grid.
- Text formatting is otherwise condition-invariant except for the number/order of evidence crops.

## Condition Summary
| condition            | total | invalid_count | invalid_rate | suspicious_count | suspicious_rate | avg_input_length_valid | avg_input_length_invalid | avg_output_tokens_valid | avg_output_tokens_invalid | invalid_near_max_rate |
| -------------------- | ----- | ------------- | ------------ | ---------------- | --------------- | ---------------------- | ------------------------ | ----------------------- | ------------------------- | --------------------- |
| gt_crop_only         | 238   | 4             | 0.0168       | 4                | 0.0168          | 1355.0812              | 1352.2500                | 1.4872                  | 8.0000                    | 1.0000                |
| gt_plus_1_irrelevant | 238   | 5             | 0.0210       | 5                | 0.0210          | 1355.0815              | 1352.8000                | 1.3948                  | 8.0000                    | 1.0000                |
| gt_plus_4_irrelevant | 238   | 13            | 0.0546       | 13               | 0.0546          | 1598.1733              | 1595.6154                | 1.5511                  | 8.0000                    | 1.0000                |
| gt_plus_8_irrelevant | 238   | 13            | 0.0546       | 16               | 0.0672          | 1598.1644              | 1595.7692                | 1.6311                  | 8.0000                    | 1.0000                |
| original_only        | 238   | 2             | 0.0084       | 2                | 0.0084          | 1085.0593              | 1082.0000                | 1.4703                  | 8.0000                    | 1.0000                |

## Invalid/Suspicious Diagnosis
| condition            | diagnosis         | count |
| -------------------- | ----------------- | ----- |
| gt_crop_only         | likely_truncation | 4     |
| gt_plus_1_irrelevant | likely_truncation | 5     |
| gt_plus_4_irrelevant | likely_truncation | 13    |
| gt_plus_8_irrelevant | likely_abstention | 3     |
| gt_plus_8_irrelevant | likely_truncation | 13    |
| original_only        | likely_truncation | 2     |

## Targeted Rerun
Rows rerun: `58`.
Among invalid/suspicious reruns, inferred length finish rate: `1.0000`.
Matched valid reruns can also reach the length cap, so inferred finish reason alone is not diagnostic; the stronger evidence is that invalid decoded outputs are 8-token prose fragments rather than option letters.

## Answers
A. +4/+8 invalid outputs likely caused by max_new_tokens truncation: **yes**. The E1 generation budget was only 8 tokens, and invalid prose frequently hits the max token budget.
B. Context length overflow / image-token truncation: **unlikely**. Diagnostics show input lengths are far below typical Qwen2.5-VL context limits, and the failures are output-length shaped rather than input overflow shaped.
C. Prompt/message formats: **consistent across conditions** except for contact-sheet visual content and evidence grid size.
D. Suspicious outputs mostly complete abstentions: **no**. Many are abstention-like starts that are cut off by the 8-token generation cap.
E. E1/E1.5 remains directionally valid, but invalid/incomplete-rate claims and +4/+8 invalid-as-wrong drops should be interpreted as partly generation-budget-sensitive. A minimal +4/+8 rerun with higher max_new_tokens is recommended before making strong claims about abstention rates.

## Minimal Rerun Plan If Requested
- Rerun only `gt_plus_4_irrelevant` and `gt_plus_8_irrelevant`.
- Use the same seed, crops, contact sheets, model, and scoring pipeline.
- Increase `--max_new_tokens` from 8 to 32 or 64.
- Keep `do_sample=False`.
- Log output token length and inferred finish reason.

# E1.5 Context Pollution Validity Analysis

## Motivation
E1 showed that adding irrelevant crops to the visual context reduces Qwen2.5-VL accuracy on V*Bench. E1.5 checks whether that result is robust after separating invalid outputs, testing paired statistical reliability, and inspecting GT crop position effects.

## Files Used
- Input CSV: `outputs/results_context_pollution_full.csv`
- No model inference, crop generation, dataset download, or model download was run.

## Valid / Invalid Definition
A generation is valid only when `pred_answer_letter` is one of `A`, `B`, `C`, or `D`. Empty predictions, parsing failures, non-option prose, and refusal-like outputs are invalid/incomplete.

## Column Notes
- category inferred from sample_id prefix or image_path parent directory
- missing expected columns: category

## Condition-Level Summary
| condition            | total_samples | valid_samples | invalid_rate | accuracy_invalid_as_wrong | valid_only_accuracy | drop_vs_gt_crop_only_valid_only | flip_rate_vs_gt_crop_only_valid_comparable |
| -------------------- | ------------- | ------------- | ------------ | ------------------------- | ------------------- | ------------------------------- | ------------------------------------------ |
| original_only        | 238           | 236           | 0.0084       | 0.6597                    | 0.6653              | 0.1296                          | 0.2845                                     |
| gt_crop_only         | 238           | 234           | 0.0168       | 0.7815                    | 0.7949              | 0.0000                          |                                            |
| gt_plus_1_irrelevant | 238           | 233           | 0.0210       | 0.7605                    | 0.7768              | 0.0180                          | 0.0996                                     |
| gt_plus_4_irrelevant | 238           | 225           | 0.0546       | 0.7311                    | 0.7733              | 0.0215                          | 0.1205                                     |
| gt_plus_8_irrelevant | 238           | 225           | 0.0546       | 0.7185                    | 0.7600              | 0.0349                          | 0.1607                                     |

## Category-Level Summary
| category          | condition            | total_samples | invalid_rate | valid_only_accuracy | drop_vs_gt_crop_only_valid_only | flip_rate_vs_gt_crop_only_valid_comparable |
| ----------------- | -------------------- | ------------- | ------------ | ------------------- | ------------------------------- | ------------------------------------------ |
| GPT4V-hard        | original_only        | 17            | 0.0000       | 0.5294              | 0.3529                          | 0.5294                                     |
| GPT4V-hard        | gt_crop_only         | 17            | 0.0000       | 0.8824              | 0.0000                          |                                            |
| GPT4V-hard        | gt_plus_1_irrelevant | 17            | 0.0000       | 0.9412              | -0.0588                         | 0.0588                                     |
| GPT4V-hard        | gt_plus_4_irrelevant | 17            | 0.0000       | 0.8235              | 0.0588                          | 0.1765                                     |
| GPT4V-hard        | gt_plus_8_irrelevant | 17            | 0.0000       | 0.8824              | 0.0000                          | 0.1176                                     |
| OCR               | original_only        | 30            | 0.0000       | 0.7667              | 0.2333                          | 0.2333                                     |
| OCR               | gt_crop_only         | 30            | 0.0000       | 1.0000              | 0.0000                          |                                            |
| OCR               | gt_plus_1_irrelevant | 30            | 0.0333       | 1.0000              | 0.0000                          | 0.0000                                     |
| OCR               | gt_plus_4_irrelevant | 30            | 0.0000       | 1.0000              | 0.0000                          | 0.0000                                     |
| OCR               | gt_plus_8_irrelevant | 30            | 0.0333       | 1.0000              | 0.0000                          | 0.0000                                     |
| direct_attributes | original_only        | 115           | 0.0174       | 0.6195              | 0.1553                          | 0.2936                                     |
| direct_attributes | gt_crop_only         | 115           | 0.0348       | 0.7748              | 0.0000                          |                                            |
| direct_attributes | gt_plus_1_irrelevant | 115           | 0.0348       | 0.7297              | 0.0450                          | 0.1101                                     |
| direct_attributes | gt_plus_4_irrelevant | 115           | 0.1130       | 0.7647              | 0.0101                          | 0.1683                                     |
| direct_attributes | gt_plus_8_irrelevant | 115           | 0.1043       | 0.7476              | 0.0272                          | 0.2059                                     |
| relative_position | original_only        | 76            | 0.0000       | 0.7237              | 0.0000                          | 0.2368                                     |
| relative_position | gt_crop_only         | 76            | 0.0000       | 0.7237              | 0.0000                          |                                            |
| relative_position | gt_plus_1_irrelevant | 76            | 0.0000       | 0.7237              | 0.0000                          | 0.1316                                     |
| relative_position | gt_plus_4_irrelevant | 76            | 0.0000       | 0.6842              | 0.0395                          | 0.0921                                     |
| relative_position | gt_plus_8_irrelevant | 76            | 0.0000       | 0.6579              | 0.0658                          | 0.1711                                     |

## Bootstrap Significance
| comparison                           | mode             | n_paired_samples | accuracy_gt_crop_only | accuracy_target_condition | mean_difference_gt_minus_target | ci95_low | ci95_high | p_one_sided_diff_gt_0 | p_two_sided_diff_ne_0 |
| ------------------------------------ | ---------------- | ---------------- | --------------------- | ------------------------- | ------------------------------- | -------- | --------- | --------------------- | --------------------- |
| gt_crop_only_vs_gt_plus_1_irrelevant | invalid_as_wrong | 238              | 0.7815                | 0.7605                    | 0.0210                          | -0.0168  | 0.0588    | 0.1651                | 0.3302                |
| gt_crop_only_vs_gt_plus_1_irrelevant | valid_only       | 231              | 0.7965                | 0.7835                    | 0.0130                          | -0.0260  | 0.0519    | 0.3070                | 0.6139                |
| gt_crop_only_vs_gt_plus_4_irrelevant | invalid_as_wrong | 238              | 0.7815                | 0.7311                    | 0.0504                          | 0.0084   | 0.0966    | 0.0126                | 0.0252                |
| gt_crop_only_vs_gt_plus_4_irrelevant | valid_only       | 224              | 0.8170                | 0.7768                    | 0.0402                          | -0.0045  | 0.0848    | 0.0429                | 0.0858                |
| gt_crop_only_vs_gt_plus_8_irrelevant | invalid_as_wrong | 238              | 0.7815                | 0.7185                    | 0.0630                          | 0.0168   | 0.1134    | 0.0056                | 0.0112                |
| gt_crop_only_vs_gt_plus_8_irrelevant | valid_only       | 224              | 0.8125                | 0.7634                    | 0.0491                          | 0.0000   | 0.0982    | 0.0315                | 0.0630                |

## GT Crop Position Findings
The largest absolute Pearson correlation between numeric GT crop position and invalid-as-wrong correctness is `0.1296`.
The detailed grouped table is saved to `outputs/e15_gt_position_analysis.csv`.

## Suspicious Output Findings
Suspicious output rows: `37`.
| section        | key                                             | count |
| -------------- | ----------------------------------------------- | ----- |
| condition      | gt_crop_only                                    | 4     |
| condition      | gt_plus_1_irrelevant                            | 5     |
| condition      | gt_plus_4_irrelevant                            | 13    |
| condition      | gt_plus_8_irrelevant                            | 13    |
| condition      | original_only                                   | 2     |
| category       | OCR                                             | 2     |
| category       | direct_attributes                               | 35    |
| top_raw_prefix | The image provided does not contain any visible | 9     |
| top_raw_prefix | The image does not provide enough detail to     | 7     |
| top_raw_prefix | The image provided does not contain any helmets | 4     |
| top_raw_prefix | The image does not contain an umbrella,         | 4     |
| top_raw_prefix | The image provided does not contain any tissue  | 2     |

## Explicit Answers
A. Valid-only monotonic drop remains: **yes**.
B. Invalid/incomplete rate increases with more irrelevant crops: **yes**. It contributes to the invalid-as-wrong drop, especially at +4/+8.
C. +4 and +8 statistical reliability: see bootstrap table. In invalid-as-wrong mode both +4/+8 should be judged by one-sided p-values; valid-only reliability is reported separately.
D. GT crop position likely confound: **unlikely**, based on weak observed association and direct position grouping.
E. Most sensitive categories by +8 valid-only drop: `relative_position` first, then `direct_attributes`.

## Conclusion
E1 remains supported if the valid-only accuracy trend still declines from GT-only through +1/+4/+8 and the paired bootstrap intervals for +4/+8 are mostly positive. The invalid-output analysis clarifies how much of the effect is format failure versus wrong valid choices.

## Caveats
- Valid-only filtering changes the estimand by dropping harder cases where the model failed to follow the output format.
- Bootstrap samples over V*Bench sample IDs and does not account for broader dataset construction uncertainty.
- GT crop position is randomized deterministically, but position groups can be uneven for some conditions.

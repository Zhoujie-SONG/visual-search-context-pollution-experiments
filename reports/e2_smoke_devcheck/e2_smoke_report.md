# E2 Smoke Report

1. Few-shot mode: with one generic few-shot fallback

2. Gate A tool-use numbers
{
  "n": 1,
  "tool_use_rate": 0.0,
  "mean_crops_per_episode": 0.0,
  "median_crops_per_episode": 0,
  "num_tool_calls_distribution": {
    "0": 1
  },
  "fraction_ge_2_crops": 0.0,
  "accuracy": 0.0,
  "invalid_rate": 0.0,
  "oom_rate": 0.0,
  "forced_answer_rate": 0.0,
  "ok_episodes": 1,
  "passes": false
}

3. Accuracy and invalid/OOM audit
- Accuracy: 0.0000
- Invalid rate: 0.0000
- OOM rate: 0.0000
- Forced-answer rate: 0.0000

4. Crop-label prevalence on smoke
- This oracle label is based on annotated GT target overlap and may undercount contextual evidence, especially for relation/context questions.
- useful crops: 0
- failed_or_non_gt crops: 0
- episodes with >=1 failed_or_non_gt crop: 0 / 1

5. Gate B HTML paths and label sanity notes
- Gate B not run because Gate A failed.

6. Oracle eviction smoke results
- Not run because Gate A failed or no failed_or_non_gt crops were found.

7. Replay sanity: original-rebuild vs live-run mismatch rate
- NA

8. Recommendation
- Gate A failed. Stop and consider adjusted prompt or DeepEyes checkpoint; do not run full E2.
# Round-6 Memory Surgery Devcheck Integrity Report

- Overall: **PASS**
- Samples: 20
- Arms: `['A_full']`
- Episode records: 20
- Prefixes are replayed from frozen baseline crop files; no prefix inference is run.
- GPU workers read sanitized prefixes and index-only plans, with no GT answer/bbox/category/coverage fields.
- B/C/E use one identical observation-eviction placeholder and retain exactly two prefix crop images.
- Evicted observations are absent from the tool-source registry; retained source IDs are never renumbered.
- Working-memory capacity is separate from cumulative acquisition budget (7 images at intervention).
- Post-intervention successful crops begin at `observation_7`.
- D never executes a post-intervention crop; any grounding tag takes precedence and is a protocol violation.
- Original image and assistant textual history remain present in every arm.
- Baseline files are read-only and fingerprint-checked.

## Reconstruction Match Rates

- Samples: 20
- Behavioral gates: `{'next_action_match': 0.95, 'next_bbox_match': 0.95, 'final_answer_match': 0.95, 'later_crop_count_match': 0.95, 'suffix_behavior_match': 0.9}`
- Observed match rates: `{'next_output_exact_match': 1.0, 'next_action_match': 1.0, 'next_bbox_match': 1.0, 'final_answer_match': 1.0, 'later_crop_count_match': 1.0, 'suffix_behavior_match': 1.0}`
- `next_output_exact_match` is diagnostic only and is not a PASS gate.

## Problems
- None.

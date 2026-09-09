# Round-6 Memory Surgery Smoke Integrity Report

- Overall: **PASS**
- Samples: 10
- Arms: `['A_full', 'B_oracle_top2', 'C_recent_top2', 'D_force_answer_r6', 'E_random_top2']`
- Episode records: 50
- Prefixes are replayed from frozen baseline crop files; no prefix inference is run.
- GPU workers read sanitized prefixes and index-only plans, with no GT answer/bbox/category/coverage fields.
- B/C/E use one identical observation-eviction placeholder and retain exactly two prefix crop images.
- Evicted observations are absent from the tool-source registry; retained source IDs are never renumbered.
- Working-memory capacity is separate from cumulative acquisition budget (7 images at intervention).
- Post-intervention successful crops begin at `observation_7`.
- D never executes a post-intervention crop; any grounding tag takes precedence and is a protocol violation.
- Original image and assistant textual history remain present in every arm.
- Baseline files are read-only and fingerprint-checked.

## Problems
- None.

# VisualNeedle Mini-o3 Full300 Integrity Report

- Overall: **PASS**
- Official sample records: 300
- Episode records: 300
- Unique episode IDs: 300
- Category counts: `{'OCR Recognition': 71, 'Color Recognition': 73, 'Entity Recognition': 64, 'Spatial Relationship': 58, 'Occluded Object Recognition': 34}`
- Crop files checked: 2069 raw + 2069 processed
- Inference image provenance: original files only from `images/`; generated observations only from recorded crops.
- `images_bbox/`, GT bbox, category, and GT answer are not accepted by `run_episode`.
- Evaluation metadata was loaded only after each shard completed all inference episodes.
- No duplicate-crop guard, GT guidance, forced-answer turn, crop injection, or sample replacement is enabled.
- Raw model output is retained in each turn record.

## Problems
- None.

## Final Scoring Integrity

- Judge model(s): `['gemini-3.8-flash']`
- Unique parsed judge records: 138
- Judge pending: 0
- Final correct sample IDs: 29
- Episode/judge completeness: **PASS**

# VisualNeedle Mini-o3 Smoke20 Integrity Report

- Overall: **PASS**
- Frozen sample records: 20
- Episode records: 20
- Unique episode IDs: 20
- Category counts: `{'Color Recognition': 4, 'Entity Recognition': 4, 'OCR Recognition': 4, 'Occluded Object Recognition': 4, 'Spatial Relationship': 4}`
- Crop files checked: 120 raw + 120 processed
- Inference image provenance: original files only from `images/`; generated observations only from recorded crops.
- `images_bbox/`, GT bbox, category, and GT answer are not accepted by `run_episode`.
- Evaluation metadata was loaded only after each shard completed all inference episodes.
- No duplicate-crop guard, GT guidance, forced-answer turn, crop injection, or sample replacement is enabled.
- Raw model output is retained in each turn record.

## Problems
- None.

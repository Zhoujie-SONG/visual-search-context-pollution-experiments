# Mini-o3 VisualNeedle KV-cache Multi-turn Validation

## 一致性

- EXACT_TRAJECTORY_MATCH: **6/20**
- SEMANTIC_TRAJECTORY_MATCH: **0/20**
- TRAJECTORY_DIVERGENCE: **14/20**
- final answer exact consistency: **13/20 (65.0%)**
- final answer normalized consistency: **13/20 (65.0%)**
- normalized accuracy OLD / NEW: **5.0% / 5.0%**

## 首次分叉

| sample | first turn | fields | OLD status | NEW status |
|---|---:|---|---|---|
| TPROMPT17518843f8cc | 3 | `["action_type", "bbox_valid", "crop_executed", "crop_presence", "final_answer", "predicted_bbox_clipped", "predicted_bbox_raw", "raw_model_output_content", "source"]` | completed | completed |
| TPROMPT5e284ebedfca | 1 | `["raw_model_output_content"]` | completed | completed |
| TPROMPT433eb7d38409 | 1 | `["raw_model_output_content"]` | completed | completed |
| TPROMPT3b824b5818d8 | 1 | `["raw_model_output_content"]` | completed | completed |
| TPROMPT37318e20e89a | 2 | `["crop.original_crop_width", "crop.predicted_bbox_clipped", "crop.predicted_bbox_raw", "crop.processed_width", "predicted_bbox_clipped", "predicted_bbox_raw", "raw_model_output_content"]` | completed | completed |
| TPROMPT13b616f8c9dc | 1 | `["raw_model_output_content"]` | completed | completed |
| TPROMPT11712e5ce7e5 | 1 | `["crop.original_crop_height", "crop.predicted_bbox_clipped", "crop.predicted_bbox_raw", "crop.processed_height", "predicted_bbox_clipped", "predicted_bbox_raw", "raw_model_output_content"]` | completed | completed |
| TPROMPT6a35efbe56dd | 1 | `["crop.original_crop_height", "crop.predicted_bbox_clipped", "crop.predicted_bbox_raw", "crop.processed_height", "crop.processed_width", "predicted_bbox_clipped", "predicted_bbox_raw", "raw_model_output_content"]` | invalid | completed |
| TPROMPT70cc4aa9b0ad | 7 | `["action_type", "bbox_valid", "crop_executed", "crop_presence", "predicted_bbox_clipped", "predicted_bbox_raw", "raw_model_output_content", "source"]` | invalid | oom |
| TPROMPTf75544649373 | 2 | `["crop.original_crop_width", "crop.predicted_bbox_clipped", "crop.predicted_bbox_raw", "crop.processed_width", "predicted_bbox_clipped", "predicted_bbox_raw", "raw_model_output_content"]` | completed | completed |
| TPROMPT176f197b8403 | 1 | `["raw_model_output_content"]` | completed | completed |
| TPROMPTb85097d0357f | 2 | `["crop.original_crop_width", "crop.predicted_bbox_clipped", "crop.predicted_bbox_raw", "crop.processed_height", "crop.processed_width", "predicted_bbox_clipped", "predicted_bbox_raw", "raw_model_output_content"]` | completed | completed |
| TPROMPT4636b72ec2d0 | 3 | `["raw_model_output_content"]` | completed | completed |
| TPROMPTcc36d493e6fa | 1 | `["crop.original_crop_height", "crop.original_crop_width", "crop.predicted_bbox_clipped", "crop.predicted_bbox_raw", "crop.processed_height", "crop.processed_width", "predicted_bbox_clipped", "predicted_bbox_raw", "raw_model_output_content"]` | invalid | completed |

## 性能

- mean episode runtime OLD / NEW: **1515.11s / 38.82s**
- median episode runtime OLD / NEW: **1350.04s / 24.03s**
- max episode runtime OLD / NEW: **3508.13s / 114.87s**
- approximate 8-shard wall time OLD / NEW: **5500.90s / 174.48s**
- aggregate runtime speedup: **39.03x**
- median per-sample speedup: **35.54x**
- mean peak-memory increase: **378.3 MB**
- `TPROMPT65ab072c3b73` OLD / NEW: **3467.48s / 71.18s (48.71x)**

## 结论

- 是否可安全固化 `use_cache=True`: **尚不能直接判定**
- 本轮只改变标准的 within-generate KV cache；未使用跨轮 cache，也未改变 prompt、processor、crop 或 termination policy。
- 分叉样本需结合逐轮 CSV 判断：若 OLD 在 `max_time=600` 被截断而 NEW 正常完成，该差异属于修复既有时间截断，而不是 cache 改变了截断前的模型决策。

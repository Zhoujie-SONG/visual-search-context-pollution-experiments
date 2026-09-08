# VisualNeedle-300 Mini-o3 Baseline Report

## Accuracy

- Raw exact match: 3.00% (9/300)
- Official normalized match: 4.33% (13/300)
- Official final accuracy: 9.67% (29/300)
- Judge pending: 0 (final accuracy complete: True)

- Judge model: gemini-3.8-flash
- Judge protocol: official VisualNeedle image-aware YES/NO prompt; user-authorized Flash substitution.

The official evaluator first applies `answers_match`, then sends unmatched answers and the original image to its configured VLM judge. If pending rows exist, they are not silently counted as judged and the displayed final value is only a lower bound.

## Accuracy by Category

| Category | N | Exact | Normalized | Final |
|---|---:|---:|---:|---:|
| Color Recognition | 73 | 0.00% | 2.74% | 9.59% |
| Entity Recognition | 64 | 0.00% | 0.00% | 7.81% |
| OCR Recognition | 71 | 8.45% | 9.86% | 12.68% |
| Occluded Object Recognition | 34 | 0.00% | 2.94% | 11.76% |
| Spatial Relationship | 58 | 5.17% | 5.17% | 6.90% |

## Search Behavior

- Mean / median crops: 6.897 / 5.500
- GT coverage >=10% hit: 66.67% (200/300)
- GT center hit: 63.00% (189/300)
- Geometrically irrelevant crops: 63.90% (1322/2069)
- Repeated/high-overlap crops: 36.44% (754/2069)
- Hit GT but wrong: 86.00% (172/200)
- Correct without GT hit: 1

## Reliability

- Status counts: `{'completed': 284, 'invalid': 10, 'oom': 6}`
- Max-round episodes: 137
- Max-image episodes: 133
- Mean episode runtime: 46.91s
- Summed episode runtime: 3.91 GPU-hours
- Final baseline-correct IDs: 29

## Next-stage Readiness

The run is technically complete and suitable for a paired misleading-crop pilot. However, only 29 baseline-correct samples are eligible for correct-to-wrong analysis, while 6 OOM, 10 invalid, and 137 max-round episodes limit statistical power. Preserve these limitations in downstream claims.

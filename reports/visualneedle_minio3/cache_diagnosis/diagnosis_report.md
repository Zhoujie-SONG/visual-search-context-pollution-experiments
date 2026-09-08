# Mini-o3 KV Cache Diagnosis

## 配置

- checkpoint `model.config.use_cache`: `None`
- checkpoint `model.config.text_config.use_cache`: `False`
- checkpoint `generation_config.use_cache`: `False`
- attention implementation: `sdpa`
- torch / transformers: `2.5.1` / `5.13.0`
- GPU: `NVIDIA GeForce RTX 3090` (`CUDA_VISIBLE_DEVICES=0`)
- model dtype: `torch.bfloat16`
- A/B: same first-round image, question, official system prompt, processor, BF16, SDPA, greedy decoding, `max_new_tokens=128`; only `use_cache` differs.

## 结果

| setting | generate time (s) | output tokens | tok/s | peak GPU MB | finish | cache returned |
|---|---:|---:|---:|---:|---|---|
| use_cache=False | 134.607 | 128 | 0.951 | 16350.0 | length | False |
| use_cache=True | 3.888 | 128 | 32.926 | 16421.6 | length | True |

- speedup: **34.63x**
- generated token IDs / outputs identical: **True**
- `use_cache=True` returned cache type: `transformers.cache_utils.DynamicCache`
- returned cache sequence length: `2902`
- 是否足以解释 0.4-0.9 token/s: **是**

## 结论与下一步

下一步在独立 smoke 副本上验证显式 `use_cache=True`，确认轨迹逐 token 一致后再考虑正式重跑。

本诊断未修改正式 runner、prompt、crop/termination policy、图像预处理、样本选择或 smoke20 输出。

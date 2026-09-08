#!/usr/bin/env bash
set -euo pipefail

cd /home/songzhoujie/cvpr27/vstar

mkdir -p /mnt/data2/szj/tmp /mnt/data2/szj/hf_cache outputs

export TMPDIR=/mnt/data2/szj/tmp
export HF_HOME=/mnt/data2/szj/hf_cache
export TRANSFORMERS_CACHE=/mnt/data2/szj/hf_cache
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

PYTHON=/home/songzhoujie/miniconda3/envs/cvpr2027/bin/python

echo "[start] $(date)"
echo "[step] resume full inference"
"$PYTHON" run_context_pollution_eval.py \
  --data_root /home/songzhoujie/cvpr27/vstar \
  --crop_dir outputs/crops \
  --output_csv outputs/results_context_pollution_full.csv \
  --model /home/songzhoujie/cvpr27/Qwen2.5-VL-7B-Instruct \
  --num_samples 238 \
  --k_values 0 1 4 8 \
  --resume

echo "[step] score"
"$PYTHON" score_results.py \
  --input_csv outputs/results_context_pollution_full.csv \
  --summary_csv outputs/summary_context_pollution_full.csv \
  --by_category_csv outputs/summary_context_pollution_full_by_category.csv \
  --overwrite

echo "[step] plot"
"$PYTHON" plot_results.py \
  --summary_csv outputs/summary_context_pollution_full.csv \
  --output_png outputs/accuracy_vs_irrelevant_crops_full.png \
  --overwrite

echo "[done] $(date)"

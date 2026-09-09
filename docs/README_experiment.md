# Context Pollution Validation Experiment

This experiment tests whether adding irrelevant visual evidence crops to a VLM context degrades multiple-choice VQA accuracy on V*Bench.

## Dataset

The local V*Bench structure is detected automatically. In this checkout the dataset root is:

```bash
/home/songzhoujie/cvpr27/vstar/data/vstar_bench
```

Each sample is represented as an image file paired with a same-stem annotation JSON inside a category folder. The JSON fields are:

- `question`: VQA question
- `options`: multiple-choice options; `options[0]` is the correct answer before shuffling
- `bbox`: target bbox list in `[x, y, width, height]`
- `target_object`: target object description

## Run A 20-Sample Debug Pass

```bash
cd /home/songzhoujie/cvpr27/vstar

python inspect_vstar.py --data_root data/vstar_bench

python generate_crops.py \
  --data_root data/vstar_bench \
  --output_dir outputs/crops \
  --num_samples 20 \
  --k_values 0 1 4 8 \
  --seed 42

python run_context_pollution_eval.py \
  --data_root data/vstar_bench \
  --crop_dir outputs/crops \
  --output_csv outputs/results_context_pollution.csv \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --num_samples 20 \
  --k_values 0 1 4 8

python score_results.py \
  --input_csv outputs/results_context_pollution.csv \
  --summary_csv outputs/summary_context_pollution.csv

python plot_results.py \
  --summary_csv outputs/summary_context_pollution.csv \
  --output_png outputs/accuracy_vs_irrelevant_crops.png
```

Use `--overwrite` on any command when intentionally replacing prior outputs.

## Run 200 Samples

Repeat the same commands with `--num_samples 200`. Crop generation is deterministic for a fixed `--seed`.

## Conditions

- `original_only`: original image only
- `gt_crop_only`: original image plus GT crop
- `gt_plus_1_irrelevant`: original image plus GT crop plus 1 irrelevant crop
- `gt_plus_4_irrelevant`: original image plus GT crop plus 4 irrelevant crops
- `gt_plus_8_irrelevant`: original image plus GT crop plus 8 irrelevant crops

Irrelevant crops are sampled from the same image with IoU `< 0.1` against the GT bbox and area in `[0.5x, 2.0x]` of the GT crop area. Evidence order is shuffled deterministically so the GT crop is not always first.

## Qwen Backend Notes

`run_context_pollution_eval.py` defaults to contact-sheet input because it is stable across VLM backends. It creates a single image containing the original image and labeled evidence crops. Use `--input_mode multi_image` to pass the original and evidence crops as separate images when the active Qwen2.5-VL environment supports multi-image inference.

If the command cannot import Qwen dependencies, install or activate an environment with:

```bash
pip install torch transformers accelerate qwen-vl-utils pillow
```

No training is performed.

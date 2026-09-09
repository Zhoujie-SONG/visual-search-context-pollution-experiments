# Round-6 Visual Memory Surgery Pilot

This implementation performs paired continuation from the sixth successful
crop observation in the frozen VisualNeedle-300 Mini-o3 baseline. It never
regenerates the prefix.

## Current status

Code only. No devcheck, smoke, formal inference, or Gemini judging has been
started. Read-only validation against the frozen baseline found:

- 300 baseline episodes
- 139 eligible no-answer episodes with at least six successful prefix crops
- 88 primary prefix-hit episodes
- 51 secondary prefix-no-hit episodes
- 695 valid `(sample, arm)` intervention plans across the eligible cohort

## Safety and causal isolation

- `full300_baseline/` is read only and protected with SHA-256 fingerprints.
- Preparation emits sanitized prefix snapshots with no GT answer, category,
  bbox, coverage, or IoU fields.
- GPU workers load only sanitized prefixes and index-only intervention plans.
- B/C/E retain exactly two prefix crops and use the same neutral eviction text.
- Original image and assistant reasoning remain intact in every arm.
- D appends only the preregistered force-answer instruction and executes no crop.
- Formal inference cannot start until the smoke integrity report is PASS.
- Smoke never starts formal inference automatically.

## Review commands

These commands do not load the model or create experiment outputs:

```bash
/mnt/data2/szj/envs/venv_e2m/bin/python -m py_compile \
  run_round6_memory_surgery.py \
  score_round6_memory_surgery.py \
  judge_round6_memory_surgery_gemini.py

/mnt/data2/szj/envs/venv_e2m/bin/python -m unittest -v \
  test_round6_memory_surgery.py

/mnt/data2/szj/envs/venv_e2m/bin/python \
  run_round6_memory_surgery.py --validate-only
```

## Execution gates

Run each stage separately after review:

```bash
./launch_round6_memory_surgery_devcheck.sh
./launch_round6_memory_surgery_smoke.sh
./launch_round6_memory_surgery_judge.sh smoke
```

The formal launcher is intentionally separate:

```bash
./launch_round6_memory_surgery_formal.sh
./launch_round6_memory_surgery_judge.sh formal
```

All launchers are resume-safe at the shard-specific raw episode level. The
judge is also resume-safe per `(sample_id, condition)`.

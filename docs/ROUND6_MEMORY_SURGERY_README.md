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
- Evicted observations are removed from both the prompt and the tool-source
  registry. A later request for an evicted `observation_N` returns a neutral
  `source_unavailable` tool error and does not reload the image.
- Source identities are stable: retained crop 5 remains `observation_5`, and
  the first successful post-intervention crop is always `observation_7`.
- Working-memory sources and cumulative image budget are separate. Every arm
  starts continuation with seven acquired images for `MAX_IMAGES` accounting,
  even when B/C/E retain only original plus two crop images in working memory.
- Original image and assistant reasoning remain intact in every arm.
- D appends only the preregistered force-answer instruction and executes no
  crop. Grounding keeps baseline precedence over answer parsing, so an output
  containing both tags is a force-answer protocol violation, not a valid answer.
- Reconstruction devcheck uses 20 samples. Behavioral gates are 0.95 for next
  action, clipped bbox, final answer, and later crop count, plus 0.90 for full
  suffix behavior. Exact next-output text is diagnostic only.
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

## Interpretation boundary

The intervention removes historical visual observation images but
intentionally preserves the assistant's textual reasoning trace. Therefore,
`B_oracle_top2 > A_full` supports a causal effect of visual observation memory
on continuation behavior. A null difference does not establish complete
absence of context pollution because information from an earlier image may
already have been written into the retained reasoning text.

The primary `GT coverage >= 10%` threshold is a geometric-hit criterion, not a
guarantee of sufficient answer evidence. Scoring separately reports primary
samples with maximum prefix coverage of at least 50% and approximately 100%.
It also emits `oracle_selection_audit.csv` to quantify duplicate coverage-only
Oracle selections.

Automatic `DESCRIPTIVE_*` case labels use point-estimate ordering and a 0.03
approximate-equality tolerance only. They are not statistical decisions;
inference must use the paired bootstrap confidence intervals and McNemar tests.

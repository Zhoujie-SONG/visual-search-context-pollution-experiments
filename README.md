# Visual Search Context Pollution Experiments

Research code, frozen configurations, aggregate tables, reports, and compact
figures for the V*Bench context-pollution and VisualNeedle active visual-search
experiments.

## Repository scope

This repository is a lightweight research record. It intentionally excludes:

- model checkpoints and Hugging Face caches
- source datasets and images
- generated crops, contact sheets, and trajectory visualizations
- API keys and local environment credentials
- large per-turn raw model traces

`code/` contains experiment and analysis scripts copied from the working tree.
`reports/` mirrors publishable files from `outputs/`. `MANIFEST.txt` records the
files included by the most recent synchronization.

The active working project remains `/home/songzhoujie/cvpr27/vstar`. Run
`publish_reports_to_github.sh` there after an experiment finishes to refresh and
push this repository.


# Reproducibility Guide

## Clean-checkout analysis

Run from the repository root on Python 3.13 or newer:

```bash
uv sync --frozen
uv run python -m unittest discover -s tests
uv run ruff check .

uv run python pipeline/stg_03_prepare_data.py --version v1
uv run python pipeline/stg_04_regressions.py --version v1
uv run python pipeline/stg_05_figures.py --version v1
uv run python pipeline/stg_06_appendices.py --version v1

uv run python pipeline/stg_03_prepare_data.py --version v1_secondary
uv run python pipeline/stg_04_regressions.py --version v1_secondary
uv run python pipeline/stg_05_figures.py --version v1_secondary
uv run python pipeline/stg_06_appendices.py --version v1_secondary

uv run python scripts/build_release_artifacts.py
```

Stage 3 downloads the pinned SentenceTransformer revision
`1110a243fdf4706b3f48f1d95db1a4f5529b4d41` on first use. Set `HF_HOME` or
`XDG_CACHE_HOME` to control its cache location.

## Expected coverage

- Versions: `v1` and `v1_secondary`.
- Model checkpoints: 22 per version (10 base/post-trained pairs and 2
  intermediate OLMo post-training checkpoints).
- Trials: 10 per model.
- Generator distributions: 5 pickle files and 5 JSON exports.
- Prepared tables: `output_data/results_<version>.csv.gz`.
- Regression outputs: model coefficients, R-squared summaries, pooled
  coefficients, interaction coefficients, and text summaries for each version.
- Rendered outputs: HTML/PDF figures and LaTeX tables.

The release builder fails on incomplete caches or files that cannot be pushed
to ordinary GitHub storage. It records the exact bytes of the published release
in `release_manifest.json`. Prepared tables and distribution exports are
deterministic; PDF metadata can vary across renderer versions, so validate
rendered filenames and source tables if exact PDF bytes differ.

## Experiment reconstruction

The released historical experiments have an unknown original seed and must be
treated as canonical. To create a new deterministic experiment set from the
released distributions:

```bash
uv run python pipeline/stg_01_setup_experiments.py --version v1 --seed 42
```

To rebuild the distributions first, configure all Snowflake variables and add
`--create-distributions`. This requires authorized access to the proprietary
tables and is outside clean-checkout reproduction.

## Model inference

Install GPU dependencies and run stage 2. All 22 released checkpoints are
selected unless aliases are passed with `--models`.

```bash
uv sync --frozen --extra full
uv run --extra full python pipeline/stg_02_run_experiments.py \
  --version v1 --gpu-id 0 --n-gpus 1
```

The reported evaluation used one NVIDIA H100 GPU, FP8 quantization, temperature
1, and XGrammar guided decoding. The primary evaluation took approximately 24
GPU hours. Inference is hardware- and provider-dependent; the committed trial
files are the canonical model outputs used by the paper.

# Monocultural Biases: Hiring-Bias Experiments

Code and reproducibility artifacts for the COLM 2026 paper **Monocultural Biases:
Correlated biases in large language models lead to unequal systemic exclusion
rates in hiring** by Matthew Bone, Fabian Stephany, and R. Maria del Rio-Chanona.

The study audits hiring decisions from ten LLM families by comparing their base
and post-trained checkpoints. It measures demographic bias, correlation in
model decisions, and systemic exclusion across race, gender, and age groups.

The repository contains the six-stage experiment pipeline, the aggregate
distributions used to generate the experiments, cached experiment definitions,
and model responses for two prompt designs. The released cache covers 22 model
checkpoints: ten base/post-trained pairs plus the supervised fine-tuning and DPO
checkpoints used for the OLMo post-training analysis.

## Reproducibility levels

- **Analysis reproduction (stages 3-6):** fully supported from the committed
  caches and outputs. No proprietary database access or model inference is
  required.
- **Experiment reconstruction (stage 1):** requires access to the proprietary
  person-history and vacancy tables described in [DATA.md](DATA.md). The
  released aggregate distributions and experiment definitions are sufficient
  to inspect the generated populations without those sources.
- **Model inference (stage 2):** requires the listed Hugging Face models, a
  CUDA/vLLM environment, and substantial GPU capacity.

`v1` is the primary paired-applicant callback experiment. It contains 760 base
experiments across 76 occupations, eight demographic variants per experiment,
and ten trials in each applicant order. `v1_secondary` is the supplementary
single-applicant 1-10 rating robustness test. Both versions use the same 22
released checkpoints.

## Installation

Requirements are Python 3.13 or newer and
[`uv`](https://docs.astral.sh/uv/).

```bash
uv sync --frozen
cp .env.example .env
```

The default cache location is `experiment_cache`, so stages 3-6 work without
editing `.env`. Run commands from the repository root: `experiments.pkl` uses
Python pickle and imports classes from `pipeline`. Only unpickle artifacts from
a trusted source.

Stage 3 downloads the pinned
`sentence-transformers/all-MiniLM-L6-v2` revision on first use. To rerun model
inference, install the GPU dependencies separately:

```bash
uv sync --frozen --extra full
```

## Pipeline

Every stage accepts `--version {v1,v1_secondary}` and defaults to `v1`.

```bash
# Proprietary-source reconstruction; omit --create-distributions to use caches
uv run python pipeline/stg_01_setup_experiments.py --version v1 --seed 42 --create-distributions

# GPU model inference (all released checkpoints unless --models is supplied)
uv run --extra full python pipeline/stg_02_run_experiments.py --version v1

# Reproduce prepared data, regressions, main figures, and appendices
uv run python pipeline/stg_03_prepare_data.py --version v1
uv run python pipeline/stg_04_regressions.py --version v1
uv run python pipeline/stg_05_figures.py --version v1
uv run python pipeline/stg_06_appendices.py --version v1
```

Repeat the last four commands with `--version v1_secondary` for the
supplementary analysis. Detailed clean-checkout instructions and verification
criteria are in [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

## Released artifacts

- `experiment_cache/*_dist.pkl`: canonical generator distributions.
- `experiment_cache/distributions/*.json.gz`: portable equivalents with a
  documented schema.
- `experiment_cache/<version>/experiments.pkl`: generated experiment objects.
- `experiment_cache/<version>/<model>/trial_<n>_results.pkl`: model responses.
- `output_data/results_<version>.csv.gz`: prepared analysis tables.
- `output_data/`: regression coefficients, summaries, and prompt dumps.
- `figures/` and `tables/`: publication figures and LaTeX tables.
- `release_manifest.json`: SHA-256 checksums, sizes, distribution counts, and
  cache coverage.

The original generation seed was not recorded, so the committed historical
experiment caches are canonical. Newly generated experiments are deterministic
for a given `--seed`, which is saved in `generation_metadata.json`.

## Tests and release validation

```bash
uv run python -m unittest discover -s tests
uv run ruff check .
uv run python scripts/build_release_artifacts.py
```

The release builder validates the 22-checkpoint x 10-trial cache for both
versions, round-trippable distribution exports, checksums all released
artifacts, and rejects files at or above GitHub's 100 MB limit.

## Data and licensing

The MIT license applies to repository code only. Data and model outputs have
source-specific terms and attribution requirements documented in
[DATA.md](DATA.md). The release contains aggregate distributions rather than
proprietary row-level person histories or job postings.

## Citation

Please cite the associated paper using [CITATION.cff](CITATION.cff) or the
following BibTeX entry:

```bibtex
@inproceedings{bone2026monocultural,
  title     = {Monocultural Biases: Correlated Biases in Large Language Models Lead to Unequal Systemic Exclusion Rates in Hiring},
  author    = {Bone, Matthew and Stephany, Fabian and del Rio-Chanona, R. Maria},
  booktitle = {Proceedings of the Third Conference on Language Modeling (COLM 2026)},
  year      = {2026},
  month     = oct,
  address   = {San Francisco, California, USA}
}
```

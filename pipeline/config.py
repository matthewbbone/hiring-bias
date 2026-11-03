from __future__ import annotations

import argparse
import os
from pathlib import Path


DEFAULT_VERSION = "v1"
SUPPORTED_VERSIONS = ("v1", "v1_secondary")
N_TRIALS = 10

ANALYSIS_MODELS = (
    "nemotron_30b_base",
    "nemotron_30b_fine",
    "granite_32b_base",
    "granite_32b_fine",
    "gemma_27b_base",
    "gemma_27b_fine",
    "olmo_32b_base",
    "olmo_32b_sft",
    "olmo_32b_dpo",
    "olmo_32b_fine",
    "qwen_35b_base",
    "qwen_35b_fine",
    "glm_32b_base",
    "glm_32b_fine",
    "ernie_28b_base",
    "ernie_28b_fine",
    "moonlight_16b_base",
    "moonlight_16b_fine",
    "ministral_14b_base",
    "ministral_14b_fine",
    "falcon_34b_base",
    "falcon_34b_fine",
)

MODEL_REPOSITORIES = {
    "nemotron_30b_base": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16",
    "nemotron_30b_fine": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
    "granite_32b_base": "ibm-granite/granite-4.0-h-small-base",
    "granite_32b_fine": "ibm-granite/granite-4.0-h-small",
    "gemma_27b_base": "google/gemma-3-27b-pt",
    "gemma_27b_fine": "google/gemma-3-27b-it",
    "olmo_32b_base": "allenai/Olmo-3-1125-32B",
    "olmo_32b_sft": "allenai/Olmo-3.1-32B-Instruct-SFT",
    "olmo_32b_dpo": "allenai/Olmo-3.1-32B-Instruct-DPO",
    "olmo_32b_fine": "allenai/Olmo-3.1-32B-Instruct",
    "qwen_35b_base": "Qwen/Qwen3.5-35B-A3B-Base",
    "qwen_35b_fine": "Qwen/Qwen3.5-35B-A3B",
    "glm_32b_base": "zai-org/GLM-4-32B-Base-0414",
    "glm_32b_fine": "zai-org/GLM-4-32B-0414",
    "ernie_28b_base": "baidu/ERNIE-4.5-VL-28B-A3B-Base-PT",
    "ernie_28b_fine": "baidu/ERNIE-4.5-VL-28B-A3B-PT",
    "moonlight_16b_base": "moonshotai/Moonlight-16B-A3B",
    "moonlight_16b_fine": "moonshotai/Moonlight-16B-A3B-Instruct",
    "ministral_14b_base": "mistralai/Ministral-3-14B-Base-2512",
    "ministral_14b_fine": "mistralai/Ministral-3-14B-Instruct-2512",
    "falcon_34b_base": "tiiuae/Falcon-H1-34B-Base",
    "falcon_34b_fine": "tiiuae/Falcon-H1-34B-Instruct",
}

SENTENCE_TRANSFORMER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SENTENCE_TRANSFORMER_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


def cache_dir() -> Path:
    return Path(os.getenv("CACHE_DIR", "experiment_cache"))


def is_secondary_version(version: str) -> bool:
    if version not in SUPPORTED_VERSIONS:
        raise ValueError(
            f"Unsupported version {version!r}; expected one of {SUPPORTED_VERSIONS}."
        )
    return version == "v1_secondary"


def add_version_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--version",
        choices=SUPPORTED_VERSIONS,
        default=DEFAULT_VERSION,
        help=f"Experiment version to process. Defaults to {DEFAULT_VERSION}.",
    )


def resolve_results_path(version: str, output_dir: Path = Path("output_data")) -> Path:
    compressed = output_dir / f"results_{version}.csv.gz"
    uncompressed = output_dir / f"results_{version}.csv"
    if compressed.exists():
        return compressed
    return uncompressed


def validate_experiment_cache(
    version: str,
    *,
    root: Path | None = None,
    models: tuple[str, ...] = ANALYSIS_MODELS,
    n_trials: int = N_TRIALS,
    reject_extra_models: bool = True,
) -> None:
    version_dir = (root or cache_dir()) / version
    missing: list[str] = []
    experiments_path = version_dir / "experiments.pkl"
    if not experiments_path.is_file():
        missing.append(str(experiments_path))

    for model in models:
        for trial in range(1, n_trials + 1):
            path = version_dir / model / f"trial_{trial}_results.pkl"
            if not path.is_file():
                missing.append(str(path))

    expected_models = set(models)
    actual_models = {
        path.name
        for path in version_dir.iterdir()
        if path.is_dir() and path.name != "logs"
    } if version_dir.is_dir() else set()
    extra_models = sorted(actual_models - expected_models)

    problems = []
    if missing:
        preview = "\n  ".join(missing[:20])
        suffix = f"\n  ... and {len(missing) - 20} more" if len(missing) > 20 else ""
        problems.append(f"Missing cache files:\n  {preview}{suffix}")
    if reject_extra_models and extra_models:
        problems.append(f"Unexpected model directories: {', '.join(extra_models)}")
    if problems:
        raise FileNotFoundError(
            f"Incomplete experiment cache for {version}:\n" + "\n".join(problems)
        )

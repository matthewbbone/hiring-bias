import os
import sys
import argparse
from datetime import datetime
from pathlib import Path

import dill as pickle
from dotenv import load_dotenv

load_dotenv()
sys.path.append(str(Path(__file__).parent.parent))

from pipeline.utils.close_llm import ClosedLLMReviewer  # noqa: E402
from pipeline.utils.open_llm import OpenRouterReviewer  # noqa: E402
from pipeline.utils.vllm_llm import VLLMReviewer  # noqa: E402
from pipeline.config import (  # noqa: E402
    ANALYSIS_MODELS,
    MODEL_REPOSITORIES,
    N_TRIALS,
    add_version_argument,
    cache_dir,
    is_secondary_version,
)


VLLM_FP8_EXCLUDED_MODELS = {
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16",
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
    "Qwen/Qwen3.5-35B-A3B-Base",
    "Qwen/Qwen3.5-35B-A3B",
}


def load_experiments(version_dir: Path):
    experiments_path = version_dir / "experiments.pkl"
    if not experiments_path.exists():
        raise FileNotFoundError(
            f"Could not find experiments at {experiments_path}. "
            "Run stage 1 first to create experiments.pkl."
        )

    with open(experiments_path, "rb") as f:
        return pickle.load(f)


def save_trial_results(model_dir: Path, trial: int, results) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    output_path = model_dir / f"trial_{trial}_results.pkl"
    with open(output_path, "wb") as f:
        pickle.dump(results, f)


def run_vllm_experiments(
    version_dir: Path,
    vllm_models: dict[str, str],
    experiments,
    n_trials: int = 10,
    n_gpus: int = 1,
    n_concurrent: int = 10,
    port: int = 8000,
    gpu_ids: str = "0",
    secondary_prompt: bool = False,
) -> None:
    """Run experiments using a local vLLM server."""
    for model, model_name in vllm_models.items():
        print(f"\n{'=' * 60}")
        print(f"Running vLLM model: {model}")
        print(f"{'=' * 60}")

        quantization = None if model in VLLM_FP8_EXCLUDED_MODELS else "fp8"
        reviewer = VLLMReviewer(
            llm=model,
            n_gpus=n_gpus,
            port=port,
            gpu_ids=gpu_ids,
            quantization=quantization,
        )

        try:
            for trial in range(1, n_trials + 1):
                print(f"Starting trial {trial} for model {model}...")
                results = reviewer.review(
                    experiments,
                    max_concurrent=n_concurrent,
                    secondary_prompt=secondary_prompt,
                )
                save_trial_results(version_dir / model_name, trial, results)
        finally:
            reviewer.close()
            del reviewer


def run_openrouter_experiments(
    version_dir: Path,
    open_models: dict[str, str],
    experiments,
    n_trials: int = 10,
    n_concurrent: int = 2,
    secondary_prompt: bool = False,
) -> None:
    """Run experiments using OpenRouter API."""
    for model, model_name in open_models.items():
        print(f"\n{'=' * 60}")
        print(f"Running OpenRouter model: {model}")
        print(f"{'=' * 60}")

        with OpenRouterReviewer(llm=model, use_json_schema=True) as reviewer:
            for trial in range(1, n_trials + 1):
                print(f"Starting trial {trial} for model {model}...")
                results = reviewer.review(
                    experiments,
                    max_concurrent=n_concurrent,
                    secondary_prompt=secondary_prompt,
                )
                save_trial_results(version_dir / model_name, trial, results)


def run_closed_experiments(
    version_dir: Path,
    close_models: dict[str, str],
    experiments,
    n_trials: int = 10,
    n_workers: int = 10,
    secondary_prompt: bool = False,
) -> None:
    """Run experiments using llm-utils for closed-source models."""
    for model, model_name in close_models.items():
        print(f"\n{'=' * 60}")
        print(f"Running closed model: {model}")
        print(f"{'=' * 60}")

        with ClosedLLMReviewer(llm=model, n_workers=n_workers) as reviewer:
            for trial in range(1, n_trials + 1):
                print(f"Starting trial {trial} for model {model}...")
                results = reviewer.review(
                    experiments,
                    secondary_prompt=secondary_prompt,
                )
                save_trial_results(version_dir / model_name, trial, results)


def run_all_experiments(
    version_dir: Path,
    vllm_models: dict[str, str] | None = None,
    open_models: dict[str, str] | None = None,
    close_models: dict[str, str] | None = None,
    n_trials: int = 10,
    n_gpus: int = 1,
    n_concurrent: int = 10,
    openrouter_concurrent: int = 2,
    closed_workers: int = 10,
    port: int = 8000,
    gpu_ids: str = "0",
    secondary_prompt: bool = False,
) -> None:
    """Run experiments on vLLM, OpenRouter, and closed-source models."""
    experiments = load_experiments(version_dir)
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    start = datetime.now()

    print(f"Loaded {len(experiments)} experiments from {version_dir / 'experiments.pkl'}")
    print(f"Experiment run timestamp: {run_timestamp}")

    if vllm_models:
        print(f"\nRunning {len(vllm_models)} vLLM models...")
        run_vllm_experiments(
            version_dir=version_dir,
            vllm_models=vllm_models,
            experiments=experiments,
            n_trials=n_trials,
            n_gpus=n_gpus,
            n_concurrent=n_concurrent,
            port=port,
            gpu_ids=gpu_ids,
            secondary_prompt=secondary_prompt,
        )

    if open_models:
        print(f"\nRunning {len(open_models)} OpenRouter models...")
        run_openrouter_experiments(
            version_dir=version_dir,
            open_models=open_models,
            experiments=experiments,
            n_trials=n_trials,
            n_concurrent=openrouter_concurrent,
            secondary_prompt=secondary_prompt,
        )

    if close_models:
        print(f"\nRunning {len(close_models)} closed-source models...")
        run_closed_experiments(
            version_dir=version_dir,
            close_models=close_models,
            experiments=experiments,
            n_trials=n_trials,
            n_workers=closed_workers,
            secondary_prompt=secondary_prompt,
        )

    elapsed_minutes = (datetime.now() - start).total_seconds() / 60
    print(f"\nTotal experiment time (mins): {elapsed_minutes:.2f}")


def main(
    version: str,
    n_trials: int = 10,
    n_gpus: int = 1,
    n_concurrent: int = 10,
    openrouter_concurrent: int = 2,
    closed_workers: int = 10,
    vllm_models: dict[str, str] | None = None,
    open_models: dict[str, str] | None = None,
    close_models: dict[str, str] | None = None,
    port: int = 8000,
    gpu_ids: str = "0",
    secondary_prompt: bool = False,
) -> None:
    print("Checking Environment Variables...")
    print("CACHE_DIR:", os.getenv("CACHE_DIR"))
    print("XDG_CACHE_HOME:", os.getenv("XDG_CACHE_HOME"))
    print("XDG_CONFIG_HOME:", os.getenv("XDG_CONFIG_HOME"))
    print("TRITON_CACHE_DIR:", os.getenv("TRITON_CACHE_DIR"))
    print("CUDA_VISIBLE_DEVICES:", os.getenv("CUDA_VISIBLE_DEVICES"))

    version_dir = cache_dir() / version
    version_dir.mkdir(parents=True, exist_ok=True)

    print("Run experiments...")
    run_all_experiments(
        version_dir=version_dir,
        vllm_models=vllm_models,
        open_models=open_models,
        close_models=close_models,
        n_trials=n_trials,
        n_gpus=n_gpus,
        n_concurrent=n_concurrent,
        openrouter_concurrent=openrouter_concurrent,
        closed_workers=closed_workers,
        port=port,
        gpu_ids=gpu_ids,
        secondary_prompt=secondary_prompt,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the model evaluation experiments.")
    add_version_argument(parser)
    parser.add_argument(
        "--gpu-id",
        "--gpu_id",
        dest="gpu_id",
        type=int,
        choices=range(8),
        default=0,
        help="GPU ID to use for local vLLM models. Defaults to 0.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=ANALYSIS_MODELS,
        default=list(ANALYSIS_MODELS),
        help="Analyzed model aliases to run. Defaults to all publication models.",
    )
    parser.add_argument("--n-trials", type=int, default=N_TRIALS)
    parser.add_argument("--n-gpus", type=int, default=1)
    parser.add_argument("--n-concurrent", type=int, default=20)
    parser.add_argument("--port", type=int, default=8036)
    args = parser.parse_args()

    vllm_models = {
        MODEL_REPOSITORIES[alias]: alias
        for alias in args.models
    }

    main(
        version=args.version,
        n_trials=args.n_trials,
        n_gpus=args.n_gpus,
        n_concurrent=args.n_concurrent,
        openrouter_concurrent=2,
        closed_workers=10,
        vllm_models=vllm_models,
        open_models=None,
        close_models=None,
        port=args.port,
        gpu_ids=str(args.gpu_id),
        secondary_prompt=is_secondary_version(args.version),
    )

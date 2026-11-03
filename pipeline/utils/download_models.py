from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import snapshot_download

load_dotenv(override=True)


DEFAULT_MODELS = [
    "allenai/Olmo-3.1-32B-Instruct-SFT",
    "allenai/Olmo-3.1-32B-Instruct-DPO",
    "allenai/Olmo-3-1125-32B",
    "allenai/Olmo-3.1-32B-Instruct",
    "google/gemma-3-27b-pt",
    "google/gemma-3-27b-it",
    "mistralai/Ministral-3-14B-Base-2512",
    "mistralai/Ministral-3-14B-Instruct-2512",
    "Qwen/Qwen3.5-35B-A3B-Base",
    "Qwen/Qwen3.5-35B-A3B",
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16",
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
    "tiiuae/Falcon-H1-34B-Base",
    "tiiuae/Falcon-H1-34B-Instruct",
    "zai-org/GLM-4-32B-Base-0414",
    "zai-org/GLM-4-32B-0414",
    "moonshotai/Moonlight-16B-A3B",
    "moonshotai/Moonlight-16B-A3B-Instruct",
    "ibm-granite/granite-4.0-h-small-base",
    "ibm-granite/granite-4.0-h-small",
]

DEFAULT_IGNORE_PATTERNS = [
    "*.onnx",
    "*.tflite",
    "*.h5",
    "*.ot",
    "*.tar.gz",
]


def default_xdg_cache_home() -> Path:
    cache_root = os.environ.get("XDG_CACHE_HOME")
    if cache_root:
        return Path(cache_root)
    return Path.home() / ".cache"


def default_hf_home() -> Path:
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home)
    return default_xdg_cache_home() / "huggingface"


def default_hf_cache_dir() -> Path:
    return default_hf_home() / "hub"


def download_snapshot(
    model_id: str,
    cache_dir: Path,
    ignore_patterns: list[str] | None = None,
) -> Path:
    print(f"\nDownloading {model_id}...")
    snapshot_path = snapshot_download(
        repo_id=model_id,
        cache_dir=str(cache_dir),
        ignore_patterns=ignore_patterns,
        resume_download=True,
    )
    snapshot_path = Path(snapshot_path)
    print(f"Downloaded to {snapshot_path}")
    return snapshot_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Hugging Face model snapshots into the default HF cache."
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=DEFAULT_MODELS,
        help="Model repo IDs to download.",
    )
    parser.add_argument(
        "--hf-cache-dir",
        type=Path,
        default=default_hf_cache_dir(),
        help="Hugging Face cache root used by snapshot_download.",
    )
    parser.add_argument(
        "--ignore-pattern",
        action="append",
        default=[],
        help="Additional snapshot_download ignore patterns.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ignore_patterns = DEFAULT_IGNORE_PATTERNS + args.ignore_pattern

    args.hf_cache_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, dict[str, str]] = {}
    failures: list[str] = []

    for model_id in args.models:
        try:
            snapshot_path = download_snapshot(
                model_id=model_id,
                cache_dir=args.hf_cache_dir,
                ignore_patterns=ignore_patterns,
            )
            summary[model_id] = {
                "snapshot": str(snapshot_path),
                "status": "downloaded",
            }
            print(f"✓ {model_id} downloaded successfully.")
        except Exception as exc:
            failures.append(model_id)
            summary[model_id] = {
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
            print(f"✗ Failed to download {model_id}: {exc}")

    manifest_path = args.hf_cache_dir / "download_manifest.json"
    manifest_path.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote manifest to {manifest_path}")

    if failures:
        print("\nThe following models failed:")
        for model_id in failures:
            print(f" - {model_id}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

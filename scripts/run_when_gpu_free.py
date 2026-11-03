from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass


DEFAULT_GPU_IDS = tuple(range(4))


@dataclass(frozen=True)
class GpuMemory:
    gpu_id: int
    free_mb: int
    total_mb: int

    @property
    def free_fraction(self) -> float:
        if self.total_mb <= 0:
            return 0.0
        return self.free_mb / self.total_mb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Wait for a GPU with enough free memory, then launch stage 2 "
            "experiments on that GPU."
        )
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=30.0,
        help="Seconds to wait between GPU memory checks. Defaults to 30.",
    )
    parser.add_argument(
        "--free-threshold",
        type=float,
        default=0.80,
        help="Required free-memory fraction. Defaults to 0.80.",
    )
    parser.add_argument(
        "--launch-delay",
        type=float,
        default=30.0,
        help="Seconds to wait after selecting a free GPU before launching. Defaults to 30.",
    )
    parser.add_argument(
        "--gpu-ids",
        type=int,
        nargs="+",
        default=list(DEFAULT_GPU_IDS),
        choices=range(4),
        help="GPU IDs to consider. Defaults to 0 1 2 3.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the command that would run, but do not launch it.",
    )
    return parser.parse_args()


def query_gpu_memory() -> list[GpuMemory]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.free,memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    gpu_memory: list[GpuMemory] = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3:
            raise ValueError(f"Could not parse nvidia-smi row: {line!r}")
        gpu_id, free_mb, total_mb = (int(part) for part in parts)
        gpu_memory.append(GpuMemory(gpu_id=gpu_id, free_mb=free_mb, total_mb=total_mb))
    return gpu_memory


def choose_gpu(
    gpu_memory: list[GpuMemory],
    allowed_gpu_ids: set[int],
    free_threshold: float,
) -> GpuMemory | None:
    candidates = [
        gpu
        for gpu in gpu_memory
        if gpu.gpu_id in allowed_gpu_ids and gpu.free_fraction >= free_threshold
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda gpu: (gpu.free_fraction, gpu.free_mb))


def build_command(gpu_id: int) -> list[str]:
    return [
        "uv",
        "run",
        "--extra",
        "full",
        "pipeline/stg_02_run_experiments.py",
        "--gpu-id",
        str(gpu_id),
    ]


def main() -> int:
    args = parse_args()
    allowed_gpu_ids = set(args.gpu_ids)

    if not 0 < args.free_threshold <= 1:
        raise ValueError("--free-threshold must be greater than 0 and at most 1.")
    if args.interval <= 0:
        raise ValueError("--interval must be greater than 0.")
    if args.launch_delay < 0:
        raise ValueError("--launch-delay must be non-negative.")

    while True:
        gpu_memory = query_gpu_memory()
        chosen_gpu = choose_gpu(gpu_memory, allowed_gpu_ids, args.free_threshold)
        checked = ", ".join(
            f"GPU {gpu.gpu_id}: {gpu.free_fraction:.1%} free"
            for gpu in gpu_memory
            if gpu.gpu_id in allowed_gpu_ids
        )

        if chosen_gpu is not None:
            command = build_command(chosen_gpu.gpu_id)
            print(
                f"Selected GPU {chosen_gpu.gpu_id} "
                f"({chosen_gpu.free_fraction:.1%} free)."
            )
            print("Running:", " ".join(command))
            if args.dry_run:
                return 0
            if args.launch_delay:
                print(
                    f"Waiting {args.launch_delay:g}s before launching.",
                    flush=True,
                )
                time.sleep(args.launch_delay)
            os.execvp(command[0], command)

        print(
            f"No GPU has >= {args.free_threshold:.0%} memory free. "
            f"Checked: {checked}. Sleeping {args.interval:g}s.",
            flush=True,
        )
        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())

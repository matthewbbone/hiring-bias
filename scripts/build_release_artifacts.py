from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import pickle
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from pipeline.config import (  # noqa: E402
    ANALYSIS_MODELS,
    N_TRIALS,
    SUPPORTED_VERSIONS,
    validate_experiment_cache,
)


CACHE_DIR = ROOT / "experiment_cache"
EXPORT_DIR = CACHE_DIR / "distributions"
MANIFEST_PATH = ROOT / "release_manifest.json"
MAX_GITHUB_FILE_SIZE = 100_000_000

DISTRIBUTIONS = {
    "occupation": "occ_dist.pkl",
    "vacancy": "vacancy_dist.pkl",
    "degree_major": "degree_major_dist.pkl",
    "first_name": "first_name_dist.pkl",
    "last_name": "last_name_dist.pkl",
}

RELEASE_ROOTS = ("experiment_cache", "raw_data", "output_data", "figures", "tables")

DATA_PROVENANCE = [
    {
        "paths": [
            "experiment_cache/occ_dist.pkl",
            "experiment_cache/distributions/occupation.json.gz",
        ],
        "source": "Aggregate proprietary person-history and occupation inputs",
        "derivation": "Occupation-conditioned degree, skill, transition, and experience weights",
        "license_status": "Cleared for research redistribution; not licensed under MIT",
    },
    {
        "paths": [
            "experiment_cache/vacancy_dist.pkl",
            "experiment_cache/distributions/vacancy.json.gz",
        ],
        "source": "Aggregate proprietary vacancy inputs",
        "derivation": "Occupation-conditioned company, title, skill, and log-salary parameters",
        "license_status": "Cleared for research redistribution; not licensed under MIT",
    },
    {
        "paths": [
            "experiment_cache/degree_major_dist.pkl",
            "experiment_cache/distributions/degree_major.json.gz",
        ],
        "source": "Aggregate proprietary education-history inputs",
        "derivation": "University weights conditional on degree major",
        "license_status": "Cleared for research redistribution; not licensed under MIT",
    },
    {
        "paths": [
            "experiment_cache/first_name_dist.pkl",
            "experiment_cache/distributions/first_name.json.gz",
            "raw_data/names/*",
        ],
        "source": "U.S. Social Security Administration national baby-name data",
        "derivation": "First-name weights by birth year and gender",
        "license_status": "Retained under U.S. government source terms; not licensed under MIT",
    },
    {
        "paths": [
            "experiment_cache/last_name_dist.pkl",
            "experiment_cache/distributions/last_name.json.gz",
            "raw_data/Names_2010Census.csv",
        ],
        "source": "2010 U.S. Census surname data",
        "derivation": "Surname choices and frequencies",
        "license_status": "Retained under U.S. government source terms; not licensed under MIT",
    },
    {
        "paths": ["raw_data/Education, Training, and Experience.xlsx"],
        "source": "O*NET database",
        "derivation": "Education ratings used to identify bachelor-level occupations",
        "license_status": "Retained under O*NET upstream terms; not licensed under MIT",
    },
    {
        "paths": ["raw_data/audit_study_names.csv"],
        "source": "Kline, Rose, and Walters, A Discrimination Report Card, public replication data",
        "derivation": "Unique race, gender, first-name, and last-name combinations; no application records",
        "license_status": "Redistributed for research replication under upstream terms; not licensed under MIT",
    },
    {
        "paths": ["raw_data/universities.py"],
        "source": "Project-curated university list",
        "derivation": "University choices used by the degree-major distribution builder",
        "license_status": "Repository-authored code covered by MIT",
    },
]


def json_compatible(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_compatible(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def write_deterministic_json_gzip(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="\n") as text:
                json.dump(value, text, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
                text.write("\n")


def export_distributions() -> dict[str, int]:
    row_counts: dict[str, int] = {}
    for name, pickle_name in DISTRIBUTIONS.items():
        with (CACHE_DIR / pickle_name).open("rb") as source:
            distribution = pickle.load(source)
        payload = json_compatible(distribution)
        write_deterministic_json_gzip(EXPORT_DIR / f"{name}.json.gz", payload)
        row_counts[name] = len(distribution)
    return row_counts


def write_historical_metadata() -> None:
    for version in SUPPORTED_VERSIONS:
        path = CACHE_DIR / version / "generation_metadata.json"
        if not path.exists():
            path.write_text(
                json.dumps(
                    {
                        "version": version,
                        "seed": None,
                        "n_exp_per_soc": 10,
                        "historical_cache": True,
                        "note": "The original generation seed was not recorded.",
                    },
                    indent=2,
                ) + "\n"
            )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def release_files() -> list[Path]:
    files: list[Path] = []
    for root_name in RELEASE_ROOTS:
        root = ROOT / root_name
        if root.exists():
            files.extend(
                path
                for path in root.rglob("*")
                if path.is_file()
                and "__pycache__" not in path.parts
                and path.suffix != ".pyc"
            )
    return sorted(files)


def build_manifest(row_counts: dict[str, int]) -> dict[str, Any]:
    files = release_files()
    oversized = [path for path in files if path.stat().st_size >= MAX_GITHUB_FILE_SIZE]
    if oversized:
        paths = ", ".join(str(path.relative_to(ROOT)) for path in oversized)
        raise ValueError(f"Files exceed GitHub's 100 MB limit: {paths}")

    cache_summary = {}
    for version in SUPPORTED_VERSIONS:
        validate_experiment_cache(version, root=CACHE_DIR)
        cache_summary[version] = {
            "models": list(ANALYSIS_MODELS),
            "model_count": len(ANALYSIS_MODELS),
            "trials_per_model": N_TRIALS,
            "generation_seed": None,
            "seed_status": "Historical canonical cache; original seed was not recorded",
        }

    return {
        "schema_version": 1,
        "primary_version": "v1",
        "supplementary_versions": ["v1_secondary"],
        "distribution_row_counts": row_counts,
        "data_provenance": DATA_PROVENANCE,
        "experiment_caches": cache_summary,
        "files": [
            {
                "path": str(path.relative_to(ROOT)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in files
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build publication distribution exports and checksums.")
    parser.parse_args()
    row_counts = export_distributions()
    write_historical_metadata()
    manifest = build_manifest(row_counts)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {len(DISTRIBUTIONS)} distribution exports and {MANIFEST_PATH.name}.")


if __name__ == "__main__":
    main()

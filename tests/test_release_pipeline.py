import gzip
import json
import os
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from pipeline.config import (
    ANALYSIS_MODELS,
    DEFAULT_VERSION,
    SUPPORTED_VERSIONS,
    cache_dir,
    validate_experiment_cache,
)
from pipeline.stg_03_prepare_data import normalize_outcome
from pipeline.stg_04_regressions import (
    POOLED_REGRESSION_FORMULA,
    SECONDARY_POOLED_REGRESSION_FORMULA,
    deterministic_summary_text,
    pooled_regression_formula,
)
from pipeline.stg_05_figures import _bootstrap_exclusion_rate
from pipeline.utils.experiments import Experiment
from scripts.build_release_artifacts import DISTRIBUTIONS, json_compatible


class ReleaseConfigurationTests(unittest.TestCase):
    def test_primary_version_and_model_manifest(self):
        self.assertEqual(DEFAULT_VERSION, "v1")
        self.assertEqual(SUPPORTED_VERSIONS, ("v1", "v1_secondary"))
        self.assertEqual(len(ANALYSIS_MODELS), 22)
        self.assertEqual(len(set(ANALYSIS_MODELS)), 22)

    def test_cache_directory_has_portable_default(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(cache_dir(), Path("experiment_cache"))

    def test_version_specific_outcome_transform(self):
        values = pd.Series([1.0, 5.0, 10.0])
        self.assertTrue(normalize_outcome(values, "v1").equals(values))
        expected = pd.Series([0.0, 0.4, 0.9])
        self.assertTrue(normalize_outcome(values, "v1_secondary").equals(expected))

    def test_version_specific_pooled_formula(self):
        self.assertEqual(pooled_regression_formula("v1"), POOLED_REGRESSION_FORMULA)
        self.assertEqual(
            pooled_regression_formula("v1_secondary"),
            SECONDARY_POOLED_REGRESSION_FORMULA,
        )

    def test_cache_validator_reports_missing_trials(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            version_dir = root / "v1"
            version_dir.mkdir()
            (version_dir / "experiments.pkl").touch()
            with self.assertRaisesRegex(FileNotFoundError, "trial_1_results.pkl"):
                validate_experiment_cache(
                    "v1",
                    root=root,
                    models=("example_model",),
                    n_trials=1,
                )

    def test_saved_regression_summaries_remove_wall_clock_fields(self):
        class FakeResult:
            @staticmethod
            def summary():
                return (
                    "Date:                Mon, 13 Jul 2026   Prob (F-statistic): 0.01\n"
                    "Time:                        06:24:02   Log-Likelihood: 12.3"
                )

        summary = deterministic_summary_text(FakeResult())
        self.assertNotIn("13 Jul 2026", summary)
        self.assertNotIn("06:24:02", summary)


class DistributionExportTests(unittest.TestCase):
    def test_distribution_exports_round_trip(self):
        cache_dir = Path("experiment_cache")
        export_dir = cache_dir / "distributions"
        for name, pickle_name in DISTRIBUTIONS.items():
            with (cache_dir / pickle_name).open("rb") as stream:
                expected = json_compatible(pickle.load(stream))
            with gzip.open(export_dir / f"{name}.json.gz", "rt", encoding="utf-8") as stream:
                actual = json.load(stream)
            self.assertEqual(actual, expected, name)

    def test_manifest_has_provenance_and_seed_status(self):
        manifest = json.loads(Path("release_manifest.json").read_text())
        self.assertGreaterEqual(len(manifest["data_provenance"]), 8)
        for record in manifest["data_provenance"]:
            self.assertTrue(record["paths"])
            self.assertTrue(record["source"])
            self.assertTrue(record["license_status"])
        for cache in manifest["experiment_caches"].values():
            self.assertIsNone(cache["generation_seed"])
            self.assertIn("not recorded", cache["seed_status"])


class DeterministicGenerationTests(unittest.TestCase):
    def test_seeded_experiment_sampling_is_repeatable(self):
        first = Experiment("a", None, None, {}, rng=np.random.default_rng(7))
        second = Experiment("b", None, None, {}, rng=np.random.default_rng(7))
        first_values = first.rng.integers(0, 10_000, size=20).tolist()
        second_values = second.rng.integers(0, 10_000, size=20).tolist()
        self.assertEqual(first_values, second_values)

    def test_binary_bootstrap_is_seeded_and_bounded(self):
        values = np.array([0.0, 1.0, 1.0, 0.0])
        first = _bootstrap_exclusion_rate(values, np.random.default_rng(42), n_boot=1_000)
        second = _bootstrap_exclusion_rate(values, np.random.default_rng(42), n_boot=1_000)
        self.assertEqual(first, second)
        self.assertEqual(first[0], 0.5)
        self.assertTrue(0 <= first[1] <= first[2] <= 1)


class NamesReleaseTests(unittest.TestCase):
    def test_names_release_contains_only_required_unique_columns(self):
        names = pd.read_csv("raw_data/audit_study_names.csv")
        self.assertEqual(
            list(names.columns),
            ["race", "gender", "firstname", "lastname"],
        )
        self.assertEqual(len(names), len(names.drop_duplicates()))


if __name__ == "__main__":
    unittest.main()

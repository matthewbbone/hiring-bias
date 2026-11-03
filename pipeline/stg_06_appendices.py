from __future__ import annotations

import argparse
import importlib
import re
import sys
import types
from contextlib import contextmanager
from itertools import combinations
from pathlib import Path

import dill as pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dotenv import load_dotenv
from plotly.subplots import make_subplots
from scipy.stats import pearsonr

sys.path.append(str(Path(__file__).parent.parent))

from pipeline.stg_05_figures import (
    STATUS_COLORS,
    STATUS_ORDER,
    SYSTEMIC_EXCLUSION_THRESHOLD,
    _base_layout,
    coefficient_display_df,
    exclude_presentation_models,
    load_regression_coefficients,
    prepare_systemic_exclusion_df,
    resolve_csv_path,
    save_plotly_figure,
)
from pipeline.config import (
    DEFAULT_VERSION,
    add_version_argument,
    cache_dir,
    resolve_results_path,
)


VERSION = DEFAULT_VERSION
RHO_VALUES = np.round(np.arange(0.1, 1.0, 0.1), 2)
MODEL_REGRESSION_TERM_ORDER = [
    "black",
    "female",
    "over40",
    "title_similarity",
    "skill_similarity",
    "degree_similarity",
]
MODEL_REGRESSION_TERM_LABELS = {
    "black": "Black",
    "female": "Female",
    "over40": "Over 40",
    "title_similarity": r"$q_t$",
    "skill_similarity": r"$q_s$",
    "degree_similarity": r"$q_d$",
}
SECONDARY_MODEL_REGRESSION_TERM_ORDER = [
    "black",
    "female",
    "over40",
    "perturbed_title_similarity",
    "perturbed_skills_similarity",
    "perturbed_degree_similarity",
]
SECONDARY_MODEL_REGRESSION_TERM_LABELS = {
    "black": "Black",
    "female": "Female",
    "over40": "Over 40",
    "perturbed_title_similarity": r"$q_t$",
    "perturbed_skills_similarity": r"$q_s$",
    "perturbed_degree_similarity": r"$q_d$",
}
MODEL_REGRESSION_INTERACTION_TERM_ORDER = [
    *MODEL_REGRESSION_TERM_ORDER,
    "black:female",
    "black:over40",
    "female:over40",
    "black:female:over40",
]
MODEL_REGRESSION_INTERACTION_TERM_LABELS = {
    **MODEL_REGRESSION_TERM_LABELS,
    "black": "Black (b)",
    "female": "Female (f)",
    "over40": "Over 40 (o)",
    "black:female": r"b$\times$f",
    "black:over40": r"b$\times$o",
    "female:over40": r"f$\times$o",
    "black:female:over40": r"b$\times$f$\times$o",
}
SECONDARY_MODEL_REGRESSION_INTERACTION_TERM_ORDER = [
    *SECONDARY_MODEL_REGRESSION_TERM_ORDER,
    "black:female",
    "black:over40",
    "female:over40",
    "black:female:over40",
]
SECONDARY_MODEL_REGRESSION_INTERACTION_TERM_LABELS = {
    **SECONDARY_MODEL_REGRESSION_TERM_LABELS,
    "black": "Black (b)",
    "female": "Female (f)",
    "over40": "Over 40 (o)",
    "black:female": r"b$\times$f",
    "black:over40": r"b$\times$o",
    "female:over40": r"f$\times$o",
    "black:female:over40": r"b$\times$f$\times$o",
}
MODEL_REGRESSION_MODEL_ORDER = [
    "ernie_28b",
    "falcon_34b",
    "gemma_27b",
    "glm_32b",
    "granite_32b",
    "ministral_14b",
    "moonlight_16b",
    "nemotron_30b",
    "olmo_32b",
    "qwen_35b",
]
MODEL_DISPLAY_LABELS = {
    "ernie_28b": "Ernie 4.5",
    "falcon_34b": "Falcon H1",
    "gemma_27b": "Gemma 3",
    "glm_32b": "GLM 4",
    "granite_32b": "Granite 4",
    "ministral_14b": "Ministral 3",
    "moonlight_16b": "Moonlight",
    "nemotron_30b": "Nemotron Nano 3",
    "olmo_32b": "Olmo 3",
    "qwen_35b": "Qwen 3.5",
}
MODEL_REGRESSION_MODEL_LABELS = MODEL_DISPLAY_LABELS
APPENDIX_PAIRWISE_ROW_LABELS = {
    "demographic_deltas": "Bias",
    "experiment_means": "Quality",
}
APPENDIX_PAIRWISE_TERM_TITLES = {
    "experiment_means": "Residual Correlation",
    "demographic_deltas": "Bias Correlation",
}
APPENDIX_PAIRWISE_ROW_ORDER = ["experiment_means", "demographic_deltas"]
APPENDIX_PAIRWISE_MODEL_ORDER = [
    "ernie_28b",
    "falcon_34b",
    "gemma_27b",
    "glm_32b",
    "granite_32b",
    "ministral_14b",
    "moonlight_16b",
    "nemotron_30b",
    "olmo_32b",
    "qwen_35b",
]
APPENDIX_PAIRWISE_MODEL_LABELS = MODEL_DISPLAY_LABELS
POOLED_REGRESSION_TERM_ORDER = [
    "black",
    "female",
    "over40",
    "post",
    "black:post",
    "female:post",
    "over40:post",
    "title_similarity",
    "skill_similarity",
    "degree_similarity",
]
POOLED_REGRESSION_TERM_LABELS = {
    "black": "Black",
    "female": "Female",
    "over40": "Over 40",
    "post": "Post-train",
    "black:post": r"Black $\times$ Post-train",
    "female:post": r"Female $\times$ Post-train",
    "over40:post": r"Over 40 $\times$ Post-train",
    "title_similarity": r"$q_t$ (title-to-title)",
    "skill_similarity": r"$q_s$ (skill-to-skill)",
    "degree_similarity": r"$q_d$ (title-to-degree)",
}
OLMO_BIAS_TERM_ORDER = [
    "black",
    "female",
    "over40",
    "title_similarity",
    "skill_similarity",
    "degree_similarity",
]
OLMO_BIAS_TERM_LABELS = {
    "black": "Black",
    "female": "Female",
    "over40": "Over 40",
    "title_similarity": r"$q_t$ (title-to-title)",
    "skill_similarity": r"$q_s$ (skill-to-skill)",
    "degree_similarity": r"$q_d$ (title-to-degree)",
}
OLMO_MODEL_VERSION_ORDER = [
    "olmo_32b_base",
    "olmo_32b_sft",
    "olmo_32b_dpo",
    "olmo_32b_fine",
]
OLMO_MODEL_VERSION_LABELS = {
    "olmo_32b_base": "OLMo 3 Base",
    "olmo_32b_fine": "OLMo 3.1 Instruct",
    "olmo_32b_sft": "OLMo 3.1 SFT",
    "olmo_32b_dpo": "OLMo 3.1 DPO",
}
OLMO_MODEL_VERSION_SHORT_LABELS = {
    "olmo_32b_base": "Base",
    "olmo_32b_sft": "+SFT",
    "olmo_32b_dpo": "+DPO",
    "olmo_32b_fine": "+RLVR",
}
OLMO_RSQUARED_SPEC_ORDER = ["demographics_only", "similarity_only"]
OLMO_RSQUARED_SPEC_LABELS = {
    "demographics_only": "Demographics",
    "similarity_only": "Human Capital Controls",
}


def load_prepared_results(version: str) -> pd.DataFrame:
    path = resolve_results_path(version)
    if not path.exists():
        raise FileNotFoundError(f"Could not find {path}. Run stage 3 first.")
    return pd.read_csv(path)


@contextmanager
def _temporary_unpickle_dependency_stubs():
    created_modules: dict[str, types.ModuleType] = {}

    def ensure_module(name: str, module: types.ModuleType) -> None:
        if name not in sys.modules:
            sys.modules[name] = module
            created_modules[name] = module

    try:
        importlib.import_module("snowflake.snowpark")
    except ModuleNotFoundError:
        snowflake_module = types.ModuleType("snowflake")
        snowpark_module = types.ModuleType("snowflake.snowpark")
        snowpark_functions_module = types.ModuleType("snowflake.snowpark.functions")
        snowpark_window_module = types.ModuleType("snowflake.snowpark.window")

        class Session:  # pragma: no cover - only used as import stub
            pass

        class DataFrame:  # pragma: no cover - only used as import stub
            pass

        class Window:  # pragma: no cover - only used as import stub
            pass

        snowflake_module.snowpark = snowpark_module
        snowpark_module.Session = Session
        snowpark_module.DataFrame = DataFrame
        snowpark_module.functions = snowpark_functions_module
        snowpark_module.window = snowpark_window_module
        snowpark_window_module.Window = Window

        ensure_module("snowflake", snowflake_module)
        ensure_module("snowflake.snowpark", snowpark_module)
        ensure_module("snowflake.snowpark.functions", snowpark_functions_module)
        ensure_module("snowflake.snowpark.window", snowpark_window_module)

    try:
        importlib.import_module("retry")
    except ModuleNotFoundError:
        retry_module = types.ModuleType("retry")

        def retry(*args, **kwargs):  # pragma: no cover - only used as import stub
            def decorator(func):
                return func

            return decorator

        retry_module.retry = retry
        ensure_module("retry", retry_module)

    try:
        yield
    finally:
        for name in reversed(list(created_modules)):
            sys.modules.pop(name, None)


def load_variant_ages(version: str) -> pd.DataFrame:
    load_dotenv()
    experiments_path = cache_dir().expanduser().resolve() / version / "experiments.pkl"
    if not experiments_path.exists():
        raise FileNotFoundError(f"Could not find cached experiments at {experiments_path}.")

    with _temporary_unpickle_dependency_stubs():
        with experiments_path.open("rb") as f:
            experiments = pickle.load(f)

    rows: list[dict[str, int]] = []
    for experiment in experiments:
        for variant_idx, profile in enumerate(experiment.variants):
            rows.append(
                {
                    "experiment_id": int(experiment.experiment_id),
                    "variant_idx": int(variant_idx),
                    "age": int(profile.age),
                }
            )

    return pd.DataFrame(rows).sort_values(["experiment_id", "variant_idx"]).reset_index(drop=True)


def _model_status(model: str) -> str:
    return "Base" if model.endswith("_base") else "Post-train"


def _trial_sort_key(path: Path) -> int:
    match = re.search(r"trial_(\d+)_results\.pkl$", path.name)
    if match is None:
        raise ValueError(f"Could not parse trial number from {path.name}.")
    return int(match.group(1))


def prepare_na_rate_df(results_df: pd.DataFrame, version: str) -> pd.DataFrame:
    load_dotenv()
    version_dir = cache_dir().expanduser().resolve() / version
    if not version_dir.exists():
        raise FileNotFoundError(f"Could not find cached results at {version_dir}.")

    model_metadata_df = (
        exclude_presentation_models(results_df).loc[:, ["model", "family"]]
        .drop_duplicates()
        .sort_values(["family", "model"])
        .reset_index(drop=True)
    )

    rows: list[dict[str, str | int | float]] = []
    for metadata in model_metadata_df.itertuples(index=False):
        model_dir = version_dir / metadata.model
        trial_paths = sorted(model_dir.glob("trial_*_results.pkl"), key=_trial_sort_key)
        if not trial_paths:
            raise FileNotFoundError(f"Could not find any trial results for model {metadata.model} in {model_dir}.")

        total_responses = 0
        na_count = 0
        for trial_path in trial_paths:
            with trial_path.open("rb") as f:
                trial_results = pickle.load(f)

            total_responses += len(trial_results)
            na_count += sum(result.get("picked_perturbed") is None for result in trial_results)

        rows.append(
            {
                "status": _model_status(metadata.model),
                "family": metadata.family,
                "model": metadata.model,
                "n_trials": len(trial_paths),
                "na_count": int(na_count),
                "total_responses": int(total_responses),
                "na_rate": float(na_count / total_responses) if total_responses else np.nan,
            }
        )

    na_df = pd.DataFrame(rows)
    status_order = pd.Categorical(na_df["status"], categories=STATUS_ORDER, ordered=True)
    return (
        na_df.assign(status=status_order)
        .sort_values(["status", "family", "model"])
        .reset_index(drop=True)
    )


def _latex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    escaped = value
    for old, new in replacements.items():
        escaped = escaped.replace(old, new)
    return escaped


def _format_model_label(model_family: str) -> str:
    if model_family in MODEL_REGRESSION_MODEL_LABELS:
        return MODEL_REGRESSION_MODEL_LABELS[model_family]
    return model_family.replace("_", "-")


def _appendix_model_family(model_name: str) -> str:
    return model_name.rsplit("_", 1)[0]


def _format_appendix_pairwise_model_label(family: str) -> str:
    if family in APPENDIX_PAIRWISE_MODEL_LABELS:
        return APPENDIX_PAIRWISE_MODEL_LABELS[family]
    return family.replace("_", "-")


def _blend_channel(start: int, end: int, weight: float) -> int:
    return int(round(start + (end - start) * weight))


def _coefficient_to_hex_color(value: float, max_abs: float) -> str:
    if not np.isfinite(value) or max_abs <= 0:
        return "FFFFFF"

    intensity = min(abs(float(value)) / float(max_abs), 1.0)
    if value >= 0:
        rgb = (
            _blend_channel(255, 46, intensity),
            _blend_channel(255, 190, intensity),
            _blend_channel(255, 115, intensity),
        )
    else:
        rgb = (
            _blend_channel(255, 231, intensity),
            _blend_channel(255, 76, intensity),
            _blend_channel(255, 60, intensity),
        )
    return "".join(f"{channel:02X}" for channel in rgb)


def _prepare_appendix_pairwise_correlations(
    plot_df: pd.DataFrame,
    *,
    index_cols: list[str],
    value_col: str,
    term: str,
) -> list[dict[str, float | str | int]]:
    rows: list[dict[str, float | str | int]] = []
    for status in STATUS_ORDER:
        status_df = plot_df.loc[plot_df["status"] == status]
        models = sorted(status_df["model"].dropna().unique())
        for model_a, model_b in combinations(models, 2):
            left = status_df.loc[
                status_df["model"] == model_a,
                [*index_cols, value_col],
            ].rename(columns={value_col: "value_a"})
            right = status_df.loc[
                status_df["model"] == model_b,
                [*index_cols, value_col],
            ].rename(columns={value_col: "value_b"})
            merged = left.merge(right, on=index_cols, how="inner").dropna()
            if len(merged) < 2 or merged["value_a"].nunique() < 2 or merged["value_b"].nunique() < 2:
                continue
            corr = pearsonr(merged["value_a"], merged["value_b"])
            rows.append(
                {
                    "term": term,
                    "term_label": APPENDIX_PAIRWISE_ROW_LABELS[term],
                    "status": status,
                    "model_a": model_a,
                    "model_b": model_b,
                    "pearson_r": float(corr.statistic),
                    "p_value": float(corr.pvalue),
                    "n_obs": int(len(merged)),
                }
            )
    return rows


def prepare_appendix_pairwise_correlation_df(results_df: pd.DataFrame) -> pd.DataFrame:
    plot_df = exclude_presentation_models(results_df).assign(
        status=lambda df: np.where(df["model"].str.endswith("_base"), "Base", "Post-train"),
        picked_perturbed=lambda df: pd.to_numeric(df["picked_perturbed"], errors="coerce"),
    )

    baseline_df = (
        plot_df.loc[plot_df["variant_idx"] == 0, ["model", "status", "experiment_id", "picked_perturbed"]]
        .rename(columns={"picked_perturbed": "baseline_outcome"})
    )
    delta_df = (
        plot_df.loc[
            plot_df["intersection_group"] != "white_male_young",
            ["model", "status", "experiment_id", "variant_idx", "picked_perturbed", "intersection_group"],
        ]
        .merge(
            baseline_df,
            on=["model", "status", "experiment_id"],
            how="left",
        )
        .assign(demographic_delta=lambda df: df["picked_perturbed"] - df["baseline_outcome"])
        .assign(
            demographic_delta=lambda df: df.groupby(["model", "status", "experiment_id"])["demographic_delta"].transform(
                lambda values: (
                    (values - values.mean()) / values.std(ddof=0)
                    if len(values) > 1 and not np.isclose(values.std(ddof=0), 0.0)
                    else np.nan
                )
            )
        )
        .dropna(subset=["demographic_delta"])
    )

    experiment_mean_df = (
        plot_df.groupby(["model", "status", "experiment_id"], as_index=False)["picked_perturbed"]
        .mean()
        .rename(columns={"picked_perturbed": "experiment_mean"})
    )

    rows: list[dict[str, float | str | int]] = []
    rows.extend(
        _prepare_appendix_pairwise_correlations(
            delta_df,
            index_cols=["experiment_id", "variant_idx"],
            value_col="demographic_delta",
            term="demographic_deltas",
        )
    )
    rows.extend(
        _prepare_appendix_pairwise_correlations(
            experiment_mean_df,
            index_cols=["experiment_id"],
            value_col="experiment_mean",
            term="experiment_means",
        )
    )
    return pd.DataFrame(rows)


def _ordered_appendix_pairwise_families(term_df: pd.DataFrame) -> list[str]:
    families = sorted(
        set(term_df["model_a"].map(_appendix_model_family)).union(term_df["model_b"].map(_appendix_model_family))
    )
    family_rank = {family: idx for idx, family in enumerate(APPENDIX_PAIRWISE_MODEL_ORDER)}
    return sorted(families, key=lambda family: (family_rank.get(family, len(family_rank)), family))


def _build_appendix_pairwise_correlation_matrix(term_df: pd.DataFrame) -> tuple[np.ndarray, list[str], list[str]]:
    families = _ordered_appendix_pairwise_families(term_df)
    matrix_df = pd.DataFrame(
        np.eye(len(families), dtype=float),
        index=families,
        columns=families,
    )

    for row in term_df.itertuples(index=False):
        family_a = _appendix_model_family(row.model_a)
        family_b = _appendix_model_family(row.model_b)
        matrix_df.loc[family_a, family_b] = float(row.pearson_r)
        matrix_df.loc[family_b, family_a] = float(row.pearson_r)

    y_families = families
    x_families = list(reversed(families))
    display_y = [_format_appendix_pairwise_model_label(family) for family in y_families]
    display_x = [_format_appendix_pairwise_model_label(family) for family in x_families]
    matrix = matrix_df.loc[y_families, x_families].to_numpy(dtype=float)
    return matrix, display_x, display_y


def _appendix_heatmap_annotation_color(value: float) -> str:
    return "white" if abs(value) >= 0.45 else "#1F2937"


def make_appendix_pairwise_correlation_heatmap_figure(pairwise_df: pd.DataFrame) -> go.Figure:
    subplot_titles = tuple(
        f"{APPENDIX_PAIRWISE_TERM_TITLES[term]}: {status}"
        for term in APPENDIX_PAIRWISE_ROW_ORDER
        for status in STATUS_ORDER
    )
    fig = make_subplots(
        rows=2,
        cols=2,
        shared_xaxes="all",
        shared_yaxes="all",
        horizontal_spacing=0.06,
        vertical_spacing=0.08,
        subplot_titles=subplot_titles,
    )

    colorscale = [
        [0.0, "#3B4CC0"],
        [0.5, "#FFF7EC"],
        [0.75, "#F39C34"],
        [1.0, "#111111"],
    ]

    for row_idx, term in enumerate(APPENDIX_PAIRWISE_ROW_ORDER, start=1):
        for col_idx, status in enumerate(STATUS_ORDER, start=1):
            term_df = pairwise_df.loc[
                (pairwise_df["term"] == term) & (pairwise_df["status"] == status)
            ].copy()
            if term_df.empty:
                continue

            matrix, x_labels, y_labels = _build_appendix_pairwise_correlation_matrix(term_df)
            annotation_text = np.vectorize(
                lambda value: "1" if np.isclose(value, 1.0) else f"{value:.2f}"
            )(matrix)

            fig.add_trace(
                go.Heatmap(
                    z=matrix,
                    x=x_labels,
                    y=y_labels,
                    zmin=-1,
                    zmax=1,
                    zmid=0,
                    coloraxis="coloraxis",
                    xgap=1,
                    ygap=1,
                    hovertemplate=(
                        "Row model: %{y}<br>"
                        "Column model: %{x}<br>"
                        "Pearson's r: %{z:.3f}<extra></extra>"
                    ),
                    showscale=False,
                ),
                row=row_idx,
                col=col_idx,
            )

            axis_idx = col_idx + (row_idx - 1) * 2
            xref = "x" if axis_idx == 1 else f"x{axis_idx}"
            yref = "y" if axis_idx == 1 else f"y{axis_idx}"
            for y_idx, y_label in enumerate(y_labels):
                for x_idx, x_label in enumerate(x_labels):
                    value = float(matrix[y_idx, x_idx])
                    if np.isnan(value):
                        continue
                    fig.add_annotation(
                        x=x_label,
                        y=y_label,
                        text=annotation_text[y_idx, x_idx],
                        showarrow=False,
                        font=dict(size=10, color=_appendix_heatmap_annotation_color(value)),
                        xref=xref,
                        yref=yref,
                    )

            fig.update_xaxes(
                row=row_idx,
                col=col_idx,
                tickangle=35,
                side="bottom",
                showgrid=False,
                zeroline=False,
                showticklabels=(row_idx == 2),
            )
            fig.update_yaxes(
                row=row_idx,
                col=col_idx,
                autorange="reversed",
                showgrid=False,
                zeroline=False,
                showticklabels=(col_idx == 1),
            )

    _base_layout(fig)
    fig.update_layout(
        coloraxis=dict(
            colorscale=colorscale,
            cmin=-1,
            cmax=1,
            colorbar=dict(
                title="Pearson's r",
                ticks="outside",
                x=1.02,
                xanchor="left",
            ),
        ),
        width=980,
        height=900,
        margin=dict(l=40, r=60, t=60, b=40),
    )
    return fig


def prepare_model_regression_table_df(
    coefficients_df: pd.DataFrame,
    specification: str,
    term_order: list[str],
    term_labels: dict[str, str],
) -> pd.DataFrame:
    coefficients_df = exclude_presentation_models(coefficients_df)
    plot_df = (
        coefficients_df.loc[
            (coefficients_df["specification"] == specification)
            & (coefficients_df["term"].isin(term_order))
        ].copy()
        .assign(
            status=lambda df: np.where(df["phase"] == "base", "Base", "Post-train"),
            family=lambda df: df["model"].str.replace(r"_(base|fine)$", "", regex=True),
            model_label=lambda df: df["model"]
            .str.replace(r"_(base|fine)$", "", regex=True)
            .map(_format_model_label),
            term_label=lambda df: df["term"].map(term_labels),
            is_significant=lambda df: df["p_value"] < 0.05,
        )
    )

    if plot_df.empty:
        raise ValueError(f"No model-specific {specification} coefficients were found for the appendix table.")

    missing_terms = (
        plot_df.groupby(["status", "family"])["term"]
        .nunique()
        .reset_index(name="n_terms")
        .loc[lambda df: df["n_terms"] != len(term_order)]
    )
    if not missing_terms.empty:
        raise ValueError(
            f"Model-specific {specification} table is missing one or more terms for: "
            + ", ".join(
                f"{row.status}/{row.family} ({int(row.n_terms)} terms)"
                for row in missing_terms.itertuples(index=False)
            )
        )

    status_order = pd.Categorical(plot_df["status"], categories=STATUS_ORDER, ordered=True)
    family_rank = {family: idx for idx, family in enumerate(MODEL_REGRESSION_MODEL_ORDER)}
    return (
        plot_df.assign(
            status=status_order,
            family_order=lambda df: df["family"].map(family_rank).fillna(len(family_rank)).astype(int),
        )
        .sort_values(["status", "family_order", "family", "term"])
        .reset_index(drop=True)
    )


def is_secondary_version(version: str) -> bool:
    return "secondary" in version.lower()


def model_regression_term_order(version: str) -> list[str]:
    if is_secondary_version(version):
        return SECONDARY_MODEL_REGRESSION_TERM_ORDER
    return MODEL_REGRESSION_TERM_ORDER


def model_regression_term_labels(version: str) -> dict[str, str]:
    if is_secondary_version(version):
        return SECONDARY_MODEL_REGRESSION_TERM_LABELS
    return MODEL_REGRESSION_TERM_LABELS


def model_regression_interaction_term_order(version: str) -> list[str]:
    if is_secondary_version(version):
        return SECONDARY_MODEL_REGRESSION_INTERACTION_TERM_ORDER
    return MODEL_REGRESSION_INTERACTION_TERM_ORDER


def model_regression_interaction_term_labels(version: str) -> dict[str, str]:
    if is_secondary_version(version):
        return SECONDARY_MODEL_REGRESSION_INTERACTION_TERM_LABELS
    return MODEL_REGRESSION_INTERACTION_TERM_LABELS


def build_model_regression_latex_table(
    regression_table_df: pd.DataFrame,
    term_order: list[str],
    term_labels: dict[str, str],
    caption: str,
    label: str,
) -> str:
    column_max_abs = {
        term_label: float(
            regression_table_df.loc[regression_table_df["term_label"] == term_label, "coef"].abs().max()
        )
        for term_label in term_labels.values()
    }
    n_columns = 1 + len(term_order) + 1
    column_format = "l" + ("c" * len(term_order)) + "r"
    header = " & ".join(["Model", *[term_labels[term] for term in term_order], r"$R^2$"]) + r" \\"

    lines = [
        r"\begin{table}[p]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{5pt}",
        r"\renewcommand{\arraystretch}{1.18}",
        rf"\begin{{tabular}}{{{column_format}}}",
        r"\toprule",
        header,
        r"\midrule",
    ]

    for status_idx, status in enumerate(STATUS_ORDER):
        status_df = regression_table_df.loc[regression_table_df["status"] == status].copy()
        if status_df.empty:
            continue

        lines.append(rf"\multicolumn{{{n_columns}}}{{l}}{{\textbf{{{_latex_escape(status)}}}}} \\")

        for family, model_df in status_df.groupby(["family", "model_label"], sort=False):
            _, model_label = family
            cell_values = []
            for term in term_order:
                term_label = term_labels[term]
                term_row = model_df.loc[model_df["term"] == term].iloc[0]
                display_value = f"{term_row['coef']:.2f}"
                if bool(term_row["is_significant"]):
                    display_value += "*"
                cell_color = _coefficient_to_hex_color(float(term_row["coef"]), column_max_abs[term_label])
                cell_values.append(rf"\cellcolor[HTML]{{{cell_color}}}{display_value}")

            row_rsquared = float(model_df["rsquared"].iloc[0])
            lines.append(" & ".join([_latex_escape(model_label), *cell_values, f"{row_rsquared:.3f}"]) + r" \\")

        if status_idx < len(STATUS_ORDER) - 1:
            lines.append(r"\midrule")

    lines.extend(
        [
            r"\hline",
            r"\end{tabular}",
            rf"\caption{{{caption}}}",
            rf"\label{{{label}}}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines)


def write_model_regression_latex_table(coefficients_df: pd.DataFrame, version: str) -> Path:
    term_order = model_regression_term_order(version)
    term_labels = model_regression_term_labels(version)
    regression_table_df = prepare_model_regression_table_df(
        coefficients_df,
        specification="full",
        term_order=term_order,
        term_labels=term_labels,
    )
    tables_dir = Path("tables")
    tables_dir.mkdir(parents=True, exist_ok=True)
    output_path = tables_dir / f"appendix_model_regressions_{version}.tex"
    output_path.write_text(
        build_model_regression_latex_table(
            regression_table_df,
            term_order=term_order,
            term_labels=term_labels,
            caption=(
                r"Model-specific hiring regressions from the full specification. "
                r"Deeper green cells indicate larger positive coefficients and deeper red cells indicate "
                r"larger negative coefficients within each column. Asterisks mark coefficients with "
                r"$p < 0.05$."
            ),
            label="tab:appendix-model-specific-regressions",
        )
        + "\n"
    )
    return output_path


def write_model_interaction_regression_latex_table(coefficients_df: pd.DataFrame, version: str) -> Path:
    term_order = model_regression_interaction_term_order(version)
    term_labels = model_regression_interaction_term_labels(version)
    regression_table_df = prepare_model_regression_table_df(
        coefficients_df,
        specification="full_interactions",
        term_order=term_order,
        term_labels=term_labels,
    )
    tables_dir = Path("tables")
    tables_dir.mkdir(parents=True, exist_ok=True)
    output_path = tables_dir / f"appendix_model_regressions_interactions_{version}.tex"
    output_path.write_text(
        build_model_regression_latex_table(
            regression_table_df,
            term_order=term_order,
            term_labels=term_labels,
            caption=(
                r"Model-specific hiring regressions from the demographic-interactions specification. "
                r"Deeper green cells indicate larger positive coefficients and deeper red cells indicate "
                r"larger negative coefficients within each column. Asterisks mark coefficients with "
                r"$p < 0.05$."
            ),
            label="tab:appendix-model-specific-regressions-interactions",
        )
        + "\n"
    )
    return output_path


def summarize_model_regression_patterns(coefficients_df: pd.DataFrame, version: str) -> pd.DataFrame:
    term_order = model_regression_term_order(version)
    term_labels = model_regression_term_labels(version)
    regression_table_df = prepare_model_regression_table_df(
        coefficients_df,
        specification="full",
        term_order=term_order,
        term_labels=term_labels,
    )
    term_label_order = pd.Categorical(
        regression_table_df["term_label"],
        categories=[term_labels[term] for term in term_order],
        ordered=True,
    )
    summary_df = (
        regression_table_df.assign(term_label=term_label_order)
        .groupby(["status", "term_label"], as_index=False, observed=False)
        .agg(
            mean_coef=("coef", "mean"),
            min_coef=("coef", "min"),
            max_coef=("coef", "max"),
            n_models=("family", "nunique"),
            n_positive=("coef", lambda x: int((x > 0).sum())),
            n_negative=("coef", lambda x: int((x < 0).sum())),
            n_significant=("is_significant", "sum"),
        )
        .sort_values(["status", "term_label"])
        .reset_index(drop=True)
    )
    return summary_df


def load_pooled_regression_coefficients(version: str) -> pd.DataFrame:
    path = resolve_csv_path(f"pooled_regression_coefficients_{version}.csv")
    if not path.exists():
        raise FileNotFoundError(f"Could not find {path}. Run stage 4 first.")
    return coefficient_display_df(pd.read_csv(path), version)


def prepare_pooled_regression_table_df(coefficients_df: pd.DataFrame) -> pd.DataFrame:
    table_df = (
        coefficients_df.loc[coefficients_df["term"].isin(POOLED_REGRESSION_TERM_ORDER)].copy()
        .assign(
            term_label=lambda df: df["term"].map(POOLED_REGRESSION_TERM_LABELS),
            is_significant=lambda df: df["p_value"] < 0.05,
            term_order=lambda df: df["term"].map({term: idx for idx, term in enumerate(POOLED_REGRESSION_TERM_ORDER)}),
        )
        .sort_values("term_order")
        .reset_index(drop=True)
    )
    if table_df.empty:
        raise ValueError("No pooled regression coefficients were found for the appendix summary table.")
    return table_df


def build_pooled_regression_latex_table(coefficients_df: pd.DataFrame) -> str:
    table_df = prepare_pooled_regression_table_df(coefficients_df)
    rsquared = float(coefficients_df["rsquared"].iloc[0])
    nobs = int(round(float(coefficients_df["nobs"].iloc[0])))

    lines = [
        r"\begin{table}[p]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{5pt}",
        r"\renewcommand{\arraystretch}{1.12}",
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"Term & Coef. & Std. Err. \\",
        r"\midrule",
    ]

    for row in table_df.itertuples(index=False):
        coef_display = f"{row.coef:.3f}"
        if bool(row.is_significant):
            coef_display += "*"
        lines.append(f"{row.term_label} & {coef_display} & {row.std_err:.3f} " + r"\\")

    lines.extend(
        [
            r"\midrule",
            rf"$R^2$ & \multicolumn{{2}}{{r}}{{{rsquared:.3f}}} \\",
            rf"$N$ & \multicolumn{{2}}{{r}}{{{nobs:,d}}} \\",
            r"\hline",
            r"\end{tabular}",
            (
                r"\caption{Compact summary of the pooled hiring regression. Reported terms show the demographic "
                r"main effects, post-training shift terms, and quality controls. Occupation and model-family "
                r"fixed effects are included in the pooled model but omitted from the table. Asterisks mark "
                r"coefficients with $p < 0.05$.}"
            ),
            r"\label{tab:appendix-pooled-regression-summary}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines)


def write_pooled_regression_latex_table(coefficients_df: pd.DataFrame, version: str) -> Path:
    tables_dir = Path("tables")
    tables_dir.mkdir(parents=True, exist_ok=True)
    output_path = tables_dir / f"appendix_pooled_regression_{version}.tex"
    output_path.write_text(build_pooled_regression_latex_table(coefficients_df) + "\n")
    return output_path


def _format_olmo_model_version(model: str) -> str:
    if model in OLMO_MODEL_VERSION_LABELS:
        return OLMO_MODEL_VERSION_LABELS[model]
    return model.replace("_", "-")


def prepare_olmo_bias_regression_table_df(coefficients_df: pd.DataFrame, version: str) -> pd.DataFrame:
    olmo_term_order = model_regression_term_order(version)
    olmo_term_labels = model_regression_term_labels(version)
    model_order = {model: idx for idx, model in enumerate(OLMO_MODEL_VERSION_ORDER)}
    term_order = {term: idx for idx, term in enumerate(olmo_term_order)}
    table_df = (
        coefficients_df.loc[
            coefficients_df["model"].str.contains("olmo", case=False, na=False)
            & (coefficients_df["specification"] == "full")
            & (coefficients_df["term"].isin(olmo_term_order))
        ].copy()
        .assign(
            model_label=lambda df: df["model"].map(_format_olmo_model_version),
            term_label=lambda df: df["term"].map(olmo_term_labels),
            is_significant=lambda df: df["p_value"] < 0.05,
            model_order=lambda df: df["model"].map(model_order).fillna(len(model_order)).astype(int),
            term_order=lambda df: df["term"].map(term_order).astype(int),
        )
        .sort_values(["model_order", "model", "term_order"])
        .reset_index(drop=True)
    )

    if table_df.empty:
        raise ValueError("No OLMo model-specific bias coefficients were found for the appendix table.")

    missing_terms = (
        table_df.groupby(["model", "model_label"])["term"]
        .nunique()
        .reset_index(name="n_terms")
        .loc[lambda df: df["n_terms"] != len(olmo_term_order)]
    )
    if not missing_terms.empty:
        raise ValueError(
            "OLMo model-specific bias table is missing one or more reported terms for: "
            + ", ".join(
                f"{row.model_label} ({int(row.n_terms)} terms)"
                for row in missing_terms.itertuples(index=False)
            )
        )

    return table_df


def build_olmo_bias_regression_latex_table(coefficients_df: pd.DataFrame, version: str) -> str:
    term_order = model_regression_term_order(version)
    term_labels = model_regression_term_labels(version)
    table_df = prepare_olmo_bias_regression_table_df(coefficients_df, version)
    model_df = (
        table_df.loc[:, ["model", "model_label", "model_order"]]
        .drop_duplicates()
        .sort_values(["model_order", "model"])
        .reset_index(drop=True)
    )

    column_format = "l" + ("r" * len(model_df))
    header = " & ".join(
        ["Variable", *[_latex_escape(str(row.model_label)) for row in model_df.itertuples(index=False)]]
    ) + r" \\"
    lines = [
        r"\begin{table}[p]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{5pt}",
        r"\renewcommand{\arraystretch}{1.12}",
        rf"\begin{{tabular}}{{{column_format}}}",
        r"\toprule",
        header,
        r"\midrule",
    ]

    for term in term_order:
        coef_values = [term_labels[term]]
        std_err_values = [""]
        for model in model_df["model"]:
            row = table_df.loc[(table_df["model"] == model) & (table_df["term"] == term)].iloc[0]
            coef_display = f"{row.coef:.3f}"
            if bool(row.is_significant):
                coef_display += "*"
            coef_values.append(coef_display)
            std_err_values.append(f"({row.std_err:.3f})")
        lines.append(" & ".join(coef_values) + r" \\")
        lines.append(" & ".join(std_err_values) + r" \\")

    lines.extend(
        [
            r"\hline",
            r"\end{tabular}",
            (
                r"\caption{Model-specific OLMo hiring-bias estimates from the full specification. "
                r"Reported terms show the demographic coefficients and quality-control coefficients "
                r"for each OLMo model version, with standard errors reported in parentheses below "
                r"each coefficient. Occupation fixed effects are included in each model-specific "
                r"regression but omitted from the table. Asterisks mark coefficients with $p < 0.05$.}"
            ),
            r"\label{tab:appendix-olmo-model-specific-bias}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines)


def write_olmo_bias_regression_latex_table(coefficients_df: pd.DataFrame, version: str) -> Path:
    tables_dir = Path("tables")
    tables_dir.mkdir(parents=True, exist_ok=True)
    output_path = tables_dir / f"appendix_olmo_model_bias_regressions_{version}.tex"
    output_path.write_text(build_olmo_bias_regression_latex_table(coefficients_df, version) + "\n")
    return output_path


def prepare_olmo_rsquared_df(coefficients_df: pd.DataFrame) -> pd.DataFrame:
    model_order = {model: idx for idx, model in enumerate(OLMO_MODEL_VERSION_ORDER)}
    spec_order = {spec: idx for idx, spec in enumerate(OLMO_RSQUARED_SPEC_ORDER)}
    plot_df = (
        coefficients_df.loc[
            coefficients_df["model"].isin(OLMO_MODEL_VERSION_ORDER)
            & (coefficients_df["term"] == "Intercept")
            & (coefficients_df["specification"].isin(OLMO_RSQUARED_SPEC_ORDER))
        ].copy()
        .assign(
            model_version=lambda df: df["model"].map(OLMO_MODEL_VERSION_SHORT_LABELS),
            model_order=lambda df: df["model"].map(model_order).astype(int),
            spec_label=lambda df: df["specification"].map(OLMO_RSQUARED_SPEC_LABELS),
            spec_order=lambda df: df["specification"].map(spec_order).astype(int),
            rsquared=lambda df: pd.to_numeric(df["rsquared"], errors="coerce"),
        )
        .dropna(subset=["rsquared"])
        .sort_values(["spec_order", "model_order"])
        .reset_index(drop=True)
    )

    if plot_df.empty:
        raise ValueError("No OLMo R-squared values were found for the appendix figure.")

    missing_specs = (
        plot_df.groupby(["model", "model_version"])["specification"]
        .nunique()
        .reset_index(name="n_specs")
        .loc[lambda df: df["n_specs"] != len(OLMO_RSQUARED_SPEC_ORDER)]
    )
    if not missing_specs.empty:
        raise ValueError(
            "OLMo R-squared figure is missing one or more specifications for: "
            + ", ".join(
                f"{row.model_version} ({int(row.n_specs)} specs)"
                for row in missing_specs.itertuples(index=False)
            )
        )

    return plot_df


def make_olmo_rsquared_figure(plot_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    colors = {
        "demographics_only": "#4C89D9",
        "similarity_only": "#2DBE73",
    }
    x_order = [
        OLMO_MODEL_VERSION_SHORT_LABELS[model]
        for model in OLMO_MODEL_VERSION_ORDER
        if model in set(plot_df["model"])
    ]

    for specification in OLMO_RSQUARED_SPEC_ORDER:
        spec_df = plot_df.loc[plot_df["specification"] == specification].copy()
        if spec_df.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=spec_df["model_version"],
                y=spec_df["rsquared"],
                mode="lines+markers",
                name=OLMO_RSQUARED_SPEC_LABELS[specification],
                line=dict(color=colors[specification], width=2.8),
                marker=dict(size=9, color=colors[specification]),
                hovertemplate=(
                    "Version: %{x}<br>"
                    f"{OLMO_RSQUARED_SPEC_LABELS[specification]}: "
                    "%{y:.1%}<extra></extra>"
                ),
            )
        )

    y_max = max(float(plot_df["rsquared"].max()) * 1.15, 0.05)
    fig.update_xaxes(
        title="Olmo Post-Training Checkpoint",
        categoryorder="array",
        categoryarray=x_order,
        showgrid=False,
        zeroline=False,
    )
    fig.update_yaxes(
        title="Explained Variance (R²)",
        range=[0, y_max],
        tickformat=".0%",
        showgrid=True,
        gridcolor="#D9DCE8",
        zeroline=False,
    )
    _base_layout(fig)
    fig.update_layout(
        title=None,
        legend=dict(
            orientation="v",
            x=0.98,
            y=0.47,
            xanchor="right",
            yanchor="middle",
            bgcolor="rgba(255,255,255,0)",
        ),
        width=720,
        height=380,
        margin=dict(l=80, r=30, t=5, b=70),
    )
    return fig


def prepare_name_signal_df() -> pd.DataFrame:
    name_df = (
        pd.read_csv("raw_data/audit_study_names.csv")
        [["race", "gender", "firstname", "lastname"]]
        .drop_duplicates()
        .assign(
            race=lambda df: df["race"].astype(str),
            gender=lambda df: df["gender"].astype(str),
            firstname=lambda df: df["firstname"].astype(str).str.strip(),
            lastname=lambda df: df["lastname"].astype(str).str.strip(),
        )
    )

    rows: list[dict[str, str]] = []
    for race, gender in [
        ("White", "Male"),
        ("White", "Female"),
        ("Black", "Male"),
        ("Black", "Female"),
    ]:
        group_df = name_df.loc[(name_df["race"] == race) & (name_df["gender"] == gender)].copy()
        rows.append(
            {
                "column_label": f"{race}-{gender}",
                "first_names": ", ".join(sorted(group_df["firstname"].unique().tolist())),
                "last_names": ", ".join(sorted(group_df["lastname"].unique().tolist())),
            }
        )

    return pd.DataFrame(rows)


def _load_generator_distribution_cache() -> tuple[dict[str, object], dict[str, object]]:
    load_dotenv()
    cache_path = cache_dir().expanduser().resolve()
    occ_path = cache_path / "occ_dist.pkl"
    vacancy_path = cache_path / "vacancy_dist.pkl"
    if not occ_path.exists():
        raise FileNotFoundError(f"Could not find cached occupation distributions at {occ_path}.")
    if not vacancy_path.exists():
        raise FileNotFoundError(f"Could not find cached vacancy distributions at {vacancy_path}.")

    with occ_path.open("rb") as f:
        occ_dist = pickle.load(f)
    with vacancy_path.open("rb") as f:
        vacancy_dist = pickle.load(f)

    return occ_dist, vacancy_dist


def _distribution_field(dist: object, key: str) -> object | None:
    if isinstance(dist, dict):
        return dist.get(key)

    try:
        if key in dist:  # type: ignore[operator]
            return dist[key]  # type: ignore[index]
    except Exception:
        pass

    attr_map = {
        "DEGREE_MAJORS": "degree_majors",
        "SKILLS": "skills",
        "TRANSITION_FROM": "transition_from",
        "TRANSITION_TO": "transition_to",
        "EXPERIENCES": "experiences",
        "COMPANY_NAME": "company_name",
        "TITLE": "title",
        "SALARY": "salary",
        "SOC_NAME": "soc_name",
        "COUNT": "count",
    }
    attr_name = attr_map.get(key, key.lower())
    return getattr(dist, attr_name, None)


def _choice_count(dist: object, key: str) -> int:
    value = _distribution_field(dist, key)
    if value is None:
        return 0
    if isinstance(value, dict):
        return len(value.get("CHOICES", []))
    choices = getattr(value, "choices", None)
    if choices is not None:
        return len(choices)
    try:
        return len(value["CHOICES"])  # type: ignore[index]
    except Exception:
        return 0


def _distribution_count(dist: object) -> int:
    value = _distribution_field(dist, "COUNT")
    if value is None:
        return 0
    return int(value)


def _truncate_label(value: str, max_length: int) -> str:
    if len(value) <= max_length:
        return value
    if max_length <= 3:
        return value[:max_length]
    return value[: max_length - 3].rstrip() + "..."


def prepare_occupation_generator_summary_df(results_df: pd.DataFrame) -> pd.DataFrame:
    occ_dist, vacancy_dist = _load_generator_distribution_cache()
    occupation_df = (
        results_df.loc[:, ["soc_code", "soc_name"]]
        .drop_duplicates()
        .sort_values(["soc_code", "soc_name"])
        .reset_index(drop=True)
    )

    rows: list[dict[str, int | str]] = []
    missing_socs: list[str] = []
    for row in occupation_df.itertuples(index=False):
        occ_entry = occ_dist.get(row.soc_code)
        vacancy_entry = vacancy_dist.get(row.soc_code)
        if occ_entry is None or vacancy_entry is None:
            missing_socs.append(str(row.soc_code))
            continue

        rows.append(
            {
                "soc_code": str(row.soc_code),
                "soc_name": str(row.soc_name),
                "n_unique_employers": _choice_count(vacancy_entry, "COMPANY_NAME"),
                "n_unique_vacancy_titles": _choice_count(vacancy_entry, "TITLE"),
                "n_unique_vacancy_skills": _choice_count(vacancy_entry, "SKILLS"),
                "n_vacancy_salaries": _distribution_count(vacancy_entry),
                "n_profile_titles": _choice_count(occ_entry, "EXPERIENCES"),
                "n_profile_degrees": _choice_count(occ_entry, "DEGREE_MAJORS"),
                "n_unique_profile_skills": _choice_count(occ_entry, "SKILLS"),
            }
        )

    if missing_socs:
        raise ValueError(
            "Cached generator distributions were missing one or more experiment occupations: "
            + ", ".join(sorted(missing_socs))
        )

    summary_df = pd.DataFrame(rows)
    if len(summary_df) != 76:
        raise ValueError(
            f"Expected 76 occupations in the experiment summary table, found {len(summary_df)}."
        )
    return summary_df


def build_occupation_generator_latex_table(summary_df: pd.DataFrame) -> str:
    header_row = (
        r"\shortstack[l]{SOC} & \shortstack[l]{Occupation} & \shortstack[l]{uniq.\\employers} & "
        r"\shortstack[l]{vacancy\\titles} & "
        r"\shortstack[l]{vacancy\\skills} & "
        r"\shortstack[l]{vacancy\\salaries} & "
        r"\shortstack[l]{profile\\titles} & "
        r"\shortstack[l]{profile\\degrees} & "
        r"\shortstack[l]{uniq. profile\\skills} \\"
    )
    caption = (
        r"\caption{Occupation-level support sizes from the cached vacancy and profile generators for the "
        r"76 occupations used in the hiring experiments. Counts reflect the saved generator objects. The "
        r"vacancy-salary column reports the cached vacancy count because the saved salary distribution "
        r"stores fitted log-normal parameters rather than a separate number of salary support points.}"
        r"\label{tab:appendix-occupation-generator-support}"
    )
    lines = [
        r"\begingroup",
        r"\fontsize{6.3}{7.2}\selectfont",
        r"\setlength{\tabcolsep}{2pt}",
        r"\renewcommand{\arraystretch}{1.02}",
        r"\begin{longtable}{p{0.9cm}p{3.4cm}p{1.0cm}p{1.05cm}p{1.1cm}p{1.05cm}p{1.0cm}p{1.0cm}p{1.1cm}}",
        r"\toprule",
        header_row,
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        header_row,
        r"\midrule",
        r"\endhead",
        r"\midrule",
        r"\multicolumn{9}{r}{Continued on next page} \\",
        r"\midrule",
        r"\endfoot",
        r"\bottomrule",
        caption + r"\\",
        r"\endlastfoot",
    ]

    for row in summary_df.itertuples(index=False):
        lines.append(
            " & ".join(
                    [
                        _latex_escape(str(row.soc_code)),
                        _latex_escape(_truncate_label(str(row.soc_name), 25)),
                        f"{int(row.n_unique_employers):d}",
                        f"{int(row.n_unique_vacancy_titles):d}",
                        f"{int(row.n_unique_vacancy_skills):d}",
                    f"{int(row.n_vacancy_salaries):d}",
                    f"{int(row.n_profile_titles):d}",
                    f"{int(row.n_profile_degrees):d}",
                    f"{int(row.n_unique_profile_skills):d}",
                ]
            )
            + r" \\"
        )

    lines.extend([r"\end{longtable}", r"\endgroup"])
    return "\n".join(lines)


def write_occupation_generator_latex_table(summary_df: pd.DataFrame, version: str) -> Path:
    tables_dir = Path("tables")
    tables_dir.mkdir(parents=True, exist_ok=True)
    output_path = tables_dir / f"appendix_occupation_generator_summary_{version}.tex"
    output_path.write_text(build_occupation_generator_latex_table(summary_df) + "\n")
    return output_path


def build_na_rate_latex_table(na_df: pd.DataFrame) -> str:
    column_format = "llrrr"
    lines = [
        r"\small",
        rf"\begin{{tabular}}{{{column_format}}}",
        r"\toprule",
        r"Status & Model & Trials & N/A rate & N/A responses \\",
        r"\midrule",
    ]

    previous_status = None
    for row in na_df.itertuples(index=False):
        status = str(row.status)
        if previous_status is not None and status != previous_status:
            lines.append(r"\midrule")

        lines.append(
            " & ".join(
                [
                    _latex_escape(status),
                    _latex_escape(row.model),
                    f"{int(row.n_trials):d}",
                    f"{row.na_rate * 100:.2f}\\%",
                    f"{int(row.na_count):d} / {int(row.total_responses):d}",
                ]
            )
            + r" \\"
        )
        previous_status = status

    lines.extend([r"\hline", r"\end{tabular}"])
    return "\n".join(lines)


def write_na_rate_latex_table(na_df: pd.DataFrame, version: str) -> Path:
    tables_dir = Path("tables")
    tables_dir.mkdir(parents=True, exist_ok=True)
    output_path = tables_dir / f"appendix_na_rates_{version}.tex"
    output_path.write_text(build_na_rate_latex_table(na_df) + "\n")
    return output_path


def build_name_signal_latex_table(name_signal_df: pd.DataFrame) -> str:
    ordered_labels = ["White-Male", "White-Female", "Black-Male", "Black-Female"]
    indexed_df = name_signal_df.set_index("column_label").loc[ordered_labels]
    lines = [
        r"\begin{table}[p]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\renewcommand{\arraystretch}{1.28}",
        r"\begin{tabular}{p{2cm}p{5cm}p{5cm}}",
        r"\toprule",
        r"Group & First name & Last name \\",
        r"\midrule",
    ]

    for column_label in ordered_labels:
        lines.append(
            " & ".join(
                [
                    _latex_escape(column_label),
                    _latex_escape(indexed_df.loc[column_label, "first_names"]),
                    _latex_escape(indexed_df.loc[column_label, "last_names"]),
                ]
            )
            + r" \\"
        )

    lines.extend([
        r"\hline",
        r"\end{tabular}",
        (
            r"\caption{First and last names used to signal race and gender in the experimental "
            r"perturbations. Each cell lists the unique names drawn from the source name dataset.}"
        ),
        r"\label{tab:appendix-name-signals}",
        r"\end{table}",
    ])
    return "\n".join(lines)


def write_name_signal_latex_table(name_signal_df: pd.DataFrame) -> Path:
    tables_dir = Path("tables")
    tables_dir.mkdir(parents=True, exist_ok=True)
    output_path = tables_dir / "appendix_name_signals.tex"
    output_path.write_text(build_name_signal_latex_table(name_signal_df) + "\n")
    return output_path


def prepare_age_callback_df(results_df: pd.DataFrame, age_df: pd.DataFrame) -> pd.DataFrame:
    plot_df = (
        exclude_presentation_models(results_df)
        .assign(
            status=lambda df: np.where(df["model"].str.endswith("_base"), "Base", "Post-train"),
            picked_perturbed=lambda df: pd.to_numeric(df["picked_perturbed"], errors="coerce"),
        )
        .merge(age_df, on=["experiment_id", "variant_idx"], how="left", validate="many_to_one")
    )

    if plot_df["age"].isna().any():
        missing = int(plot_df["age"].isna().sum())
        raise ValueError(f"Age join failed for {missing} result rows.")

    point_df = (
        plot_df.groupby(["status", "experiment_id", "variant_idx", "age"], as_index=False)
        .agg(callback_rate=("picked_perturbed", "mean"))
        .sort_values(["status", "age", "experiment_id", "variant_idx"])
        .reset_index(drop=True)
    )

    return point_df


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        raise ValueError(f"Expected a 6-digit hex color, got {hex_color!r}.")
    red = int(hex_color[0:2], 16)
    green = int(hex_color[2:4], 16)
    blue = int(hex_color[4:6], 16)
    return f"rgba({red}, {green}, {blue}, {alpha})"


def _age_bucket_start(age: int) -> int | None:
    if 21 <= age <= 35:
        return 21 + ((age - 21) // 5) * 5
    if 45 <= age <= 59:
        return 45 + ((age - 45) // 5) * 5
    return None


def _age_bucket_label(age_bucket_start: int) -> str:
    if age_bucket_start == 21:
        return "22-25"
    if age_bucket_start == 55:
        return "55-58"
    return f"{age_bucket_start}-{age_bucket_start + 4}"


def _age_bucket_mid(age_bucket_start: int) -> float:
    if age_bucket_start == 21:
        return 23.5
    if age_bucket_start == 55:
        return 56.5
    return float(age_bucket_start + 2)


def _format_rho_tick(value: float) -> str:
    return f"{value:.1f}".lstrip("0")


def make_age_callback_figure(point_df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(
        rows=1,
        cols=2,
        shared_yaxes=True,
        horizontal_spacing=0.1,
        subplot_titles=("Base", "Post-train"),
    )

    for col_idx, status in enumerate(STATUS_ORDER, start=1):
        status_df = point_df.loc[point_df["status"] == status].copy()
        status_df["age_bucket_start"] = status_df["age"].map(_age_bucket_start)
        status_df = status_df.loc[status_df["age_bucket_start"].notna()].copy()
        status_df["age_bucket_start"] = status_df["age_bucket_start"].astype(int)
        status_df["age_bucket_mid"] = status_df["age_bucket_start"].map(_age_bucket_mid)
        status_df["age_bucket_label"] = status_df["age_bucket_start"].map(_age_bucket_label)
        age_summary_df = (
            status_df.groupby(["age_bucket_start", "age_bucket_mid", "age_bucket_label"], as_index=False)
            .agg(
                callback_mean=("callback_rate", "mean"),
                callback_std=("callback_rate", "std"),
            )
            .sort_values("age_bucket_start")
            .assign(
                callback_std=lambda df: df["callback_std"].fillna(0.0),
                band_low=lambda df: np.clip(df["callback_mean"] - df["callback_std"], 0.0, 1.0),
                band_high=lambda df: np.clip(df["callback_mean"] + df["callback_std"], 0.0, 1.0),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=age_summary_df["age_bucket_mid"],
                y=age_summary_df["band_high"] * 100,
                mode="lines",
                showlegend=False,
                line=dict(color="rgba(0,0,0,0)", width=0),
                hoverinfo="skip",
            ),
            row=1,
            col=col_idx,
        )
        fig.add_trace(
            go.Scatter(
                x=age_summary_df["age_bucket_mid"],
                y=age_summary_df["band_low"] * 100,
                mode="lines",
                showlegend=False,
                line=dict(color="rgba(0,0,0,0)", width=0),
                fill="tonexty",
                fillcolor=_hex_to_rgba(STATUS_COLORS[status], 0.18),
                customdata=np.column_stack(
                    [
                        age_summary_df["age_bucket_label"],
                        age_summary_df["callback_mean"] * 100,
                        age_summary_df["callback_std"] * 100,
                    ]
                ),
                hovertemplate=(
                    "Age bucket: %{customdata[0]}<br>"
                    "Mean callback rate: %{customdata[1]:.1f}%<br>"
                    "Std. dev.: %{customdata[2]:.1f} pp<extra></extra>"
                ),
            ),
            row=1,
            col=col_idx,
        )
        fig.add_trace(
            go.Scatter(
                x=age_summary_df["age_bucket_mid"],
                y=age_summary_df["callback_mean"] * 100,
                mode="lines",
                showlegend=False,
                line=dict(color=STATUS_COLORS[status], width=2.4),
                customdata=age_summary_df["age_bucket_label"],
                hovertemplate="Age bucket: %{customdata}<br>Mean callback rate: %{y:.1f}%<extra></extra>",
            ),
            row=1,
            col=col_idx,
        )

        fig.update_xaxes(
            row=1,
            col=col_idx,
            title="Age bucket",
            tickmode="array",
            tickvals=age_summary_df["age_bucket_mid"].tolist(),
            ticktext=age_summary_df["age_bucket_label"].tolist(),
            showgrid=True,
            gridcolor="#D9DCE8",
            zeroline=False,
        )

    fig.update_yaxes(
        title="Callback rate",
        ticksuffix="%",
        range=[0, 100],
        matches="y",
        showgrid=True,
        gridcolor="#D9DCE8",
        zeroline=False,
    )
    _base_layout(fig)
    fig.update_layout(
        margin=dict(l=90, r=30, t=35, b=70),
        width=800,
        height=300,
    )
    return fig


def _prepare_systemic_status_frame(
    status_df: pd.DataFrame,
    index_cols: list[str],
) -> tuple[pd.DataFrame, list[str], int]:
    expected_models = int(status_df["model"].nunique())
    systemic_df = (
        status_df.pivot_table(
            index=index_cols,
            columns="model",
            values="picked_perturbed",
            aggfunc="mean",
        )
        .reset_index()
    )
    model_cols = [col for col in systemic_df.columns if col not in set(index_cols)]
    systemic_df = systemic_df.dropna(subset=model_cols).copy()
    systemic_df = systemic_df.loc[
        systemic_df[model_cols].notna().sum(axis=1) == expected_models
    ].copy()
    return systemic_df, model_cols, expected_models


def prepare_global_systemic_exclusion_grid(
    results_df: pd.DataFrame,
    rho_values: np.ndarray = RHO_VALUES,
) -> pd.DataFrame:
    plot_df = exclude_presentation_models(results_df).assign(
        status=lambda df: np.where(df["model"].str.endswith("_base"), "Base", "Post-train"),
        picked_perturbed=lambda df: pd.to_numeric(df["picked_perturbed"], errors="coerce"),
    )

    rows: list[dict[str, float | int | str]] = []
    for status in STATUS_ORDER:
        status_df = plot_df.loc[plot_df["status"] == status].copy()
        systemic_df, model_cols, expected_models = _prepare_systemic_status_frame(
            status_df,
            ["experiment_id", "variant_idx"],
        )

        model_values = systemic_df[model_cols].to_numpy(dtype=float)
        tau_values = np.arange(1, expected_models + 1)
        for tau in tau_values:
            for rho in rho_values:
                below_threshold_counts = (model_values < float(rho)).sum(axis=1)
                rows.append(
                    {
                        "status": status,
                        "rho": float(rho),
                        "tau": int(tau),
                        "rate": float((below_threshold_counts >= tau).mean()),
                        "n": int(len(systemic_df)),
                    }
                )

    return pd.DataFrame(rows)


def prepare_demographic_variant_range_grid(
    results_df: pd.DataFrame,
    rho_values: np.ndarray = RHO_VALUES,
) -> pd.DataFrame:
    plot_df = exclude_presentation_models(results_df).assign(
        status=lambda df: np.where(df["model"].str.endswith("_base"), "Base", "Post-train"),
        picked_perturbed=lambda df: pd.to_numeric(df["picked_perturbed"], errors="coerce"),
    )

    rows: list[dict[str, float | int | str]] = []
    for status in STATUS_ORDER:
        status_df = plot_df.loc[plot_df["status"] == status].copy()
        systemic_df, model_cols, expected_models = _prepare_systemic_status_frame(
            status_df,
            ["experiment_id", "variant_idx", "intersection_group"],
        )

        model_values = systemic_df[model_cols].to_numpy(dtype=float)
        tau_values = np.arange(1, expected_models + 1)
        for tau in tau_values:
            for rho in rho_values:
                systemic_excluded = ((model_values < float(rho)).sum(axis=1) >= tau).astype(float)
                group_rates = (
                    systemic_df.assign(systemic_excluded=systemic_excluded)
                    .groupby("intersection_group", as_index=False)["systemic_excluded"]
                    .mean()
                )
                rows.append(
                    {
                        "status": status,
                        "rho": float(rho),
                        "tau": int(tau),
                        "rate_range": float(
                            group_rates["systemic_excluded"].max() - group_rates["systemic_excluded"].min()
                        ),
                        "min_rate": float(group_rates["systemic_excluded"].min()),
                        "max_rate": float(group_rates["systemic_excluded"].max()),
                        "n_groups": int(len(group_rates)),
                        "n": int(len(systemic_df)),
                    }
                )

    return pd.DataFrame(rows)


def _make_two_panel_systemic_exclusion_heatmap_figure(
    grid_df: pd.DataFrame,
    value_col: str,
    colorbar_title: str,
    hover_label: str,
    colorscale: str | list = "Viridis",
) -> go.Figure:
    fig = make_subplots(
        rows=1,
        cols=2,
        shared_yaxes=True,
        horizontal_spacing=0.08,
        subplot_titles=("Base", "Post-train"),
    )

    zmax = float(grid_df[value_col].max() * 100)

    for col_idx, status in enumerate(STATUS_ORDER, start=1):
        status_df = grid_df.loc[grid_df["status"] == status].copy()
        matrix_df = (
            status_df.pivot(index="tau", columns="rho", values=value_col)
            .sort_index()
            .sort_index(axis=1)
        )

        x_values = [round(float(value), 2) for value in matrix_df.columns.to_list()]
        y_values = [int(value) for value in matrix_df.index.to_list()]
        z_values = matrix_df.to_numpy(dtype=float) * 100

        fig.add_trace(
            go.Heatmap(
                x=x_values,
                y=y_values,
                z=z_values,
                zmin=0,
                zmax=zmax,
                zsmooth=False,
                xgap=1,
                ygap=1,
                coloraxis="coloraxis",
                hovertemplate=(
                    "rho: %{x:.1f}<br>"
                    "tau: %{y}<br>"
                    f"{hover_label}: %{{z:.1f}}%<extra></extra>"
                ),
            ),
            row=1,
            col=col_idx,
        )
        fig.update_xaxes(
            row=1,
            col=col_idx,
            title="ρ",
            tickmode="array",
            tickvals=x_values,
            ticktext=[_format_rho_tick(value) for value in x_values],
            showgrid=False,
            zeroline=False,
        )

    fig.update_yaxes(
        title="τ",
        tickmode="array",
        tickvals=list(range(1, 11)),
        matches="y",
        showgrid=False,
        zeroline=False,
        row=1,
        col=1,
    )
    fig.update_yaxes(
        title="",
        tickmode="array",
        tickvals=list(range(1, 11)),
        matches="y",
        showgrid=False,
        zeroline=False,
        row=1,
        col=2,
    )
    _base_layout(fig)
    fig.update_layout(
        coloraxis=dict(
            cmin=0,
            cmax=zmax,
            colorscale=colorscale,
            colorbar=dict(title=colorbar_title, ticksuffix="%"),
        ),
        margin=dict(l=20, r=20, t=20, b=20),
        width=800,
        height=300,
    )
    return fig


def make_global_systemic_exclusion_heatmap_figure(grid_df: pd.DataFrame) -> go.Figure:
    return _make_two_panel_systemic_exclusion_heatmap_figure(
        grid_df,
        value_col="rate",
        colorbar_title="Rate",
        hover_label="Systemic exclusion",
        colorscale="Viridis",
    )


def make_demographic_variant_range_heatmap_figure(grid_df: pd.DataFrame) -> go.Figure:
    return _make_two_panel_systemic_exclusion_heatmap_figure(
        grid_df,
        value_col="rate_range",
        colorbar_title="Range",
        hover_label="Range across demographic variants",
        colorscale="Cividis",
    )


def appendix_systemic_exclusion_z(results_df: pd.DataFrame, preferred_z: int = 10) -> int:
    model_counts = (
        exclude_presentation_models(results_df)
        .assign(status=lambda df: np.where(df["model"].str.endswith("_base"), "Base", "Post-train"))
        .groupby("status")["model"]
        .nunique()
    )
    if model_counts.empty:
        raise ValueError("No models are available for systemic exclusion.")
    return min(preferred_z, int(model_counts.min()))


def print_appendix_summary(
    age_point_df: pd.DataFrame,
    grid_df: pd.DataFrame,
    variant_range_grid_df: pd.DataFrame,
    systemic_overall_df: pd.DataFrame,
    na_df: pd.DataFrame,
    name_signal_df: pd.DataFrame,
    regression_summary_df: pd.DataFrame,
    systemic_z: int,
) -> None:
    print("\n=== Appendix Age Figure Checks ===")
    summary = (
        age_point_df.groupby("status", as_index=False)
        .agg(
            n_points=("callback_rate", "size"),
            min_age=("age", "min"),
            max_age=("age", "max"),
            min_callback=("callback_rate", "min"),
            max_callback=("callback_rate", "max"),
        )
        .sort_values("status")
        .reset_index(drop=True)
    )
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n=== Appendix Heatmap Grid Checks ===")
    heatmap_summary = (
        grid_df.groupby("status", as_index=False)
        .agg(
            n_cells=("rate", "size"),
            min_rho=("rho", "min"),
            max_rho=("rho", "max"),
            min_tau=("tau", "min"),
            max_tau=("tau", "max"),
        )
        .sort_values("status")
        .reset_index(drop=True)
    )
    print(heatmap_summary.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    print("\n=== Appendix Variant-Range Heatmap Checks ===")
    variant_range_summary = (
        variant_range_grid_df.groupby("status", as_index=False)
        .agg(
            n_cells=("rate_range", "size"),
            min_range=("rate_range", "min"),
            max_range=("rate_range", "max"),
            min_rho=("rho", "min"),
            max_rho=("rho", "max"),
            min_tau=("tau", "min"),
            max_tau=("tau", "max"),
        )
        .sort_values("status")
        .reset_index(drop=True)
    )
    print(variant_range_summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    crosscheck_df = (
        grid_df.loc[
            (np.isclose(grid_df["rho"], SYSTEMIC_EXCLUSION_THRESHOLD))
            & (grid_df["tau"] == systemic_z)
        ]
        .merge(
            systemic_overall_df.loc[:, ["status", "estimate"]].rename(columns={"estimate": "stage5_estimate"}),
            on="status",
            how="left",
            validate="one_to_one",
        )
        .assign(
            appendix_estimate=lambda df: df["rate"],
            abs_diff=lambda df: (df["rate"] - df["stage5_estimate"]).abs(),
        )
        .loc[:, ["status", "appendix_estimate", "stage5_estimate", "abs_diff", "n"]]
        .sort_values("status")
        .reset_index(drop=True)
    )
    print(f"\n=== rho=0.5, tau={systemic_z} Cross-check ===")
    print(crosscheck_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    print("\n=== Appendix N/A Rate Table Checks ===")
    na_summary_df = (
        na_df.loc[:, ["status", "model", "n_trials", "na_count", "total_responses", "na_rate"]]
        .assign(status=lambda df: df["status"].astype(str))
        .sort_values(["status", "model"])
        .reset_index(drop=True)
    )
    print(na_summary_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    print("\n=== Appendix Name Signal Table Checks ===")
    name_summary_df = (
        name_signal_df.assign(
            n_first_names=lambda df: df["first_names"].str.split(", ").str.len(),
            n_last_names=lambda df: df["last_names"].str.split(", ").str.len(),
        )
        .loc[:, ["column_label", "n_first_names", "n_last_names"]]
        .reset_index(drop=True)
    )
    print(name_summary_df.to_string(index=False))

    print("\n=== Appendix Model Regression Summary ===")
    print(
        regression_summary_df.to_string(
            index=False,
            formatters={
                "mean_coef": lambda x: f"{x:.3f}",
                "min_coef": lambda x: f"{x:.3f}",
                "max_coef": lambda x: f"{x:.3f}",
            },
        )
    )


def print_olmo_regression_summary(
    coefficients_df: pd.DataFrame,
    olmo_rsquared_df: pd.DataFrame,
    version: str,
) -> None:
    print("\n=== Appendix Olmo Regression Table Values ===")
    term_order = model_regression_term_order(version)
    term_labels = model_regression_term_labels(version)
    olmo_table_df = prepare_olmo_bias_regression_table_df(coefficients_df, version)
    model_df = (
        olmo_table_df.loc[:, ["model", "model_label", "model_order"]]
        .drop_duplicates()
        .sort_values(["model_order", "model"])
        .reset_index(drop=True)
    )

    coefficient_rows: list[dict[str, str]] = []
    for term in term_order:
        row: dict[str, str] = {"variable": term_labels[term]}
        for model_row in model_df.itertuples(index=False):
            coef_row = olmo_table_df.loc[
                (olmo_table_df["model"] == model_row.model)
                & (olmo_table_df["term"] == term)
            ].iloc[0]
            coef_display = f"{coef_row.coef:.3f}"
            if bool(coef_row.is_significant):
                coef_display += "*"
            row[str(model_row.model_label)] = f"{coef_display} ({coef_row.std_err:.3f})"
        coefficient_rows.append(row)

    print(pd.DataFrame(coefficient_rows).to_string(index=False))

    print("\n=== Appendix Olmo R-squared Plot Values ===")
    rsquared_summary_df = (
        olmo_rsquared_df.pivot(
            index=["model_order", "model_version"],
            columns="specification",
            values="rsquared",
        )
        .reset_index()
        .sort_values("model_order")
        .rename(
            columns={
                "model_version": "checkpoint",
                "demographics_only": "demographics_r2",
                "similarity_only": "human_capital_controls_r2",
            }
        )
        .loc[:, ["checkpoint", "demographics_r2", "human_capital_controls_r2"]]
        .reset_index(drop=True)
    )
    print(rsquared_summary_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))


def main(version: str = VERSION) -> None:
    results_df = load_prepared_results(version)
    regression_coefficients_df = load_regression_coefficients(version)
    pooled_regression_coefficients_df = load_pooled_regression_coefficients(version)
    age_df = load_variant_ages(version)
    pairwise_df = prepare_appendix_pairwise_correlation_df(results_df)
    na_df = prepare_na_rate_df(results_df, version)
    name_signal_df = prepare_name_signal_df()
    occupation_generator_summary_df = prepare_occupation_generator_summary_df(results_df)
    na_table_path = write_na_rate_latex_table(na_df, version)
    name_signal_table_path = write_name_signal_latex_table(name_signal_df)
    occupation_generator_table_path = write_occupation_generator_latex_table(
        occupation_generator_summary_df,
        version,
    )
    regression_table_path = write_model_regression_latex_table(regression_coefficients_df, version)
    regression_interactions_table_path = write_model_interaction_regression_latex_table(
        regression_coefficients_df,
        version,
    )
    pooled_regression_table_path = write_pooled_regression_latex_table(
        pooled_regression_coefficients_df,
        version,
    )
    olmo_bias_regression_table_path = write_olmo_bias_regression_latex_table(
        regression_coefficients_df,
        version,
    )
    regression_summary_df = summarize_model_regression_patterns(regression_coefficients_df, version)

    olmo_rsquared_df = prepare_olmo_rsquared_df(regression_coefficients_df)
    olmo_rsquared_paths = save_plotly_figure(
        make_olmo_rsquared_figure(olmo_rsquared_df),
        f"appendix_olmo_rsquared_{version}",
    )
    age_point_df = prepare_age_callback_df(results_df, age_df)
    age_paths = save_plotly_figure(
        make_age_callback_figure(age_point_df),
        f"appendix_age_callback_by_status_{version}",
    )
    pairwise_heatmap_paths = save_plotly_figure(
        make_appendix_pairwise_correlation_heatmap_figure(pairwise_df),
        f"appendix_pairwise_quality_bias_correlation_heatmaps_{version}",
    )

    grid_df = prepare_global_systemic_exclusion_grid(results_df)
    heatmap_paths = save_plotly_figure(
        make_global_systemic_exclusion_heatmap_figure(grid_df),
        f"appendix_global_systemic_exclusion_heatmaps_{version}",
    )
    variant_range_grid_df = prepare_demographic_variant_range_grid(results_df)
    variant_range_paths = save_plotly_figure(
        make_demographic_variant_range_heatmap_figure(variant_range_grid_df),
        f"appendix_demographic_variant_systemic_exclusion_range_heatmaps_{version}",
    )

    systemic_z = appendix_systemic_exclusion_z(results_df)
    _, systemic_overall_df = prepare_systemic_exclusion_df(
        results_df,
        threshold=SYSTEMIC_EXCLUSION_THRESHOLD,
        z=systemic_z,
    )
    print_appendix_summary(
        age_point_df,
        grid_df,
        variant_range_grid_df,
        systemic_overall_df,
        na_df,
        name_signal_df,
        regression_summary_df,
        systemic_z,
    )
    print_olmo_regression_summary(regression_coefficients_df, olmo_rsquared_df, version)

    print(f"Wrote table to {na_table_path}")
    print(f"Wrote table to {name_signal_table_path}")
    print(f"Wrote table to {occupation_generator_table_path}")
    print(f"Wrote table to {regression_table_path}")
    print(f"Wrote table to {regression_interactions_table_path}")
    print(f"Wrote table to {pooled_regression_table_path}")
    print(f"Wrote table to {olmo_bias_regression_table_path}")
    for path in [*olmo_rsquared_paths, *age_paths, *pairwise_heatmap_paths, *heatmap_paths, *variant_range_paths]:
        print(f"Wrote figure to {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_version_argument(parser)
    args = parser.parse_args()
    main(version=args.version)

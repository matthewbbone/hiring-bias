from __future__ import annotations

import argparse
import importlib.util
from itertools import combinations
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde, pearsonr

sys.path.append(str(Path(__file__).parent.parent))

from pipeline.utils.model_filters import (
    OLMO_APPENDIX_ONLY_MODELS,
    exclude_olmo_appendix_only_models,
)
from pipeline.config import add_version_argument, is_secondary_version, resolve_results_path


TERM_LABELS = {
    "black": "Black",
    "female": "Female",
    "over40": "Over 40",
}
PAIRWISE_ROW_LABELS = {
    "demographic_deltas": "Bias",
    "experiment_means": "Residual",
}
INTERSECTION_GROUP_LABELS = {
    "black_female_old": "B|F|>40",
    "white_male_old": "W|M|>40",
    "white_female_old": "W|F|>40",
    "black_male_old": "B|M|>40",
    "white_male_young": "W|M|<40",
    "black_male_young": "B|M|<40",
    "black_female_young": "B|F|<40",
    "white_female_young": "W|F|<40",
}
INTERACTION_TERM_LABELS = {
    "black:post": "Black",
    "female:post": "Female",
    "over40:post": "Over 40",
}

TERM_ORDER = ["black", "female", "over40"]
PAIRWISE_ROW_ORDER = ["demographic_deltas", "experiment_means"]
INTERSECTION_GROUP_ORDER = [
    "black_female_old",
    "white_male_old",
    "white_female_old",
    "black_male_old",
    "white_male_young",
    "black_male_young",
    "black_female_young",
    "white_female_young",
]
INTERACTION_TERM_ORDER = ["black:post", "female:post", "over40:post"]
STATUS_ORDER = ["Base", "Post-train"]
STATUS_COLORS = {
    "Base": "#4C89D9",
    "Post-train": "#9C27B0",
}
STATUS_OFFSETS = {
    "Base": 0.14,
    "Post-train": -0.14,
}
PRESENTATION_EXCLUDED_MODELS = OLMO_APPENDIX_ONLY_MODELS
SYSTEMIC_EXCLUSION_THRESHOLD = 0.5
INTERSECTION_LEGEND_KEY = (
    "W=White, B=Black, M=Male, F=Female"
    "<br>&lt;40=Under 40, &gt;40=Over 40"
)
PLOTLY_CONFIG = {"displayModeBar": False}
OUTPUT_DATA_DIR = Path("output_data")


def coefficient_display_df(coefficients_df: pd.DataFrame, version: str) -> pd.DataFrame:
    display_df = coefficients_df.copy()
    if not is_secondary_version(version):
        return display_df

    if "coef" in display_df:
        display_df["coef"] = pd.to_numeric(display_df["coef"], errors="coerce") * 10
        if "term" in display_df:
            display_df.loc[display_df["term"] == "Intercept", "coef"] += 1
    if "std_err" in display_df:
        display_df["std_err"] = pd.to_numeric(display_df["std_err"], errors="coerce") * 10
    return display_df


def coefficient_tickformat(version: str) -> str:
    return ".2f" if is_secondary_version(version) else ".0%"


def coefficient_axis_title(default_title: str, version: str) -> str:
    if is_secondary_version(version):
        return default_title.replace("(%)", "(rating points, 1-10 scale)")
    return default_title


def coefficient_hover_template(version: str) -> str:
    if is_secondary_version(version):
        return (
            "<b>%{customdata[0]}</b><br>"
            "Family: %{customdata[1]}<br>"
            "Coefficient: %{x:.2f}<br>"
            "95% CI: [%{customdata[2]:.2f}, %{customdata[3]:.2f}]<extra></extra>"
        )
    return (
        "<b>%{customdata[0]}</b><br>"
        "Family: %{customdata[1]}<br>"
        "Coefficient: %{x:.1%}<br>"
        "95% CI: [%{customdata[2]:.1%}, %{customdata[3]:.1%}]<extra></extra>"
    )


def interaction_hover_template(version: str) -> str:
    if is_secondary_version(version):
        return (
            "<b>%{customdata[0]}</b><br>"
            "Interaction effect: %{x:.2f}<br>"
            "SE: %{customdata[1]:.2f}<br>"
            "95% CI: [%{customdata[2]:.2f}, %{customdata[3]:.2f}]<extra></extra>"
        )
    return (
        "<b>%{customdata[0]}</b><br>"
        "Interaction effect: %{x:.1%}<br>"
        "SE: %{customdata[1]:.1%}<br>"
        "95% CI: [%{customdata[2]:.1%}, %{customdata[3]:.1%}]<extra></extra>"
    )


def resolve_csv_path(filename: str) -> Path:
    output_path = OUTPUT_DATA_DIR / filename
    return output_path if output_path.exists() else Path(filename)


def load_regression_coefficients(version: str) -> pd.DataFrame:
    path = resolve_csv_path(f"regression_coefficients_{version}.csv")
    if not path.exists():
        raise FileNotFoundError(f"Could not find {path}. Run stage 4 first.")
    return coefficient_display_df(pd.read_csv(path), version)


def load_pooled_interactions(version: str) -> pd.DataFrame:
    path = resolve_csv_path(f"pooled_post_interactions_{version}.csv")
    if not path.exists():
        raise FileNotFoundError(f"Could not find {path}. Run stage 4 first.")
    return coefficient_display_df(pd.read_csv(path), version)


def load_regression_soc_coefficients(version: str) -> pd.DataFrame:
    path = resolve_csv_path(f"regression_soc_coefficients_{version}.csv")
    if not path.exists():
        raise FileNotFoundError(f"Could not find {path}. Run stage 4 first.")
    return pd.read_csv(path)


def load_prepared_results(version: str) -> pd.DataFrame:
    path = resolve_results_path(version, OUTPUT_DATA_DIR)
    if not path.exists():
        raise FileNotFoundError(f"Could not find {path}. Run stage 3 first.")
    return pd.read_csv(path)


def save_plotly_figure(fig: go.Figure, stem: str) -> list[Path]:
    figures_dir = Path("figures")
    figures_dir.mkdir(parents=True, exist_ok=True)
    output_paths = []

    html_path = figures_dir / f"{stem}.html"
    fig.write_html(
        html_path,
        config=PLOTLY_CONFIG,
        include_plotlyjs="cdn",
        div_id=stem,
    )
    output_paths.append(html_path)

    if importlib.util.find_spec("kaleido") is None:
        return output_paths

    pdf_path = figures_dir / f"{stem}.pdf"
    fig.write_image(pdf_path, scale=2, format="pdf")
    output_paths.append(pdf_path)
    return output_paths


def _base_layout(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        font=dict(size=14, color="#324B72"),
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=90, r=30, t=70, b=70),
    )
    return fig


def exclude_presentation_models(df: pd.DataFrame) -> pd.DataFrame:
    return exclude_olmo_appendix_only_models(df)


def prepare_plot_df(coefficients_df: pd.DataFrame) -> pd.DataFrame:
    coefficients_df = exclude_presentation_models(coefficients_df)
    plot_df = (
        coefficients_df.loc[
            (coefficients_df["specification"] == "full")
            & (coefficients_df["term"].isin(TERM_ORDER))
        ].copy()
        .assign(
            term_label=lambda df: df["term"].map(TERM_LABELS),
            status=lambda df: np.where(df["model"].str.endswith("_base"), "Base", "Post-train"),
            family=lambda df: df["model"].str.replace(r"_(base|post)$", "", regex=True),
            ci_low=lambda df: df["coef"] - 1.96 * df["std_err"],
            ci_high=lambda df: df["coef"] + 1.96 * df["std_err"],
        )
    )

    if plot_df.empty:
        raise ValueError("No full-specification regression coefficients were found for the callback bias figure.")

    base_y = {term: len(TERM_ORDER) - 1 - idx for idx, term in enumerate(TERM_ORDER)}
    plot_df["y_base"] = plot_df["term"].map(base_y).astype(float)
    plot_df["y"] = plot_df["y_base"] + plot_df["status"].map(STATUS_OFFSETS)

    jitter_values = pd.Series(0.0, index=plot_df.index)
    for _, group in plot_df.groupby(["term", "status"], sort=False):
        n = len(group)
        jitter = np.array([0.0]) if n == 1 else np.linspace(-0.045, 0.045, n)
        jitter_values.loc[group.index] = jitter
    plot_df["y"] = plot_df["y"] + jitter_values
    return plot_df.sort_values(["term", "status", "family"]).reset_index(drop=True)


def make_bias_coefficient_figure(plot_df: pd.DataFrame, version: str = "") -> go.Figure:
    fig = go.Figure()

    for status in STATUS_ORDER:
        status_df = plot_df.loc[plot_df["status"] == status]
        fig.add_trace(
            go.Scatter(
                x=status_df["coef"],
                y=status_df["y"],
                mode="markers",
                name=status,
                legendgroup=status,
                marker=dict(color=STATUS_COLORS[status], size=9, opacity=0.85),
                error_x=dict(
                    type="data",
                    symmetric=False,
                    array=(status_df["ci_high"] - status_df["coef"]),
                    arrayminus=(status_df["coef"] - status_df["ci_low"]),
                    color=STATUS_COLORS[status],
                    thickness=1.5,
                    width=0,
                ),
                hovertemplate=coefficient_hover_template(version),
                customdata=np.column_stack(
                    [
                        status_df["term_label"],
                        status_df["family"],
                        status_df["ci_low"],
                        status_df["ci_high"],
                    ]
                ),
            )
        )

    fig.add_vline(x=0.0, line_color="red", line_dash="dash", line_width=2)
    fig.update_yaxes(
        tickmode="array",
        tickvals=[2, 1, 0],
        ticktext=[TERM_LABELS[term] for term in TERM_ORDER],
        showgrid=False,
        zeroline=False,
    )
    fig.update_xaxes(
        title=coefficient_axis_title("Estimated coefficient (%)", version),
        tickformat=coefficient_tickformat(version),
        showgrid=True,
        gridcolor="#D9DCE8",
        zeroline=False,
    )
    _base_layout(fig)
    fig.update_layout(
        title=dict(text="Hiring Task", x=0.5, xanchor="center", font=dict(size=24)),
        legend=dict(x=0.98, y=0.98, xanchor="right", yanchor="top"),
    )
    return fig


def prepare_pooled_interaction_df(interactions_df: pd.DataFrame) -> pd.DataFrame:
    plot_df = (
        interactions_df.loc[interactions_df["term"].isin(INTERACTION_TERM_ORDER)].copy()
        .assign(
            term_label=lambda df: df["term"].map(INTERACTION_TERM_LABELS),
            se_low=lambda df: df["coef"] - df["std_err"],
            se_high=lambda df: df["coef"] + df["std_err"],
            ci_low=lambda df: df["coef"] - 1.96 * df["std_err"],
            ci_high=lambda df: df["coef"] + 1.96 * df["std_err"],
            color=lambda df: np.where(df["coef"] >= 0, "#2DBE73", "#E74C3C"),
        )
    )
    base_y = {term: len(INTERACTION_TERM_ORDER) - 1 - idx for idx, term in enumerate(INTERACTION_TERM_ORDER)}
    plot_df["y"] = plot_df["term"].map(base_y).astype(float)
    return plot_df.sort_values("y", ascending=False).reset_index(drop=True)


def make_pooled_interaction_figure(plot_df: pd.DataFrame, version: str = "") -> go.Figure:
    x_min = float(min(plot_df["ci_low"].min(), 0.0))
    x_max = float(max(plot_df["ci_high"].max(), 0.0))
    x_pad = max((x_max - x_min) * 0.06, 0.01)
    x_left = x_min - x_pad
    x_right = x_max + x_pad

    fig = go.Figure()
    fig.add_vrect(x0=x_left, x1=0.0, fillcolor="#F6E8E5", opacity=0.8, line_width=0, layer="below")
    fig.add_vrect(x0=0.0, x1=x_right, fillcolor="#E6F3EE", opacity=0.8, line_width=0, layer="below")
    fig.add_vline(x=0.0, line_color="black", line_width=2)

    fig.add_trace(
        go.Scatter(
            x=plot_df["coef"],
            y=plot_df["y"],
            mode="markers",
            showlegend=False,
            marker=dict(
                color=plot_df["color"],
                size=18,
                line=dict(color="white", width=2),
            ),
            error_x=dict(
                type="data",
                symmetric=False,
                array=(plot_df["ci_high"] - plot_df["coef"]),
                arrayminus=(plot_df["coef"] - plot_df["ci_low"]),
                color="#A8ADA7",
                thickness=2.4,
                width=8,
            ),
            hovertemplate=interaction_hover_template(version),
            customdata=np.column_stack(
                [plot_df["term_label"], plot_df["std_err"], plot_df["ci_low"], plot_df["ci_high"]]
            ),
        )
    )

    fig.update_yaxes(
        tickmode="array",
        tickvals=[2, 1, 0],
        ticktext=[INTERACTION_TERM_LABELS[term] for term in INTERACTION_TERM_ORDER],
        showgrid=False,
        zeroline=False,
    )
    fig.update_xaxes(
        title=coefficient_axis_title("Interaction effect of post-training (%)", version),
        range=[x_left, x_right],
        tickformat=coefficient_tickformat(version),
        showgrid=True,
        gridcolor="#C9CFC8",
        zeroline=False,
    )
    _base_layout(fig)
    fig.update_layout(title=dict(text="Hiring Task", x=0.5, xanchor="center", font=dict(size=26)))
    return fig


def make_combined_bias_interaction_figure(
    bias_plot_df: pd.DataFrame,
    interaction_plot_df: pd.DataFrame,
    version: str = "",
) -> go.Figure:
    bias_title = "Rating Biases" if is_secondary_version(version) else "Callback Biases"
    fig = make_subplots(
        rows=1,
        cols=2,
        shared_yaxes=True,
        horizontal_spacing=0.08,
        subplot_titles=(f"A. {bias_title} (β)", "B. Post-training Bias Shifts (λ)"),
    )

    for status in STATUS_ORDER:
        status_df = bias_plot_df.loc[bias_plot_df["status"] == status]
        fig.add_trace(
            go.Scatter(
                x=status_df["coef"],
                y=status_df["y"],
                mode="markers",
                name=status,
                legendgroup=status,
                marker=dict(color=STATUS_COLORS[status], size=9, opacity=0.85),
                error_x=dict(
                    type="data",
                    symmetric=False,
                    array=(status_df["ci_high"] - status_df["coef"]),
                    arrayminus=(status_df["coef"] - status_df["ci_low"]),
                    color=STATUS_COLORS[status],
                    thickness=1.5,
                    width=0,
                ),
                hovertemplate=coefficient_hover_template(version),
                customdata=np.column_stack(
                    [
                        status_df["term_label"],
                        status_df["family"],
                        status_df["ci_low"],
                        status_df["ci_high"],
                    ]
                ),
            ),
            row=1,
            col=1,
        )

    fig.update_yaxes(
        row=1,
        col=1,
        tickmode="array",
        tickvals=[2, 1, 0],
        ticktext=[TERM_LABELS[term] for term in TERM_ORDER],
        showgrid=False,
        zeroline=False,
        showline=True,
        linecolor="black",
        linewidth=1,
        mirror=False,
    )
    fig.update_xaxes(
        row=1,
        col=1,
        title=None,
        tickformat=coefficient_tickformat(version),
        showgrid=True,
        gridcolor="#D9DCE8",
        zeroline=True,
        zerolinecolor="black",
        zerolinewidth=2.5,
        showline=True,
        linecolor="black",
        linewidth=1,
        mirror=False,
    )

    x_min = float(min(interaction_plot_df["ci_low"].min(), 0.0))
    x_max = float(max(interaction_plot_df["ci_high"].max(), 0.0))
    x_pad = max((x_max - x_min) * 0.06, 0.01)
    x_left = x_min - x_pad
    x_right = x_max + x_pad
    fig.add_trace(
        go.Scatter(
            x=interaction_plot_df["coef"],
            y=interaction_plot_df["y"],
            mode="markers",
            showlegend=False,
            marker=dict(
                color=interaction_plot_df["color"],
                size=18,
                line=dict(color="white", width=2),
            ),
            error_x=dict(
                type="data",
                symmetric=False,
                array=(interaction_plot_df["ci_high"] - interaction_plot_df["coef"]),
                arrayminus=(interaction_plot_df["coef"] - interaction_plot_df["ci_low"]),
                color="#A8ADA7",
                thickness=2.4,
                width=8,
            ),
            hovertemplate=interaction_hover_template(version),
            customdata=np.column_stack(
                [
                    interaction_plot_df["term_label"],
                    interaction_plot_df["std_err"],
                    interaction_plot_df["ci_low"],
                    interaction_plot_df["ci_high"],
                ]
            ),
        ),
        row=1,
        col=2,
    )
    fig.update_yaxes(
        row=1,
        col=2,
        tickmode="array",
        tickvals=[2, 1, 0],
        ticktext=[INTERACTION_TERM_LABELS[term] for term in INTERACTION_TERM_ORDER],
        showgrid=False,
        zeroline=False,
        showticklabels=False,
        showline=True,
        linecolor="black",
        linewidth=1,
        mirror=False,
    )
    fig.update_xaxes(
        row=1,
        col=2,
        title=None,
        range=[x_left, x_right],
        tickformat=coefficient_tickformat(version),
        showgrid=True,
        gridcolor="#D9DCE8",
        zeroline=True,
        zerolinecolor="black",
        zerolinewidth=2.5,
        showline=True,
        linecolor="black",
        linewidth=1,
        mirror=False,
    )

    _base_layout(fig)
    fig.update_layout(
        legend=dict(
            orientation="v",
            x=0.45,
            y=.05,
            xanchor="right",
            yanchor="bottom",
            bgcolor="rgba(255, 255, 255, 0.7)",
        ),
        margin=dict(l=20, r=20, t=20, b=20),
        width=800,
        height=300,
    )
    return fig


def _prepare_pairwise_correlations(
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
                    "term_label": PAIRWISE_ROW_LABELS[term],
                    "status": status,
                    "model_a": model_a,
                    "model_b": model_b,
                    "pearson_r": float(corr.statistic),
                    "p_value": float(corr.pvalue),
                    "n_obs": int(len(merged)),
                }
            )
    return rows


def prepare_pairwise_correlation_df(results_df: pd.DataFrame) -> pd.DataFrame:
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
        _prepare_pairwise_correlations(
            delta_df,
            index_cols=["experiment_id", "variant_idx"],
            value_col="demographic_delta",
            term="demographic_deltas",
        )
    )
    rows.extend(
        _prepare_pairwise_correlations(
            experiment_mean_df,
            index_cols=["experiment_id"],
            value_col="experiment_mean",
            term="experiment_means",
        )
    )
    return pd.DataFrame(rows)


def _density_curve(values: np.ndarray, x_grid: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) < 2 or np.allclose(values, values[0]):
        return np.zeros_like(x_grid)
    kde = gaussian_kde(values)
    return kde(x_grid)


def _model_family(model_name: str) -> str:
    return model_name.rsplit("_", 1)[0]


def _paired_family_pair_id(frame: pd.DataFrame) -> pd.Series:
    family_a = frame["model_a"].map(_model_family)
    family_b = frame["model_b"].map(_model_family)
    return pd.Series(
        np.where(
            family_a <= family_b,
            family_a + "||" + family_b,
            family_b + "||" + family_a,
        ),
        index=frame.index,
    )


def paired_mean_correlation_shift(term_df: pd.DataFrame) -> dict[str, float | int] | None:
    base_df = (
        term_df.loc[term_df["status"] == "Base", ["model_a", "model_b", "pearson_r"]]
        .assign(pair_id=lambda df: _paired_family_pair_id(df))
        .rename(columns={"pearson_r": "pearson_r_base"})
    )
    post_df = (
        term_df.loc[term_df["status"] == "Post-train", ["model_a", "model_b", "pearson_r"]]
        .assign(pair_id=lambda df: _paired_family_pair_id(df))
        .rename(columns={"pearson_r": "pearson_r_post"})
    )

    paired_df = (
        base_df[["pair_id", "pearson_r_base"]]
        .merge(post_df[["pair_id", "pearson_r_post"]], on="pair_id", how="inner")
        .dropna()
    )
    if paired_df.empty:
        return None

    return {
        "n_pairs": int(len(paired_df)),
        "base_mean_r": float(paired_df["pearson_r_base"].mean()),
        "post_mean_r": float(paired_df["pearson_r_post"].mean()),
        "delta_r": float((paired_df["pearson_r_post"] - paired_df["pearson_r_base"]).mean()),
    }


def make_pairwise_correlation_figure(pairwise_df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(
        rows=len(PAIRWISE_ROW_ORDER),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
    )

    bins = np.linspace(-1, 1, 17)
    x_grid = np.linspace(-1, 1, 300)

    for row_idx, term in enumerate(PAIRWISE_ROW_ORDER, start=1):
        term_df = pairwise_df.loc[pairwise_df["term"] == term]
        for status in STATUS_ORDER:
            status_df = term_df.loc[term_df["status"] == status]
            if status_df.empty:
                continue

            fig.add_trace(
                go.Histogram(
                    x=status_df["pearson_r"],
                    xbins=dict(start=-1, end=1, size=(bins[1] - bins[0])),
                    histnorm="probability density",
                    marker=dict(color=STATUS_COLORS[status], line=dict(color=STATUS_COLORS[status], width=1)),
                    opacity=0.22,
                    name=status,
                    legendgroup=status,
                    showlegend=(row_idx == 1),
                ),
                row=row_idx,
                col=1,
            )

            density = _density_curve(status_df["pearson_r"].to_numpy(), x_grid)
            fig.add_trace(
                go.Scatter(
                    x=x_grid,
                    y=density,
                    mode="lines",
                    line=dict(color=STATUS_COLORS[status], width=3),
                    legendgroup=status,
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=row_idx,
                col=1,
            )

        shift_result = paired_mean_correlation_shift(term_df)
        if shift_result is not None:
            annotation = f"\u0394r\u0304 = {float(shift_result['delta_r']):.3f}"
            y_max = 0.0
            for status in STATUS_ORDER:
                status_vals = term_df.loc[term_df["status"] == status, "pearson_r"].to_numpy()
                density = _density_curve(status_vals, x_grid)
                if len(density):
                    y_max = max(y_max, float(np.nanmax(density)))
            axis_suffix = "" if row_idx == 1 else str(row_idx + 1)
            fig.add_annotation(
                x=0.04,
                y=0.86,
                xref=f"x{axis_suffix} domain",
                yref=f"y{axis_suffix} domain",
                text=annotation,
                showarrow=False,
                align="left",
                font=dict(size=11, color="#324B72"),
            )

        fig.add_vline(x=0, row=row_idx, col=1, line_color="#B8B8B8", line_width=1.5)
        fig.update_yaxes(title_text=PAIRWISE_ROW_LABELS[term], row=row_idx, col=1, showticklabels=False)

    fig.update_xaxes(range=[-1, 1], title_text="Pairwise Pearson's r", row=len(PAIRWISE_ROW_ORDER), col=1)
    _base_layout(fig)
    fig.update_layout(
        title=dict(text="Hiring Task", x=0.5, xanchor="center", font=dict(size=28)),
        barmode="overlay",
        height=360,
        legend=dict(
            orientation="h",
            x=0.5,
            y=1.08,
            xanchor="center",
            yanchor="bottom",
            font=dict(size=16),
        ),
    )
    return fig


def _bootstrap_exclusion_rate(
    values: np.ndarray,
    rng: np.random.Generator,
    n_boot: int = 10_000,
) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")

    point_estimate = float(values.mean())
    if np.isin(values, [0.0, 1.0]).all():
        boot_means = rng.binomial(len(values), point_estimate, size=n_boot) / len(values)
    else:
        boot_means = np.empty(n_boot, dtype=float)
        chunk_size = max(1, min(1_000, n_boot))
        for start in range(0, n_boot, chunk_size):
            stop = min(start + chunk_size, n_boot)
            sample_idx = rng.integers(0, len(values), size=(stop - start, len(values)))
            boot_means[start:stop] = values[sample_idx].mean(axis=1)

    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])
    return point_estimate, float(ci_low), float(ci_high)


def _resolve_systemic_exclusion_z(z: int | None, n_models: int) -> int:
    if z is None:
        return n_models
    if z < 1:
        raise ValueError("z must be at least 1.")
    if z > n_models:
        raise ValueError(f"z={z} exceeds the number of available models ({n_models}).")
    return z


def _systemic_exclusion_axis_title(z: int, threshold: float) -> str:
    model_label = "model" if z == 1 else "models"
    return (
        "Systemic exclusion "
        f"(% of experiment rows where >= {z} {model_label} have picked_perturbed < {threshold:g})"
    )


def prepare_systemic_exclusion_df(
    results_df: pd.DataFrame,
    n_boot: int = 10_000,
    seed: int = 42,
    threshold: float = SYSTEMIC_EXCLUSION_THRESHOLD,
    z: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    plot_df = exclude_presentation_models(results_df).assign(
        status=lambda df: np.where(df["model"].str.endswith("_base"), "Base", "Post-train"),
        picked_perturbed=lambda df: pd.to_numeric(df["picked_perturbed"], errors="coerce"),
    )

    rng = np.random.default_rng(seed)
    group_rows: list[dict[str, float | str | int]] = []
    overall_rows: list[dict[str, float | str | int]] = []

    for status in STATUS_ORDER:
        status_df = plot_df.loc[plot_df["status"] == status].copy()
        expected_models = status_df["model"].dropna().nunique()
        z_for_status = _resolve_systemic_exclusion_z(z, expected_models)

        systemic_df = (
            status_df.pivot_table(
                index=["experiment_id", "variant_idx", "intersection_group"],
                columns="model",
                values="picked_perturbed",
                aggfunc="mean",
            )
            .reset_index()
        )
        model_cols = [
            col
            for col in systemic_df.columns
            if col not in {"experiment_id", "variant_idx", "intersection_group"}
        ]
        systemic_df = systemic_df.dropna(subset=model_cols).copy()
        systemic_df = systemic_df.loc[
            systemic_df[model_cols].notna().sum(axis=1) == expected_models
        ].copy()
        systemic_df["systemic_excluded"] = (
            systemic_df[model_cols].to_numpy() < threshold
        ).sum(axis=1) >= z_for_status
        systemic_df["systemic_excluded"] = systemic_df["systemic_excluded"].astype(float)

        estimate, ci_low, ci_high = _bootstrap_exclusion_rate(
            systemic_df["systemic_excluded"].to_numpy(),
            rng=rng,
            n_boot=n_boot,
        )
        overall_rows.append(
            {
                "status": status,
                "estimate": estimate,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "n": int(len(systemic_df)),
                "z": z_for_status,
                "threshold": float(threshold),
            }
        )

        for group in INTERSECTION_GROUP_ORDER:
            subset = systemic_df.loc[systemic_df["intersection_group"] == group, "systemic_excluded"]
            estimate, ci_low, ci_high = _bootstrap_exclusion_rate(
                subset.to_numpy(),
                rng=rng,
                n_boot=n_boot,
            )
            group_rows.append(
                {
                    "status": status,
                    "intersection_group": group,
                    "label": INTERSECTION_GROUP_LABELS[group],
                    "estimate": estimate,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "n": int(subset.notna().sum()),
                    "z": z_for_status,
                    "threshold": float(threshold),
                }
            )

    return pd.DataFrame(group_rows), pd.DataFrame(overall_rows)


def make_systemic_exclusion_figure(
    group_df: pd.DataFrame,
    overall_df: pd.DataFrame,
    z: int | None = None,
    threshold: float = SYSTEMIC_EXCLUSION_THRESHOLD,
) -> go.Figure:
    plot_order = (
        group_df.loc[group_df["status"] == "Post-train"]
        .sort_values("estimate", ascending=False)["intersection_group"]
        .tolist()
    )
    if not plot_order:
        plot_order = INTERSECTION_GROUP_ORDER

    group_df = group_df.copy()
    base_y = {group: len(plot_order) - 1 - idx for idx, group in enumerate(plot_order)}
    group_df["y_base"] = group_df["intersection_group"].map(base_y).astype(float)
    group_df["y"] = group_df["y_base"] + group_df["status"].map(STATUS_OFFSETS)

    fig = go.Figure()
    for status in STATUS_ORDER:
        row = overall_df.loc[overall_df["status"] == status].iloc[0]
        fig.add_vrect(
            x0=row["ci_low"] * 100,
            x1=row["ci_high"] * 100,
            fillcolor=STATUS_COLORS[status],
            opacity=0.14,
            line_width=0,
            layer="below",
        )

    for status in STATUS_ORDER:
        status_df = group_df.loc[group_df["status"] == status]
        fig.add_trace(
            go.Scatter(
                x=status_df["estimate"] * 100,
                y=status_df["y"],
                mode="markers",
                name=status,
                legendgroup=status,
                marker=dict(
                    color=STATUS_COLORS[status],
                    size=10,
                    symbol="circle" if status == "Base" else "diamond",
                ),
                error_x=dict(
                    type="data",
                    symmetric=False,
                    array=(status_df["ci_high"] - status_df["estimate"]) * 100,
                    arrayminus=(status_df["estimate"] - status_df["ci_low"]) * 100,
                    color=STATUS_COLORS[status],
                    thickness=2,
                    width=4,
                ),
                hovertemplate=(
                    "<b>%{customdata[0]} (%{customdata[1]})</b><br>"
                    "Systemic exclusion: %{x:.1f}%<br>"
                    "95% CI: [%{customdata[2]:.1f}%%, %{customdata[3]:.1f}%%]<br>"
                    "n = %{customdata[4]}<extra></extra>"
                ),
                customdata=np.column_stack(
                    [
                        status_df["label"],
                        status_df["status"],
                        status_df["ci_low"] * 100,
                        status_df["ci_high"] * 100,
                        status_df["n"],
                    ]
                ),
            )
        )

    fig.update_yaxes(
        tickmode="array",
        tickvals=list(range(len(plot_order) - 1, -1, -1)),
        ticktext=[INTERSECTION_GROUP_LABELS[group] for group in plot_order],
        title="Intersection (Race|Gender|Age)",
        showgrid=True,
        gridcolor="#D9DCE8",
        zeroline=False,
    )
    title_z = int(overall_df["z"].iloc[0]) if "z" in overall_df else z
    if title_z is None:
        raise ValueError("z must be provided or present in overall_df.")
    title_threshold = (
        float(overall_df["threshold"].iloc[0])
        if "threshold" in overall_df
        else threshold
    )
    fig.update_xaxes(
        title=_systemic_exclusion_axis_title(title_z, title_threshold),
        ticksuffix="%",
        showgrid=True,
        gridcolor="#D9DCE8",
        zeroline=False,
    )
    _base_layout(fig)
    fig.update_layout(
        title=dict(text="Hiring Task", x=0.5, xanchor="center", font=dict(size=24)),
        legend=dict(
            orientation="h",
            x=0.98,
            y=0.02,
            xanchor="right",
            yanchor="bottom",
            font=dict(size=12),
        ),
    )
    return fig


def make_combined_pairwise_systemic_figure(
    pairwise_df: pd.DataFrame,
    systemic_group_df: pd.DataFrame,
    systemic_overall_df: pd.DataFrame,
    z: int | None = None,
    threshold: float = SYSTEMIC_EXCLUSION_THRESHOLD,
) -> go.Figure:
    n_pairwise_rows = len(PAIRWISE_ROW_ORDER)
    fig = make_subplots(
        rows=n_pairwise_rows,
        cols=2,
        shared_xaxes="columns",
        vertical_spacing=0.05,
        horizontal_spacing=0.14,
        specs=[[{}, {"rowspan": n_pairwise_rows}]] + [[{}, None] for _ in range(n_pairwise_rows - 1)],
        subplot_titles=("A. Monoculture (correlations)", "B. Systemic Exclusion"),
    )

    bins = np.linspace(-1, 1, 17)
    x_grid = np.linspace(-1, 1, 300)

    for row_idx, term in enumerate(PAIRWISE_ROW_ORDER, start=1):
        term_df = pairwise_df.loc[pairwise_df["term"] == term]
        for status in STATUS_ORDER:
            status_df = term_df.loc[term_df["status"] == status]
            if status_df.empty:
                continue

            fig.add_trace(
                go.Histogram(
                    x=status_df["pearson_r"],
                    xbins=dict(start=-1, end=1, size=(bins[1] - bins[0])),
                    histnorm="probability density",
                    marker=dict(
                        color=STATUS_COLORS[status],
                        line=dict(color=STATUS_COLORS[status], width=1),
                    ),
                    opacity=0.22,
                    name=status,
                    legendgroup=status,
                    showlegend=(row_idx == 1),
                ),
                row=row_idx,
                col=1,
            )

            density = _density_curve(status_df["pearson_r"].to_numpy(), x_grid)
            fig.add_trace(
                go.Scatter(
                    x=x_grid,
                    y=density,
                    mode="lines",
                    line=dict(color=STATUS_COLORS[status], width=3),
                    legendgroup=status,
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=row_idx,
                col=1,
            )

        shift_result = paired_mean_correlation_shift(term_df)
        if shift_result is not None:
            annotation = f"\u0394r\u0304 = {float(shift_result['delta_r']):.3f}"
            xref_name = "x" if row_idx == 1 else f"x{row_idx + 1}"
            yref_name = "y" if row_idx == 1 else f"y{row_idx + 1}"
            fig.add_annotation(
                x=0.03,
                y=0.86,
                xref=f"{xref_name} domain",
                yref=f"{yref_name} domain",
                text=annotation,
                showarrow=False,
                align="left",
                font=dict(size=11, color="#324B72"),
            )

        fig.add_vline(x=0, row=row_idx, col=1, line_color="#B8B8B8", line_width=1.5)
        fig.update_yaxes(
            title_text=PAIRWISE_ROW_LABELS[term],
            row=row_idx,
            col=1,
            showticklabels=False,
            title_font=dict(size=12),
        )

    fig.update_xaxes(
        range=[-1, 1],
        row=len(PAIRWISE_ROW_ORDER),
        col=1,
    )

    plot_order = (
        systemic_group_df.loc[systemic_group_df["status"] == "Post-train"]
        .sort_values("estimate", ascending=False)["intersection_group"]
        .tolist()
    )
    if not plot_order:
        plot_order = INTERSECTION_GROUP_ORDER

    systemic_group_df = systemic_group_df.copy()
    base_y = {group: len(plot_order) - 1 - idx for idx, group in enumerate(plot_order)}
    systemic_group_df["y_base"] = systemic_group_df["intersection_group"].map(base_y).astype(float)
    systemic_group_df["y"] = systemic_group_df["y_base"] + systemic_group_df["status"].map(STATUS_OFFSETS)

    for status in STATUS_ORDER:
        row = systemic_overall_df.loc[systemic_overall_df["status"] == status].iloc[0]
        fig.add_shape(
            type="rect",
            x0=row["ci_low"] * 100,
            x1=row["ci_high"] * 100,
            y0=0,
            y1=1,
            xref="x2",
            yref="y2 domain",
            fillcolor=STATUS_COLORS[status],
            opacity=0.22,
            line_width=0,
            layer="below",
        )

    for status in STATUS_ORDER:
        status_df = systemic_group_df.loc[systemic_group_df["status"] == status]
        fig.add_trace(
            go.Scatter(
                x=status_df["estimate"] * 100,
                y=status_df["y"],
                mode="markers",
                name=status,
                legendgroup=status,
                showlegend=False,
                marker=dict(
                    color=STATUS_COLORS[status],
                    size=10,
                    symbol="circle" if status == "Base" else "diamond",
                ),
                error_x=dict(
                    type="data",
                    symmetric=False,
                    array=(status_df["ci_high"] - status_df["estimate"]) * 100,
                    arrayminus=(status_df["estimate"] - status_df["ci_low"]) * 100,
                    color=STATUS_COLORS[status],
                    thickness=2,
                    width=4,
                ),
                hovertemplate=(
                    "<b>%{customdata[0]} (%{customdata[1]})</b><br>"
                    "Systemic exclusion: %{x:.1f}%<br>"
                    "95% CI: [%{customdata[2]:.1f}%%, %{customdata[3]:.1f}%%]<br>"
                    "n = %{customdata[4]}<extra></extra>"
                ),
                customdata=np.column_stack(
                    [
                        status_df["label"],
                        status_df["status"],
                        status_df["ci_low"] * 100,
                        status_df["ci_high"] * 100,
                        status_df["n"],
                    ]
                ),
            ),
            row=1,
            col=2,
        )

    fig.update_yaxes(
        row=1,
        col=2,
        tickmode="array",
        tickvals=list(range(len(plot_order) - 1, -1, -1)),
        ticktext=[INTERSECTION_GROUP_LABELS[group] for group in plot_order],
        title="Intersection (Race|Gender|Age)",
        showgrid=True,
        gridcolor="#D9DCE8",
        zeroline=False,
        title_font=dict(size=12),
        tickfont=dict(size=12),
    )
    fig.update_xaxes(
        row=1,
        col=2,
        ticksuffix="%",
        showgrid=True,
        gridcolor="#D9DCE8",
        zeroline=False,
    )

    _base_layout(fig)
    fig.update_layout(
        barmode="overlay",
        width=800,
        height=220 + 60 * n_pairwise_rows,
        margin=dict(l=40, r=10, t=20, b=70),
        legend=dict(
            orientation="h",
            x=0.5,
            y=-0.3,
            xanchor="center",
            yanchor="bottom",
            font=dict(size=15),
            title=dict(text="Status", font=dict(size=12)),
        ),
    )
    return fig


def print_bias_and_interaction_summary(
    bias_plot_df: pd.DataFrame,
    pooled_plot_df: pd.DataFrame,
    version: str = "",
) -> None:
    scale_label = " (1-10 rating points)" if is_secondary_version(version) else ""
    print(f"\n=== Bias Coefficients{scale_label} ===")
    bias_summary = (
        bias_plot_df.loc[:, ["term_label", "status", "family", "coef", "ci_low", "ci_high"]]
        .sort_values(["term_label", "status", "family"])
        .reset_index(drop=True)
    )
    print(bias_summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print(f"\n=== Post-Training Bias Shifts{scale_label} ===")
    pooled_summary = (
        pooled_plot_df.loc[:, ["term_label", "coef", "ci_low", "ci_high"]]
        .sort_values("term_label")
        .reset_index(drop=True)
    )
    print(pooled_summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


def print_pairwise_and_systemic_summary(
    pairwise_df: pd.DataFrame,
    systemic_group_df: pd.DataFrame,
    systemic_overall_df: pd.DataFrame,
) -> None:
    print("\n=== Pairwise Experiment Correlations ===")
    pairwise_rows: list[dict[str, float | str]] = []
    for term in PAIRWISE_ROW_ORDER:
        term_df = pairwise_df.loc[pairwise_df["term"] == term]
        shift_result = paired_mean_correlation_shift(term_df)
        if shift_result is None:
            continue
        pairwise_rows.append(
            {
                "row": PAIRWISE_ROW_LABELS[term],
                "base_mean_r": float(shift_result["base_mean_r"]),
                "post_mean_r": float(shift_result["post_mean_r"]),
                "delta_r": float(shift_result["delta_r"]),
                "n_pairs": int(shift_result["n_pairs"]),
            }
        )
    if pairwise_rows:
        pairwise_summary = pd.DataFrame(pairwise_rows)
        print(pairwise_summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    title_z = int(systemic_overall_df["z"].iloc[0]) if "z" in systemic_overall_df else None
    threshold = (
        float(systemic_overall_df["threshold"].iloc[0])
        if "threshold" in systemic_overall_df
        else SYSTEMIC_EXCLUSION_THRESHOLD
    )
    criterion = (
        f">= {title_z} model{'s' if title_z != 1 else ''} with picked_perturbed < {threshold:g}"
        if title_z is not None
        else f"picked_perturbed < {threshold:g}"
    )

    print(f"\n=== Systemic Exclusion Overall CI Bands ({criterion}) ===")
    overall_summary = systemic_overall_df.copy()
    overall_summary["estimate_pct"] = overall_summary["estimate"] * 100
    overall_summary["ci_low_pct"] = overall_summary["ci_low"] * 100
    overall_summary["ci_high_pct"] = overall_summary["ci_high"] * 100
    overall_summary = overall_summary.loc[:, ["status", "estimate_pct", "ci_low_pct", "ci_high_pct", "n"]]
    print(overall_summary.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    print(f"\n=== Systemic Exclusion By Group ({criterion}) ===")
    group_summary = systemic_group_df.copy()
    group_summary["estimate_pct"] = group_summary["estimate"] * 100
    group_summary["ci_low_pct"] = group_summary["ci_low"] * 100
    group_summary["ci_high_pct"] = group_summary["ci_high"] * 100
    group_summary = (
        group_summary.loc[:, ["status", "label", "estimate_pct", "ci_low_pct", "ci_high_pct", "n"]]
        .sort_values(["status", "estimate_pct"], ascending=[True, False])
        .reset_index(drop=True)
    )
    print(group_summary.to_string(index=False, float_format=lambda x: f"{x:.2f}"))


def main(version: str, systemic_exclusion_z: int | None = None) -> None:
    systemic_suffix = f"_z{systemic_exclusion_z}" if systemic_exclusion_z is not None else ""

    coefficients_df = load_regression_coefficients(version)
    bias_plot_df = prepare_plot_df(coefficients_df)

    pooled_interactions_df = load_pooled_interactions(version)
    pooled_plot_df = prepare_pooled_interaction_df(pooled_interactions_df)
    print_bias_and_interaction_summary(bias_plot_df, pooled_plot_df, version)
    combined_paths = save_plotly_figure(
        make_combined_bias_interaction_figure(bias_plot_df, pooled_plot_df, version),
        f"bias_and_pooled_post_interactions_{version}",
    )

    prepared_results_df = load_prepared_results(version)
    pairwise_df = prepare_pairwise_correlation_df(prepared_results_df)

    systemic_group_df, systemic_overall_df = prepare_systemic_exclusion_df(
        prepared_results_df,
        z=systemic_exclusion_z,
    )
    print_pairwise_and_systemic_summary(pairwise_df, systemic_group_df, systemic_overall_df)
    pairwise_systemic_paths = save_plotly_figure(
        make_combined_pairwise_systemic_figure(
            pairwise_df,
            systemic_group_df,
            systemic_overall_df,
            z=systemic_exclusion_z,
        ),
        f"pairwise_demographic_correlations_and_systemic_exclusion_{version}{systemic_suffix}",
    )

    for path in [*combined_paths, *pairwise_systemic_paths]:
        print(f"Wrote figure to {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_version_argument(parser)
    parser.add_argument(
        "--z",
        "--systemic-exclusion-z",
        dest="systemic_exclusion_z",
        type=int,
        default=9,
        help="Define systemic exclusion as >= z models with picked_perturbed < 0.5. Defaults to 9.",
    )
    args = parser.parse_args()
    main(version=args.version, systemic_exclusion_z=args.systemic_exclusion_z)

from __future__ import annotations

import argparse
from pathlib import Path
import re

import pandas as pd
import statsmodels.formula.api as smf
import sys
from dotenv import load_dotenv

sys.path.append(str(Path(__file__).parent.parent))
load_dotenv()

from pipeline.utils.model_filters import exclude_olmo_appendix_only_models  # noqa: E402
from pipeline.config import (  # noqa: E402
    N_TRIALS,
    add_version_argument,
    is_secondary_version,
    resolve_results_path,
)


FULL_REGRESSION_FORMULA = (
    "picked_perturbed ~ black + female + over40 + C(soc_code) +"
    "title_similarity + skill_similarity + degree_similarity"
)

SECONDARY_FULL_REGRESSION_FORMULA = (
    "picked_perturbed ~ black + female + over40 + C(soc_code) +"
    "perturbed_title_similarity + perturbed_skills_similarity + perturbed_degree_similarity"
)

FULL_REGRESSION_INTERACTIONS_FORMULA = (
    "picked_perturbed ~ black * female * over40 + C(soc_code) +"
    "title_similarity + skill_similarity + degree_similarity"
)

SECONDARY_FULL_REGRESSION_INTERACTIONS_FORMULA = (
    "picked_perturbed ~ black * female * over40 + C(soc_code) +"
    "perturbed_title_similarity + perturbed_skills_similarity + perturbed_degree_similarity"
)

DEMOGRAPHICS_ONLY_REGRESSION_FORMULA = (
    "picked_perturbed ~ black + female + over40"
)

SECONDARY_DEMOGRAPHICS_ONLY_REGRESSION_FORMULA = (
    "picked_perturbed ~ black + female + over40"
)

SIMILARITY_ONLY_REGRESSION_FORMULA = (
    "picked_perturbed ~ title_similarity + skill_similarity + degree_similarity"
)

SECONDARY_SIMILARITY_ONLY_REGRESSION_FORMULA = (
    "picked_perturbed ~ perturbed_title_similarity + perturbed_skills_similarity + perturbed_degree_similarity"
)

POOLED_REGRESSION_FORMULA = (
    "picked_perturbed ~ black * post + female * post + over40 * post + C(soc_code) +"
    "C(family) + title_similarity + skill_similarity + degree_similarity"
)

SECONDARY_POOLED_REGRESSION_FORMULA = (
    "picked_perturbed ~ black * post + female * post + over40 * post + C(soc_code) +"
    "C(family) + perturbed_title_similarity + perturbed_skills_similarity + perturbed_degree_similarity"
)

MODEL_REGRESSION_SPECS = {
    "full": FULL_REGRESSION_FORMULA,
    "full_interactions": FULL_REGRESSION_INTERACTIONS_FORMULA,
    "demographics_only": DEMOGRAPHICS_ONLY_REGRESSION_FORMULA,
    "similarity_only": SIMILARITY_ONLY_REGRESSION_FORMULA,
}
SECONDARY_MODEL_REGRESSION_SPECS = {
    "full": SECONDARY_FULL_REGRESSION_FORMULA,
    "full_interactions": SECONDARY_FULL_REGRESSION_INTERACTIONS_FORMULA,
    "demographics_only": SECONDARY_DEMOGRAPHICS_ONLY_REGRESSION_FORMULA,
    "similarity_only": SECONDARY_SIMILARITY_ONLY_REGRESSION_FORMULA,
}
OUTPUT_DATA_DIR = Path("output_data")


def model_regression_specs(version: str) -> dict[str, str]:
    if is_secondary_version(version):
        return SECONDARY_MODEL_REGRESSION_SPECS
    return MODEL_REGRESSION_SPECS


def model_similarity_columns(version: str) -> list[str]:
    if is_secondary_version(version):
        return [
            "perturbed_title_similarity",
            "perturbed_skills_similarity",
            "perturbed_degree_similarity",
        ]
    return [
        "title_similarity",
        "skill_similarity",
        "degree_similarity",
    ]


def coerce_numeric_column(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df:
        return pd.Series(pd.NA, index=df.index)
    return pd.to_numeric(df[column], errors="coerce")


def output_data_csv_path(filename: str) -> Path:
    OUTPUT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DATA_DIR / filename


def output_data_path(filename: str) -> Path:
    OUTPUT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DATA_DIR / filename


def resolve_csv_path(filename: str) -> Path:
    output_path = OUTPUT_DATA_DIR / filename
    return output_path if output_path.exists() else Path(filename)


def deterministic_summary_text(result: object) -> str:
    summary = str(result.summary())
    summary = re.sub(
        r"Date:\s+.*?(?=\s{2,}Prob \(F-statistic\):)",
        "Date:                     omitted",
        summary,
    )
    return re.sub(
        r"Time:\s+.*?(?=\s{2,}Log-Likelihood:)",
        "Time:                     omitted",
        summary,
    )


def model_phase(model_name: str) -> str:
    return "base" if model_name.endswith("_base") else "post"


def summarize_model_rsquared(coefficients_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    coefficients_df = exclude_olmo_appendix_only_models(coefficients_df)
    rsquared_df = (
        coefficients_df.loc[
            coefficients_df["term"] == "Intercept",
            ["model", "phase", "specification", "nobs", "rsquared"],
        ]
        .drop_duplicates()
        .sort_values(["phase", "model", "specification"])
        .reset_index(drop=True)
    )

    model_only_rsquared_df = (
        rsquared_df.loc[rsquared_df["specification"].isin(["demographics_only", "similarity_only"])]
        .assign(
            report_group=lambda df: (
                df["phase"]
                + "-"
                + df["specification"].map(
                    {
                        "demographics_only": "demographics",
                        "similarity_only": "similarity",
                    }
                )
            )
        )
        .reset_index(drop=True)
    )

    mean_rsquared_df = (
        model_only_rsquared_df.groupby("report_group", as_index=False)
        .agg(
            mean_rsquared=("rsquared", "mean"),
            n_models=("model", "nunique"),
        )
        .sort_values("report_group")
        .reset_index(drop=True)
    )

    return model_only_rsquared_df, mean_rsquared_df


def print_model_rsquared_summary(
    model_only_rsquared_df: pd.DataFrame,
    mean_rsquared_df: pd.DataFrame,
) -> None:
    print("\n=== Model-Specific R-squared ===")

    rsquared_display_df = (
        model_only_rsquared_df.pivot(
            index=["phase", "model"],
            columns="specification",
            values="rsquared",
        )
        .reset_index()
        .rename(
            columns={
                "demographics_only": "demographics_r2",
                "similarity_only": "similarity_r2",
            }
        )
        .sort_values(["phase", "model"])
        .reset_index(drop=True)
    )
    print(rsquared_display_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    print("\n=== Mean R-squared By Group ===")
    print(mean_rsquared_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))


def load_prepared_results(version: str, n_trials: int) -> pd.DataFrame:
    results_path = resolve_results_path(version, OUTPUT_DATA_DIR)
    if results_path.exists():
        return pd.read_csv(results_path)
    from pipeline.stg_03_prepare_data import aggregate_results

    return aggregate_results(version, n_trials)


def prepare_regression_df(results_df: pd.DataFrame) -> pd.DataFrame:
    return results_df.assign(
        black=lambda df: (df["variant_idx"] // 4).astype(int),
        female=lambda df: ((df["variant_idx"] % 4) // 2).astype(int),
        over40=lambda df: (df["variant_idx"] % 2).astype(int),
        post=lambda df: pd.to_numeric(df["post"], errors="coerce"),
        picked_perturbed=lambda df: pd.to_numeric(df["picked_perturbed"], errors="coerce"),
        # perturbed_first=lambda df: pd.to_numeric(df["perturbed_first"], errors="coerce"),
        title_similarity=lambda df: coerce_numeric_column(df, "title_similarity"),
        skill_similarity=lambda df: coerce_numeric_column(df, "skill_similarity"),
        degree_similarity=lambda df: coerce_numeric_column(df, "degree_similarity"),
        perturbed_title_similarity=lambda df: coerce_numeric_column(df, "perturbed_title_similarity"),
        perturbed_skills_similarity=lambda df: coerce_numeric_column(df, "perturbed_skills_similarity"),
        perturbed_degree_similarity=lambda df: coerce_numeric_column(df, "perturbed_degree_similarity"),
    )


def run_model_regressions(
    results_df: pd.DataFrame,
    version: str,
) -> tuple[dict[tuple[str, str], object], pd.DataFrame]:
    fitted_models: dict[tuple[str, str], object] = {}
    coefficient_rows: list[dict[str, float | str]] = []

    prepared_df = prepare_regression_df(results_df)
    regression_specs = model_regression_specs(version)
    similarity_columns = model_similarity_columns(version)

    for model_name in sorted(prepared_df["model"].dropna().unique()):
        df = (
            prepared_df.loc[prepared_df["model"] == model_name].copy()
            .dropna(
                subset=[
                    "picked_perturbed",
                    "black",
                    "female",
                    "over40",
                    # "perturbed_first",
                    *similarity_columns,
                    "experiment_id",
                ]
            )
        )

        for specification, formula in regression_specs.items():
            result = smf.ols(
                formula,
                data=df,
            ).fit(
                cov_type="cluster",
                cov_kwds={"groups": df["experiment_id"]},
            )

            fitted_models[(model_name, specification)] = result

            for term, coefficient in result.params.items():
                coefficient_rows.append(
                    {
                        "model": model_name,
                        "phase": model_phase(model_name),
                        "specification": specification,
                        "term": term,
                        "coef": float(coefficient),
                        "std_err": float(result.bse[term]),
                        "t_value": float(result.tvalues[term]),
                        "p_value": float(result.pvalues[term]),
                        "nobs": float(result.nobs),
                        "rsquared": float(result.rsquared),
                    }
                )

    return fitted_models, pd.DataFrame(coefficient_rows)


def pooled_regression_formula(version: str) -> str:
    if is_secondary_version(version):
        return SECONDARY_POOLED_REGRESSION_FORMULA
    return POOLED_REGRESSION_FORMULA


def run_pooled_regression(
    results_df: pd.DataFrame,
    version: str,
) -> tuple[object, pd.DataFrame, pd.DataFrame]:
    prepared_df = exclude_olmo_appendix_only_models(prepare_regression_df(results_df))
    similarity_columns = model_similarity_columns(version)
    df = prepared_df.dropna(
        subset=[
            "picked_perturbed",
            "post",
            "black",
            "female",
            "over40",
            # "perturbed_first",
            *similarity_columns,
            "experiment_id",
            "model",
        ]
    ).copy()

    result = smf.ols(
        pooled_regression_formula(version),
        data=df,
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": df["experiment_id"]},
    )

    coefficients_df = pd.DataFrame(
        [
            {
                "term": term,
                "coef": float(coefficient),
                "std_err": float(result.bse[term]),
                "t_value": float(result.tvalues[term]),
                "p_value": float(result.pvalues[term]),
                "nobs": float(result.nobs),
                "rsquared": float(result.rsquared),
            }
            for term, coefficient in result.params.items()
        ]
    )

    interaction_terms = {"black:post", "female:post", "over40:post"}
    interaction_df = (
        coefficients_df.loc[coefficients_df["term"].isin(interaction_terms)]
        .copy()
        .sort_values("term")
        .reset_index(drop=True)
    )

    return result, coefficients_df, interaction_df


def main(version: str, n_trials: int = N_TRIALS) -> None:
    results_df = load_prepared_results(version, n_trials)
    fitted_models, coefficients_df = run_model_regressions(results_df, version)
    model_only_rsquared_df, mean_rsquared_df = summarize_model_rsquared(coefficients_df)
    pooled_result, pooled_coefficients_df, pooled_interactions_df = run_pooled_regression(
        results_df,
        version,
    )

    summaries_path = output_data_path(f"regression_summaries_{version}.txt")
    with summaries_path.open("w") as f:
        for (model_name, specification), result in fitted_models.items():
            heading = f"=== {model_name} | {specification} ==="
            summary_text = deterministic_summary_text(result)
            print(heading)
            print(summary_text)
            print()

            f.write(f"{heading}\n")
            f.write(f"{summary_text}\n\n")

    coefficients_path = output_data_csv_path(f"regression_coefficients_{version}.csv")
    coefficients_df.to_csv(coefficients_path, index=False)

    model_rsquared_path = output_data_csv_path(f"regression_model_rsquared_{version}.csv")
    model_only_rsquared_df.to_csv(model_rsquared_path, index=False)

    mean_rsquared_path = output_data_csv_path(f"regression_group_mean_rsquared_{version}.csv")
    mean_rsquared_df.to_csv(mean_rsquared_path, index=False)

    print_model_rsquared_summary(model_only_rsquared_df, mean_rsquared_df)

    pooled_summary_path = output_data_path(f"pooled_regression_summary_{version}.txt")
    pooled_summary_path.write_text(deterministic_summary_text(pooled_result))

    pooled_coefficients_path = output_data_csv_path(f"pooled_regression_coefficients_{version}.csv")
    pooled_coefficients_df.to_csv(pooled_coefficients_path, index=False)

    pooled_interactions_path = output_data_csv_path(f"pooled_post_interactions_{version}.csv")
    pooled_interactions_df.to_csv(pooled_interactions_path, index=False)

    print(f"Wrote summaries to {summaries_path}")
    print(f"Wrote coefficients to {coefficients_path}")
    print(f"Wrote model R-squared values to {model_rsquared_path}")
    print(f"Wrote group mean R-squared values to {mean_rsquared_path}")
    print(f"Wrote pooled summary to {pooled_summary_path}")
    print(f"Wrote pooled coefficients to {pooled_coefficients_path}")
    print(f"Wrote pooled post interactions to {pooled_interactions_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run publication regressions.")
    add_version_argument(parser)
    parser.add_argument("--n-trials", type=int, default=N_TRIALS)
    args = parser.parse_args()
    main(args.version, args.n_trials)

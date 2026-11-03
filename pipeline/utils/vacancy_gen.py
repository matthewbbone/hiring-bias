from __future__ import annotations

from pathlib import Path
from typing import Literal
import os

import dill as pickle
import numpy as np
from snowflake.snowpark import Session, DataFrame as SPDataFrame
from snowflake.snowpark import functions as sfunc
from snowflake.snowpark.window import Window

from pipeline.utils.distributions import (
    ChoiceDist,
    SalaryDist,
    VacancyDist,
    vacancy_dist_from_dict,
    vacancy_dist_to_dict,
)

class VacancyGenerator:
    """
    Generates synthetic vacancy traits by sampling from Snowflake distributions.

    Builds per-SOC distributions for company names, titles, skills, and
    fits a log-normal distribution to salaries. Distributions are cached
    locally for reuse.
    """

    def __init__(
        self,
        session: Session,
        cache: Path
    ) -> None:
        self.session: Session = session
        self.cache: Path = cache
        self.vacancy_dist: dict[str, VacancyDist] = {}
        self.global_salary_params: SalaryDist = SalaryDist(mu=0.0, sigma=1.0)
        
    def _as_col(self, col_or_name: str | object) -> object:
        if isinstance(col_or_name, str):
            return sfunc.col(col_or_name)
        return col_or_name

    def _build_conditional_dist(
        self,
        base_sp: SPDataFrame,
        value_col: str | object,
        *,
        min_count: int,
        value_col_alias: str,
    ) -> SPDataFrame:

        value_col_expr = self._as_col(value_col)

        return (
            base_sp
            .group_by([sfunc.col("BGI_SOC6"), sfunc.col("BGI_SOC6_NAME"), value_col_expr])
            .agg(
                sfunc.count(sfunc.lit(1)).alias("COUNT")
            )
            .filter(
                sfunc.col("COUNT") >= min_count
            )
            .with_column(
                "FREQ",
                (sfunc.col("COUNT") / sfunc.sum(sfunc.col("COUNT")).over(
                    Window.partitionBy([sfunc.col("BGI_SOC6"), sfunc.col("BGI_SOC6_NAME")]))
                )
            )
            .select(
                sfunc.col("BGI_SOC6").alias("SOC"),
                sfunc.col("BGI_SOC6_NAME").alias("SOC_NAME"),
                value_col_expr.alias(value_col_alias),
                sfunc.col("FREQ"),
            )
        )

    def _compute_global_salary_params(
        self,
        base_sp: SPDataFrame
    ) -> None:
        global_salary_df = (
            base_sp
            .filter(sfunc.col("SALARY") > 0)
            .select(
                sfunc.ln(sfunc.col("SALARY")).alias("LOG_SALARY")
            )
            .agg(
                sfunc.avg("LOG_SALARY").alias("MU"),
                sfunc.stddev("LOG_SALARY").alias("SIGMA")
            )
            .to_pandas()
        )

        if len(global_salary_df) > 0:
            mu = float(global_salary_df["MU"].iloc[0])
            sigma = float(global_salary_df["SIGMA"].iloc[0])
            if not np.isnan(mu) and not np.isnan(sigma) and sigma > 0:
                self.global_salary_params = SalaryDist(mu=mu, sigma=sigma)

    def create_vacancy_dist(
        self,
        soc_codes: list[str] | None = None,
        year: int = 2024,
        education: list[Literal[
            "No Education Listed",
            "High school or GED",
            "Associate degree",
            "Bachelor's Degree",
            "Master's Degree",
            "Ph.D. or professional degree",
        ]] = ["Bachelor's Degree"],
        company_threshold: int = 10,
        title_threshold: int = 10,
        skills_threshold: int = 100,
    ) -> None:
        """
        Build per-SOC distributions for vacancy traits.

        Args:
            soc_codes: Optional list of SOC codes to include (None = all)
            year: Year to filter job postings by
            education: Education levels to include
        """
        vacancies_sp = (
            self.session.read.table(os.environ["VACANCIES_TABLE"])
            .filter(
                (sfunc.year(sfunc.col("POST_DATE")) == year) &
                (sfunc.col("BGI_MIN_REQUIRED_DEGREE").isin(education)) &
                (sfunc.col("BGI_SOC6").is_not_null()) &
                (sfunc.col("BGI_SOC6_NAME") != "Chief Executives") &
                (sfunc.col("COMPANY_RAW").is_not_null()) &
                (sfunc.col("JOBTITLE_RAW").is_not_null()) &
                (sfunc.length(sfunc.col("DESCRIPTION")) > 250) &
                (sfunc.array_size(sfunc.col("SKILLS_NAME")) >= 5) &
                (sfunc.col("BGI_NAICS4") != '5613') &
                (
                    (sfunc.col("BGI_SALARY_MIN_ANNUALIZED").is_not_null()) |
                    (sfunc.col("BGI_SALARY_MAX_ANNUALIZED").is_not_null())
                )
            )
            .select(
                sfunc.col("COMPANY_RAW"),
                sfunc.col("JOBTITLE_RAW"),
                sfunc.col("BGI_SOC6"),
                sfunc.col("BGI_SOC6_NAME"),
                sfunc.col("SKILLS_NAME"),
                ((
                    sfunc.coalesce(
                        sfunc.col("BGI_SALARY_MIN_ANNUALIZED"),
                        sfunc.col("BGI_SALARY_MAX_ANNUALIZED")
                    ) +
                    sfunc.coalesce(
                        sfunc.col("BGI_SALARY_MAX_ANNUALIZED"),
                        sfunc.col("BGI_SALARY_MIN_ANNUALIZED")
                    )
                ) / 2.0).alias("SALARY")
            )
        )

        if soc_codes is not None:
            vacancies_sp = vacancies_sp.filter(sfunc.col("BGI_SOC6").isin(soc_codes))

        self._compute_global_salary_params(vacancies_sp)

        company_sp = self._build_conditional_dist(
            vacancies_sp,
            value_col="COMPANY_RAW",
            value_col_alias="COMPANY_RAW",
            min_count=company_threshold
        )
        title_sp = self._build_conditional_dist(
            vacancies_sp,
            value_col="JOBTITLE_RAW",
            value_col_alias="JOBTITLE_RAW",
            min_count=title_threshold
        )

        skills_base_sp = (
            vacancies_sp
            .select(
                sfunc.col("BGI_SOC6"),
                sfunc.col("BGI_SOC6_NAME"),
                sfunc.col("SKILLS_NAME")
            )
            .filter(sfunc.col("SKILLS_NAME").is_not_null())
            .join_table_function("flatten", sfunc.col("SKILLS_NAME"))
            .select(
                sfunc.col("BGI_SOC6"),
                sfunc.col("BGI_SOC6_NAME"),
                sfunc.regexp_replace(
                    sfunc.to_varchar(sfunc.col("VALUE")),
                    '^\"|\"$',
                    ''
                ).alias("SKILL")
            )
        )
        skills_sp = self._build_conditional_dist(
            skills_base_sp,
            value_col="SKILL",
            value_col_alias="SKILL",
            min_count=skills_threshold
        )

        salary_sp = (
            vacancies_sp
            .filter(sfunc.col("SALARY") > 0)
            .group_by([sfunc.col("BGI_SOC6"), sfunc.col("BGI_SOC6_NAME")])
            .agg(
                sfunc.avg(sfunc.ln(sfunc.col("SALARY"))).alias("MU"),
                sfunc.stddev(sfunc.ln(sfunc.col("SALARY"))).alias("SIGMA")
            )
            .select(
                sfunc.col("BGI_SOC6").alias("SOC"),
                sfunc.col("MU"),
                sfunc.col("SIGMA")
            )
        )
        
        soc_name_map = company_sp.select("SOC", "SOC_NAME").distinct().to_pandas().set_index("SOC")["SOC_NAME"].to_dict()

        print("Pulling Vacancy Counts...")
        vacancy_counts = (
            vacancies_sp
            .group_by(sfunc.col("BGI_SOC6"))
            .agg(sfunc.count(sfunc.lit(1)).alias("N"))
            .to_pandas()
            .set_index("BGI_SOC6")["N"]
            .to_dict()
        )

        print("Pulling Company Data...")
        company_df = company_sp.to_pandas().sort_values("FREQ", ascending=False).groupby("SOC")
        print("Pulling Title Data...")
        title_df = title_sp.to_pandas().sort_values("FREQ", ascending=False).groupby("SOC")
        print("Pulling Skills Data...")
        skills_df = skills_sp.to_pandas().sort_values("FREQ", ascending=False).groupby("SOC")
        print("Pulling Salary Data...")
        salary_df = salary_sp.to_pandas().set_index("SOC")

        for soc_code in soc_codes:
            
            try:
                
                company = company_df.get_group(soc_code)
                title = title_df.get_group(soc_code)
                skills = skills_df.get_group(soc_code)
                
                # ensure we have enough data to build distributions
                assert len(company) > 10
                assert len(title) > 10
                assert len(skills) > 10
                
            except KeyError:
                continue
            except AssertionError:
                continue

            salary_row = salary_df.loc[soc_code] if soc_code in salary_df.index else None
            mu = float(salary_row["MU"]) if salary_row is not None else self.global_salary_params.mu
            sigma = float(salary_row["SIGMA"]) if salary_row is not None else self.global_salary_params.sigma

            self.vacancy_dist[soc_code] = VacancyDist(
                company_name=ChoiceDist(
                    choices=company["COMPANY_RAW"].tolist(),
                    freq=list(company["FREQ"] / company["FREQ"].sum())
                ),
                title=ChoiceDist(
                    choices=title["JOBTITLE_RAW"].tolist(),
                    freq=list(title["FREQ"] / title["FREQ"].sum())
                ),
                skills=ChoiceDist(
                    choices=skills["SKILL"].tolist(),
                    freq=list(skills["FREQ"] / skills["FREQ"].sum())
                ),
                salary=SalaryDist(mu=mu, sigma=sigma),
                soc_name=soc_name_map.get(soc_code, ""),
                count=int(vacancy_counts.get(soc_code, 0))
            )

    def reweighting_skills(
        self,
    ) -> None:
        """
        Apply TF-IDF reweighting to skill distributions.
        """
        print("TFIDF Reweighting Vacancy Skills...")

        tfs = {}
        idfs = {}
        vocabulary = list(set([
            skill for dist in self.vacancy_dist.values() if "SKILLS" in dist
            for skill in dist["SKILLS"]["CHOICES"]
        ]))
        N = len(self.vacancy_dist)

        for soc_code, dist in self.vacancy_dist.items():
            if "SKILLS" not in dist:
                continue
            tf = {}
            total_count = sum(dist["SKILLS"]["FREQ"])
            for skill, freq in zip(dist["SKILLS"]["CHOICES"], dist["SKILLS"]["FREQ"]):
                tf[skill] = freq / total_count
            tfs[soc_code] = tf

        for skill in vocabulary:
            df = sum(1 for dist in self.vacancy_dist.values() if "SKILLS" in dist and skill in dist["SKILLS"]["CHOICES"])
            idfs[skill] = np.log(N / (1 + df))

        for soc_code, dist in self.vacancy_dist.items():
            if "SKILLS" not in dist:
                continue
            reweighted_skills = []
            reweighted_freqs = []
            for skill, freq in zip(dist["SKILLS"]["CHOICES"], dist["SKILLS"]["FREQ"]):
                tf = tfs[soc_code].get(skill, 0)
                idf = idfs.get(skill, 0)
                reweighted_freq = tf * idf
                reweighted_skills.append(skill)
                reweighted_freqs.append(reweighted_freq)

            # Clamp negatives from TF-IDF and normalize
            reweighted_freqs = [max(0.0, f) for f in reweighted_freqs]
            total_reweighted = sum(reweighted_freqs)
            if total_reweighted > 0:
                reweighted_freqs = [f / total_reweighted for f in reweighted_freqs]
            else:
                # Fallback to original distribution (normalized) if all weights are zero
                orig_freqs = dist["SKILLS"]["FREQ"]
                total_orig = sum(orig_freqs)
                if total_orig > 0:
                    reweighted_freqs = [f / total_orig for f in orig_freqs]
                else:
                    # Ultimate fallback: uniform
                    n = len(reweighted_skills)
                    reweighted_freqs = [1.0 / n] * n if n > 0 else []

            sorted_indices = np.argsort(reweighted_freqs)[::-1]
            dist["SKILLS"].choices = [reweighted_skills[i] for i in sorted_indices]
            dist["SKILLS"].freq = [reweighted_freqs[i] for i in sorted_indices]

    def save_vacancy_dist_to_cache(
        self
    ) -> None:
        """Save vacancy distributions to local cache."""
        data = {
            soc_code: vacancy_dist_to_dict(dist)
            for soc_code, dist in self.vacancy_dist.items()
        }
        with open(self.cache / "vacancy_dist.pkl", "wb") as f:
            pickle.dump(data, f)

    def load_vacancy_dist_from_cache(
        self
    ) -> None:
        """Load vacancy distributions from local cache."""
        with open(self.cache / "vacancy_dist.pkl", "rb") as f:
            data = pickle.load(f)

        self.vacancy_dist = {
            soc_code: vacancy_dist_from_dict(dist)
            for soc_code, dist in data.items()
        }

        mus = [dist["SALARY"].mu for dist in self.vacancy_dist.values()]
        sigmas = [dist["SALARY"].sigma for dist in self.vacancy_dist.values()]
        if mus and sigmas:
            self.global_salary_params = SalaryDist(
                mu=float(np.mean(mus)),
                sigma=float(np.mean(sigmas)),
            )

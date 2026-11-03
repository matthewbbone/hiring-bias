from snowflake.snowpark import Session
from snowflake.snowpark import DataFrame as SPDataFrame
from pathlib import Path
import pandas as pd
from snowflake.snowpark import functions as sfunc
from snowflake.snowpark.window import Window
import os
from typing import Literal
import dill as pickle
import numpy as np
import polars as pl

from pipeline.utils.distributions import (
    ChoiceDist,
    OccupationDist,
    TransitionDist,
    choice_dist_from_dict,
    choice_dist_to_dict,
    occupation_dist_from_dict,
    occupation_dist_to_dict,
)


class ProfileGenerator:
    """
    Generates realistic synthetic profiles by sampling from Snowflake data distributions.

    This class queries Snowflake to build probability distributions for various
    profile attributes (degree majors, skills, job transitions, names) conditional
    on occupation. These distributions are then used to generate realistic synthetic
    profiles that match real-world patterns.

    The generator caches distributions locally to avoid repeated Snowflake queries.

    Attributes:
        session: Snowflake session for database queries
        cache: Path to local cache directory for pickled distributions
        occ_dist: Occupation-conditional distributions (degree, skills, transitions)
        degree_major_dist: School distributions conditional on degree major
        first_name_dist: First name distributions by birth year and gender
        last_name_dist: Last name distribution from Census data
    """

    def __init__(
        self,
        session: Session,
        cache: Path
    ) -> None:

        self.session: Session = session
        self.cache: Path = cache
        self.occ_dist: dict[str, OccupationDist] = {}
        self.degree_major_dist: dict[str, ChoiceDist] = {}
        self.first_name_dist: dict[int, dict[str, ChoiceDist]] = {}
        self.last_name_dist: ChoiceDist = ChoiceDist(choices=[], freq=[])

    def referenced_occupations(
        self,
    ) -> list[str]:
        """
        Get all SOC codes referenced in the occupation distribution.

        This includes both direct occupations and those appearing in
        job transition paths (TRANSITION_FROM and TRANSITION_TO).

        Returns:
            List of unique SOC codes used in the distribution
        """
        # Collect SOC codes from main distribution and transition paths
        soc_codes = set()
        for k, v in self.occ_dist.items():
            soc_codes.add(k)
            if "TRANSITION_FROM" in v:
                # Add origin occupations (where people transition FROM)
                for from_soc in v["TRANSITION_FROM"]["CHOICES"]:
                    soc_codes.add(from_soc)
                # Add destination occupations (where people transition TO)
                for to_soc in v["TRANSITION_TO"]["CHOICES"]:
                    soc_codes.add(to_soc)

        print("Number of referenced occupations:", len(soc_codes))

        return list(soc_codes)
    
    def _as_col(self, col_or_name: str | object) -> object:
        if isinstance(col_or_name, str):
            return sfunc.col(col_or_name)
        return col_or_name

    def _build_conditional_dist(
        self,
        base_sp: SPDataFrame,
        group_col: str | object,
        value_col: str | object,
        *,
        min_count: int,
        group_col_alias: str = "SOC_CODE",
        value_col_alias: str = "VALUE",
    ) -> SPDataFrame:

        group_col_expr = self._as_col(group_col)
        value_col_expr = self._as_col(value_col)

        return (
            base_sp
            .group_by([group_col_expr, value_col_expr])
            .agg(
                sfunc.count_distinct("PERSON_ID").alias("COUNT")
            )
            .filter(
                sfunc.col("COUNT") >= min_count
            )
            .with_column(
                "FREQ",
                (sfunc.col("COUNT") / sfunc.sum(sfunc.col("COUNT")).over(Window.partitionBy(group_col_expr)))
            )
            .select(
                group_col_expr.alias(group_col_alias),
                value_col_expr.alias(value_col_alias),
                sfunc.col("FREQ"),
            )
        )

    def transitions_data(
        self,
        transition_min_threshold: int = 10
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Query Snowflake to build job transition probability matrix.

        Analyzes sequential job changes within individual career histories
        to compute transition probabilities between occupations.

        Returns:
            DataFrame with columns: FROM_SOC, TO_SOC, COUNT, FROM_PROB, TO_PROB
            where FROM_PROB is P(TO|FROM) and TO_PROB is P(FROM|TO)
        """
        # Get experience records with valid start dates and SOC codes
        experiences_sp = (
            self.session.table(os.environ["EXPERIENCE_TABLE"])
            .select(
                sfunc.col("PERSON_ID"),
                sfunc.col("BGI_START_DATE"),
                sfunc.col("BGI_END_DATE"),
                sfunc.col("BGI_SOC6")
            )
            .filter(
                sfunc.col("BGI_START_DATE").isNotNull() &
                sfunc.col("BGI_SOC6").isNotNull()
            )
        )

        # Window to order jobs chronologically within each person's career
        w = (
            Window.partitionBy(
                sfunc.col("PERSON_ID")
            )
            .orderBy(
                sfunc.col("BGI_START_DATE").asc(),
                sfunc.col("BGI_END_DATE").asc_nulls_last()
            )
        )

        # Create pairs of consecutive jobs using LEAD window function
        with_next = (
            experiences_sp
            .select(
                sfunc.col("PERSON_ID"),
                sfunc.col("BGI_SOC6").alias("FROM_SOC"),
                sfunc.lead(sfunc.col("BGI_SOC6")).over(w).alias("TO_SOC")
            )
        )

        # Keep only valid transitions (both FROM and TO exist)
        transitions_sp = (
            with_next
            .filter(
                (sfunc.col("FROM_SOC").isNotNull()) &
                (sfunc.col("TO_SOC").isNotNull())
            )
        )

        # Count transitions by (FROM, TO) pairs, filter to significant transitions
        by_pair = (
            transitions_sp
            .group_by("FROM_SOC", "TO_SOC")
            .agg(
                sfunc.count("PERSON_ID").alias("COUNT")
            )
            .filter(
                sfunc.col("COUNT") >= transition_min_threshold 
            )
        )

        # Compute conditional probabilities in both directions
        w_from = Window.partitionBy("FROM_SOC")
        w_to = Window.partitionBy("TO_SOC")

        return (
            by_pair
            .with_column("FROM_TOTAL", sfunc.sum(sfunc.col("COUNT")).over(w_from))
            .with_column("TO_TOTAL", sfunc.sum(sfunc.col("COUNT")).over(w_to))
            .with_column("FROM_PROB", sfunc.col("COUNT") / sfunc.col("FROM_TOTAL"))
            .with_column("TO_PROB", sfunc.col("COUNT") / sfunc.col("TO_TOTAL"))
            .to_pandas()
        )
    
    def create_occ_dist(
        self,
        soc_codes: list[str],
        education: list[Literal[
            "High School",
            "Associate Degree",
            "Bachelor's Degree",
            "Master's Degree",
            "Doctorate Degree",
            "Postgraduate Diploma/Certificate",
            "Diploma of Foundation Studies",
            "Certificate",
            "Professional Diploma/Certificate",
            "Advanced Diploma/Certificate"
        ]] = ["Bachelor's Degree"],
        degree_min_threshold: int = 10,
        skills_min_threshold: int = 100,
        experience_min_threshold: int = 10,
        soc_name_map: dict[str, str] | None = None,
    ) -> None:
        """
        Build occupation-conditional distributions from Snowflake data.

        Queries Snowflake to compute probability distributions for:
        - Degree majors given occupation
        - Skills given occupation
        - Job transitions (origin and destination occupations)
        - Job titles/experiences given occupation

        Args:
            soc_codes: List of SOC codes to include
            education: Education levels to filter profiles by
            soc_name_map: Optional mapping of SOC code to occupation name
        """
        if soc_name_map is None:
            soc_name_map = {}
        # Filter to profiles with specified education level
        education_id_sp = (
            self.session.read.table(os.environ["EDUCATION_TABLE"])
            .filter(sfunc.col("BGI_DEGREE_MAX_PER_ID").isin(education))
            .select(sfunc.col("PERSON_ID")).distinct()
        )

        # Filter to profiles with relevant US work experience
        experience_id_sp = (
            self.session.read.table(os.environ["EXPERIENCE_TABLE"])
            .join(education_id_sp, on="PERSON_ID", how="inner")
            .filter(
                (sfunc.col("BGI_SOC6").isin(soc_codes)) &
                (sfunc.col("BGI_BEST_COUNTRY") == "United States")
            )
            .select(sfunc.col("PERSON_ID"), sfunc.col("BGI_SOC6")).distinct()
        )

        # Only keep occupations with sufficient sample size (10k+ profiles)
        min_count_occs = (
            experience_id_sp
            .group_by("BGI_SOC6")
            .agg(
                sfunc.count_distinct("PERSON_ID").alias("COUNT")
            )
            .filter(sfunc.col("COUNT") >= 10_000)
            .select(sfunc.col("BGI_SOC6")).distinct()
        )

        # Final filtered set of person-occupation pairs
        filtered_ids_sp = (
            experience_id_sp
            .join(min_count_occs, on="BGI_SOC6", how="inner")
        )

        print("Number of SOC Codes:", len(min_count_occs.collect()))
        print("Total Filtered Profiles:", filtered_ids_sp.count())

        # Build degree major distribution conditional on occupation
        degree_major_base_sp = (
            self.session.read.table(os.environ["EDUCATION_TABLE"])
            .join(filtered_ids_sp, on="PERSON_ID", how="inner")
            .filter(
                (sfunc.col("BGI_DEGREE_MAX_PER_ENTRY").isin(education)) &
                (sfunc.col("BGI_MAJOR_CIP4_NAME").isNotNull()) &
                (sfunc.col("BGI_SCHOOL_NAME").isNotNull())
            )
            # Combine degree level and major into single string
            .with_column(
                "DEGREE_MAJOR",
                sfunc.when(
                    sfunc.col("BGI_DEGREE_MAX_PER_ENTRY") == sfunc.lit("High School"),
                    sfunc.lit("High School")
                ).otherwise(
                    sfunc.concat(
                        sfunc.col("BGI_DEGREE_MAX_PER_ENTRY"),
                        sfunc.lit(", "),
                        sfunc.col("BGI_MAJOR_CIP4_NAME")
                    )
                )
            )
        )
        degree_major_sp = self._build_conditional_dist(
            degree_major_base_sp,
            group_col="BGI_SOC6",
            value_col="DEGREE_MAJOR",
            group_col_alias="SOC_CODE",
            value_col_alias="DEGREE_MAJOR",
            min_count=degree_min_threshold
        )

        # Build skills distribution conditional on occupation
        skills_base_sp = (
            self.session.read.table(os.environ["SKILLS_TABLE"])
            .join(filtered_ids_sp, on="PERSON_ID", how="inner")
            .filter(sfunc.col("SKILL_NAME").isNotNull())
        )
        skills_sp = self._build_conditional_dist(
            skills_base_sp,
            group_col="BGI_SOC6",
            value_col="SKILL_NAME",
            group_col_alias="SOC_CODE",
            value_col_alias="SKILL",
            min_count=skills_min_threshold
        )

        print("Pulling Degree Major Data...")
        degree_major_df = degree_major_sp.to_pandas() \
            .sort_values("FREQ", ascending=False).groupby("SOC_CODE")
        print("Pulling Skills Data...")
        skills_df = skills_sp.to_pandas() \
            .sort_values("FREQ", ascending=False).groupby("SOC_CODE")
        print("Pulling Transitions data...")
        transition_df  = self.transitions_data()

        print("Pulling Profile Counts...")
        profile_counts = (
            filtered_ids_sp
            .group_by("BGI_SOC6")
            .agg(sfunc.count_distinct("PERSON_ID").alias("N"))
            .to_pandas()
            .set_index("BGI_SOC6")["N"]
            .to_dict()
        )

        # Build the occupation distribution dictionary
        for soc_code in soc_codes:

            try:
                
                degree_majors = degree_major_df.get_group(soc_code)
                skills = skills_df.get_group(soc_code)
                
                assert len(degree_majors) > 1
                assert len(skills) > 10
                
            except KeyError:
                continue
            except AssertionError:
                continue

            # Get inbound transitions (people who moved INTO this occupation)
            from_transitions_df = (
                transition_df[transition_df["TO_SOC"] == soc_code]
                .sort_values("FROM_PROB", ascending=False)
            )
            # Get outbound transitions (people who moved FROM this occupation)
            to_transitions_df = (
                transition_df[transition_df["FROM_SOC"] == soc_code]
                .sort_values("TO_PROB", ascending=False)
            )

            self.occ_dist[soc_code] = OccupationDist(
                degree_majors=ChoiceDist(
                    choices=degree_majors["DEGREE_MAJOR"].to_list(),
                    freq=list(degree_majors["FREQ"] / degree_majors["FREQ"].sum()),
                ),
                skills=ChoiceDist(
                    choices=skills["SKILL"].to_list(),
                    freq=list(skills["FREQ"] / skills["FREQ"].sum()),
                ),
                transition_from=TransitionDist(
                    choices=from_transitions_df["FROM_SOC"].to_list(),
                    freq=list(from_transitions_df["FROM_PROB"] / from_transitions_df["FROM_PROB"].sum()),
                    prob=from_transitions_df["FROM_PROB"].to_list(),
                    total=from_transitions_df["FROM_TOTAL"].mean(),
                ),
                transition_to=TransitionDist(
                    choices=to_transitions_df["TO_SOC"].to_list(),
                    freq=list(to_transitions_df["TO_PROB"] / to_transitions_df["TO_PROB"].sum()),
                    prob=to_transitions_df["TO_PROB"].to_list(),
                    total=to_transitions_df["TO_TOTAL"].mean(),
                ),
                soc_name=soc_name_map.get(soc_code, ""),
                count=int(profile_counts.get(soc_code, 0)),
            )

        # Get all SOCs referenced in transitions for experience lookup
        referenced_socs = self.referenced_occupations()

        # Build job title/experience distribution for all referenced occupations
        experience_base_sp = (
            self.session.read.table(os.environ["EXPERIENCE_TABLE"])
            .filter(
                sfunc.col("TITLE_RAW").isNotNull(),
                sfunc.col("BGI_SOC6").isin(referenced_socs)
            )
        )
        experience_sp = self._build_conditional_dist(
            experience_base_sp,
            group_col=filtered_ids_sp["BGI_SOC6"],
            value_col="TITLE_RAW",
            group_col_alias="SOC_CODE",
            value_col_alias="EXPERIENCE",
            min_count=experience_min_threshold
        )

        print("Pulling Experience Data...")
        experience_df = experience_sp.to_pandas() \
            .sort_values("FREQ", ascending=False).groupby("SOC_CODE")

        # Add experience distribution to each occupation
        for soc in referenced_socs:
            try:
                experiences = experience_df.get_group(soc)
                assert len(experiences) > 10
            except KeyError:
                continue
            except AssertionError:
                continue

            if soc not in self.occ_dist:
                self.occ_dist[soc] = OccupationDist()

            self.occ_dist[soc].experiences = ChoiceDist(
                choices=experiences["EXPERIENCE"].to_list(),
                freq=list(experiences["FREQ"] / experiences["FREQ"].sum()),
            )

        # Build school distribution conditional on degree major
        self.create_degree_major_dist(
            filtered_ids_sp
        )

    def reweighting_skills(
        self,
    ) -> None:
        """
        Apply TF-IDF reweighting to skill distributions.

        Reduces weight of generic skills (common across all occupations)
        and increases weight of occupation-specific skills. This makes
        generated profiles more distinctive per occupation.
        """
        print("TFIDF Reweighting Skills...")

        # Build vocabulary of all skills across occupations
        tfs = {}
        idfs = {}
        vocabulary = list(set([
            skill for dist in self.occ_dist.values() if "SKILLS" in dist
            for skill in dist["SKILLS"]["CHOICES"]
        ]))
        N = len(self.occ_dist)

        # Calculate term frequency for each skill within each occupation
        for soc_code, dist in self.occ_dist.items():
            if "SKILLS" not in dist:
                continue
            tf = {}
            total_count = sum(dist["SKILLS"]["FREQ"])
            for skill, freq in zip(dist["SKILLS"]["CHOICES"], dist["SKILLS"]["FREQ"]):
                tf[skill] = freq / total_count
            tfs[soc_code] = tf

        # Calculate inverse document frequency (how rare is skill across occupations)
        for skill in vocabulary:
            df = sum(1 for dist in self.occ_dist.values() if "SKILLS" in dist and skill in dist["SKILLS"]["CHOICES"])
            idfs[skill] = np.log(N / (1 + df))

        # Apply TF-IDF reweighting
        for soc_code, dist in self.occ_dist.items():
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

            # Normalize to sum to 1
            total_reweighted = sum(reweighted_freqs)
            if total_reweighted > 0:
                reweighted_freqs = [f / total_reweighted for f in reweighted_freqs]

            # Re-sort by new weights (highest first)
            sorted_indices = np.argsort(reweighted_freqs)[::-1]
            self.occ_dist[soc_code]["SKILLS"]["CHOICES"] = [reweighted_skills[i] for i in sorted_indices]
            self.occ_dist[soc_code]["SKILLS"]["FREQ"] = [reweighted_freqs[i] for i in sorted_indices]

    def save_occ_dist_to_cache(
        self
    ) -> None:
        """Save occupation distribution to local cache."""
        with open(self.cache / "occ_dist.pkl", "wb") as f:
            data = {
                soc_code: occupation_dist_to_dict(dist)
                for soc_code, dist in self.occ_dist.items()
            }
            pickle.dump(data, f)

    def load_occ_dist_from_cache(
        self
    ) -> None:
        """
        Load occupation distribution from local cache.

        Also cleans any None values that may have been cached,
        re-normalizing frequencies after filtering.
        """
        with open(self.cache / "occ_dist.pkl", "rb") as f:
            data = pickle.load(f)

        if isinstance(data, dict):
            self.occ_dist = {
                soc_code: occupation_dist_from_dict(dist)
                for soc_code, dist in data.items()
            }
        else:
            self.occ_dist = data

        # Clean None values from cached data (data quality fix)
        for soc_code in self.occ_dist:
            for key in ["SKILLS", "EXPERIENCES", "DEGREE_MAJORS", "TRANSITION_FROM", "TRANSITION_TO"]:
                if key in self.occ_dist[soc_code]:
                    choices = self.occ_dist[soc_code][key]["CHOICES"]
                    freqs = self.occ_dist[soc_code][key]["FREQ"]

                    # Filter out None values
                    valid_indices = [i for i, choice in enumerate(choices) if choice is not None]
                    if len(valid_indices) < len(choices):
                        filtered_choices = [choices[i] for i in valid_indices]
                        filtered_freqs = [freqs[i] for i in valid_indices]

                        # Renormalize frequencies to sum to 1
                        total_freq = sum(filtered_freqs)
                        if total_freq > 0:
                            filtered_freqs = [f / total_freq for f in filtered_freqs]

                        self.occ_dist[soc_code][key]["CHOICES"] = filtered_choices
                        self.occ_dist[soc_code][key]["FREQ"] = filtered_freqs

    def create_degree_major_dist(
        self,
        filtered_ids_sp: SPDataFrame,
        school_min_threshold: int = 10,
    ) -> None:
        """
        Build school distribution conditional on degree major.

        For each degree major, computes probability distribution over
        schools that grant that degree.

        Args:
            filtered_ids_sp: Snowpark DataFrame of filtered person IDs
        """
        # Get all degree majors used in occupation distribution
        all_degree_majors = set()
        for k, v in self.occ_dist.items():
            if "DEGREE_MAJORS" in v:
                all_degree_majors.update(v["DEGREE_MAJORS"]["CHOICES"])

        # Query school distribution per degree major
        schools_base_sp = (
            self.session.read.table(os.environ["EDUCATION_TABLE"])
            .join(filtered_ids_sp, on="PERSON_ID", how="inner")
            .filter(
                (sfunc.col("BGI_DEGREE_MAX_PER_ENTRY").isNotNull()) &
                (sfunc.col("BGI_MAJOR_CIP4_NAME").isNotNull()) &
                (sfunc.col("BGI_SCHOOL_NAME").isNotNull())
            )
            .with_column(
                "DEGREE_MAJOR",
                sfunc.when(
                    sfunc.col("BGI_DEGREE_MAX_PER_ENTRY") == sfunc.lit("High School"),
                    sfunc.lit("High School")
                ).otherwise(
                    sfunc.concat(
                        sfunc.col("BGI_DEGREE_MAX_PER_ENTRY"),
                        sfunc.lit(", "),
                        sfunc.col("BGI_MAJOR_CIP4_NAME")
                    )
                )
            )
            .filter(sfunc.col("DEGREE_MAJOR").isin(list(all_degree_majors))) \
        )
        schools_sp = self._build_conditional_dist(
            schools_base_sp,
            group_col="DEGREE_MAJOR",
            value_col="BGI_SCHOOL_NAME",
            group_col_alias="DEGREE_MAJOR",
            value_col_alias="SCHOOL",
            min_count=school_min_threshold
        )

        print("Pulling Schools Data...")
        schools_df = schools_sp.to_pandas() \
            .sort_values("FREQ", ascending=False).groupby("DEGREE_MAJOR")

        for degree_major in schools_df.groups.keys():
            df = schools_df.get_group(degree_major)
            self.degree_major_dist[degree_major] = ChoiceDist(
                choices=df["SCHOOL"].tolist(),
                freq=list(df["FREQ"] / df["FREQ"].sum()),
            )

    def save_degree_major_dist_to_cache(
        self
    ) -> None:
        """Save degree major distribution to local cache."""
        with open(self.cache / "degree_major_dist.pkl", "wb") as f:
            data = {
                degree_major: choice_dist_to_dict(dist)
                for degree_major, dist in self.degree_major_dist.items()
            }
            pickle.dump(data, f)

    def load_degree_major_dist_from_cache(
        self
    ) -> None:
        """Load degree major distribution from local cache."""
        with open(self.cache / "degree_major_dist.pkl", "rb") as f:
            data = pickle.load(f)

        if isinstance(data, dict):
            self.degree_major_dist = {
                degree_major: choice_dist_from_dict(dist)
                for degree_major, dist in data.items()
            }
        else:
            self.degree_major_dist = data

    def create_first_name_dist(
        self
    ) -> None:
        """
        Build first name distribution from SSA baby names data.

        Loads Social Security Administration baby names by year and gender.
        Names are stored by birth year to enable age-appropriate name selection.
        """
        filenames = os.listdir("raw_data/names")
        for f in filenames:
            if f.endswith(".txt"):
                year = int(f[3:7])  # Extract year from filename (e.g., yob1990.txt)
                year_pl = pl.read_csv(
                    f"raw_data/names/{f}",
                    has_header=False,
                    new_columns=["FIRST_NAME", "GENDER", "COUNT"]
                ).group_by("GENDER")
                self.first_name_dist[year] = {}
                for g, g_pl in year_pl:
                    self.first_name_dist[year][g[0]] = ChoiceDist(
                        choices=g_pl["FIRST_NAME"].to_list(),
                        freq=list(g_pl["COUNT"] / g_pl["COUNT"].sum()),
                    )

    def save_first_name_dist_to_cache(
        self
    ) -> None:
        """Save first name distribution to local cache."""
        with open(self.cache / "first_name_dist.pkl", "wb") as f:
            data = {
                year: {gender: choice_dist_to_dict(dist) for gender, dist in by_gender.items()}
                for year, by_gender in self.first_name_dist.items()
            }
            pickle.dump(data, f)

    def load_first_name_dist_from_cache(
        self
    ) -> None:
        """Load first name distribution from local cache."""
        with open(self.cache / "first_name_dist.pkl", "rb") as f:
            data = pickle.load(f)

        if isinstance(data, dict):
            self.first_name_dist = {
                year: {
                    gender: choice_dist_from_dict(dist)
                    for gender, dist in by_gender.items()
                }
                for year, by_gender in data.items()
            }
        else:
            self.first_name_dist = data

    def create_last_name_dist(
        self
    ) -> None:
        """
        Build last name distribution from US Census data.

        Uses 2010 Census surnames data for population-representative
        last name frequencies.
        """
        last_pl = (
            pl.scan_csv("raw_data/Names_2010Census.csv")
            .filter(
                pl.col("name") != "ALL OTHER NAMES"  # Exclude aggregate category
            )
            .select(
                pl.col("name"),
                (pl.col("count") / pl.col("count").sum()).alias("FREQ")
            )
            .collect()
        )
        self.last_name_dist = ChoiceDist(
            choices=last_pl["name"].to_list(),
            freq=list(last_pl["FREQ"] / last_pl["FREQ"].sum()),
        )

    def save_last_name_dist_to_cache(
        self
    ) -> None:
        """Save last name distribution to local cache."""
        with open(self.cache / "last_name_dist.pkl", "wb") as f:
            pickle.dump(choice_dist_to_dict(self.last_name_dist), f)

    def load_last_name_dist_from_cache(
        self
    ) -> None:
        """Load last name distribution from local cache."""
        with open(self.cache / "last_name_dist.pkl", "rb") as f:
            data = pickle.load(f)

        self.last_name_dist = choice_dist_from_dict(data)

    def create_all_dists(
        self,
        soc_codes: list[str],
        education: list[Literal[
            "High School",
            "Associate Degree",
            "Bachelor's Degree",
            "Master's Degree",
            "Doctorate Degree",
            "Postgraduate Diploma/Certificate",
            "Diploma of Foundation Studies",
            "Certificate",
            "Professional Diploma/Certificate",
            "Advanced Diploma/Certificate"
        ]] = ["Bachelor's Degree"],
        soc_name_map: dict[str, str] | None = None,
    ) -> None:
        """
        Create all distributions needed for profile generation.

        Args:
            soc_codes: SOC codes to build distributions for
            education: Education levels to include
            soc_name_map: Optional mapping of SOC code to occupation name
        """
        self.create_occ_dist(soc_codes, education, soc_name_map=soc_name_map)
        self.create_first_name_dist()
        self.create_last_name_dist()

    def save_all_dists_to_cache(
        self
    ) -> None:
        """Save all distributions to local cache."""
        self.save_occ_dist_to_cache()
        self.save_degree_major_dist_to_cache()
        self.save_first_name_dist_to_cache()
        self.save_last_name_dist_to_cache()

    def load_all_dists_from_cache(
        self
    ) -> None:
        """Load all distributions from local cache."""
        self.load_occ_dist_from_cache()
        self.load_degree_major_dist_from_cache()
        self.load_first_name_dist_from_cache()
        self.load_last_name_dist_from_cache()

import argparse
import json
import os
from pathlib import Path
import sys

import dill as pickle
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from snowflake.snowpark import Session

load_dotenv()

sys.path.append(str(Path(__file__).parent.parent))

from pipeline.utils.generator import ExperimentGenerator  # noqa: E402
from pipeline.config import add_version_argument, cache_dir  # noqa: E402

def get_bachelor_soc_codes() -> list[str]:
    
    """Get SOC codes for bachelor's degree occupations."""
    educ_df = pd.read_excel("raw_data/Education, Training, and Experience.xlsx")
    educ_df = educ_df[educ_df["Scale ID"] == "RL"]
    educ_df = educ_df[["O*NET-SOC Code", "Title", "Category", "Data Value"]]
    educ_df["BACHELOR"] = educ_df["Category"] >= 6
    educ_df["POSTGRAD"] = educ_df["Category"] >= 8
    educ_df["PCT_BACHELOR"] = educ_df["Data Value"] * educ_df["BACHELOR"]
    educ_df["PCT_POSTGRAD"] = educ_df["Data Value"] * educ_df["POSTGRAD"]
    educ_df = educ_df.groupby(["O*NET-SOC Code", "Title"]).agg({
        "PCT_BACHELOR": "sum",
        "PCT_POSTGRAD": "sum"
    }).reset_index()
    educ_df = educ_df[(educ_df["PCT_BACHELOR"] > 66) & (educ_df["PCT_POSTGRAD"] < 33)]
    educ_df["SOC"] = educ_df["O*NET-SOC Code"].str.slice(0, 7)
    valid_soc_codes = set(educ_df["SOC"].tolist())
    print(f"Number of SOC codes with >66% bachelor and <33% postgrad: {len(valid_soc_codes)}")
    return list(valid_soc_codes)

def create_distributions(
    session: Session,
    valid_soc_codes: list[str],
    rng: np.random.Generator,
) -> None:

    ex = ExperimentGenerator(session, rng=rng)

    ex.create_vacancy_generator(
        soc_codes=valid_soc_codes,
    )
    
    ex.create_profile_generator(valid_soc_codes)

    all_soc_codes = ex.profile_generator.referenced_occupations()
    ex.create_vacancy_generator(
        soc_codes=all_soc_codes,
    )
    
def create_experiments(
    session: Session | None,
    version: str,
    n_exp_per_soc: int,
    rng: np.random.Generator,
    seed: int,
) -> None:

    ex = ExperimentGenerator(session, rng=rng)
    
    ex.load_vacancy_generator_from_cache()
    ex.vacancy_generator.reweighting_skills()
    
    ex.load_profile_generator_from_cache()
    ex.profile_generator.reweighting_skills()
                             
    vacancy_soc_codes = set(ex.vacancy_generator.vacancy_dist.keys())
    working_soc_codes = [
        k for k, v in ex.profile_generator.occ_dist.items()
        if v.skills is not None and k in vacancy_soc_codes
    ]
    ex.create_experiments(
        soc_codes=working_soc_codes,
        n_exp_per_soc=n_exp_per_soc,
        experiment_type="hiring"
    )
    
    cache_root = cache_dir()
    
    # remove generators before pickling
    for exp in ex.experiments:
        exp.profile_generator = None
        exp.vacancy_generator = None
        exp.names_dict = None
    version_dir = cache_root / version
    version_dir.mkdir(parents=True, exist_ok=True)
    with open(version_dir / "experiments.pkl", "wb") as f:
        pickle.dump(ex.experiments, f)
    (version_dir / "generation_metadata.json").write_text(
        json.dumps(
            {
                "version": version,
                "seed": seed,
                "n_exp_per_soc": n_exp_per_soc,
                "historical_cache": False,
            },
            indent=2,
        ) + "\n"
    )
    

def build_snowflake_session() -> Session:
    connection_parameters = {
        "user": os.getenv("SNOWFLAKE_USER"),
        "password": os.getenv("SNOWFLAKE_PASSWORD"),
        "account": os.getenv("SNOWFLAKE_ACCOUNT"),
        "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE"),
        "database": os.getenv("SNOWFLAKE_DATABASE"),
        "schema": os.getenv("SNOWFLAKE_SCHEMA"),
    }
    missing = [key for key, value in connection_parameters.items() if not value]
    if missing:
        raise ValueError(
            "Missing Snowflake settings: " + ", ".join(f"SNOWFLAKE_{key.upper()}" for key in missing)
        )
    return Session.builder.configs(connection_parameters).create()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create experiment distributions and definitions.")
    add_version_argument(parser)
    parser.add_argument("--seed", type=int, default=42, help="Generation seed. Defaults to 42.")
    parser.add_argument("--n-exp-per-soc", type=int, default=10)
    parser.add_argument(
        "--create-distributions",
        action="store_true",
        help="Rebuild proprietary-source distributions before creating experiments.",
    )
    parser.add_argument(
        "--distributions-only",
        action="store_true",
        help="Create distributions without replacing experiments.pkl.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    session = None

    if args.create_distributions or args.distributions_only:
        session = build_snowflake_session()
        try:
            create_distributions(session, get_bachelor_soc_codes(), rng)
        finally:
            session.close()

    if not args.distributions_only:
        create_experiments(
            None,
            args.version,
            n_exp_per_soc=args.n_exp_per_soc,
            rng=rng,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()

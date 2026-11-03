from snowflake.snowpark import Session
from pathlib import Path
import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

from pipeline.utils.vacancy_gen import VacancyGenerator
from pipeline.utils.profile_gen import ProfileGenerator
from pipeline.utils.experiments import HiringExperiment, ApplyingExperiment

load_dotenv()

class ExperimentGenerator:
    
    def __init__(
        self,
        session: Session | None,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.rng = rng or np.random.default_rng()
        self.session = session
        self.cache = Path(os.getenv("CACHE_DIR", "experiment_cache"))
        self.vacancy_generator: VacancyGenerator = VacancyGenerator(
            session,
            cache=self.cache
        )
        self.profile_generator: ProfileGenerator = ProfileGenerator(
            session,
            cache=self.cache
        )
        self.experiments: list[HiringExperiment|ApplyingExperiment] = []
        
        # creating name distributions for profile perturbation
        name_df = (
            pd.read_csv("raw_data/audit_study_names.csv")
            [["race", "gender", "firstname", "lastname"]]
            .drop_duplicates()
            .reset_index(drop=True)
            .rename(columns={"firstname": "FIRST_NAME", "lastname": "LAST_NAME"})
        )

        name_df["RACE"] = name_df["race"].map({"White": 0, "Black": 1})
        name_df["GENDER"] = name_df["gender"].map({"Male": 0, "Female": 1})

        self.names_dict = {
            key: {
                "FIRST_NAMES": group["FIRST_NAME"].dropna().unique().tolist(),
                "LAST_NAMES": group["LAST_NAME"].dropna().unique().tolist(),
            }
            for key, group in name_df.groupby(["RACE", "GENDER"])
        }

    def create_vacancy_generator(self, soc_codes: list[str]) -> None:
        self.vacancy_generator.create_vacancy_dist(soc_codes)
        self.vacancy_generator.save_vacancy_dist_to_cache()
        
    def load_vacancy_generator_from_cache(self) -> None:
        self.vacancy_generator.load_vacancy_dist_from_cache()

    def create_profile_generator(self, soc_codes: list[str]) -> None:
        
        soc_name_map = {
            soc: dist.soc_name
            for soc, dist in self.vacancy_generator.vacancy_dist.items()
            if dist.soc_name
        }
        
        self.profile_generator.create_all_dists(soc_codes, soc_name_map=soc_name_map)
        self.profile_generator.save_all_dists_to_cache()

    def load_profile_generator_from_cache(self) -> None:
        self.profile_generator.load_all_dists_from_cache()
        
    def create_experiments(
        self, 
        soc_codes: list[str],
        n_exp_per_soc: int, 
        experiment_type: str = "hiring",
    ) -> None:
        
        experiment_class = HiringExperiment if experiment_type == "hiring" else ApplyingExperiment
        
        ctr = 0
        for soc_code in tqdm(soc_codes, desc="Creating experiments"):
            for _ in range(n_exp_per_soc):
                soc_name = self.vacancy_generator.vacancy_dist[soc_code].soc_name
                experiment = experiment_class(
                    experiment_id=ctr,
                    profile_generator=self.profile_generator,
                    vacancy_generator=self.vacancy_generator,
                    names_dict=self.names_dict,
                    rng=self.rng,
                )
                experiment.generate_experiment(soc_code, soc_name)
                ctr += 1
                
                self.experiments.append(experiment)

import argparse
from functools import lru_cache
from pathlib import Path
import sys

import dill as pickle
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))
load_dotenv()

from sentence_transformers import SentenceTransformer  # noqa: E402
from pipeline.config import (  # noqa: E402
    ANALYSIS_MODELS,
    N_TRIALS,
    SENTENCE_TRANSFORMER_MODEL,
    SENTENCE_TRANSFORMER_REVISION,
    add_version_argument,
    cache_dir,
    is_secondary_version,
    validate_experiment_cache,
)
from pipeline.utils.llm import expand_experiment_rows  # noqa: E402

OUTPUT_DATA_DIR = Path("output_data")


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    try:
        from huggingface_hub import snapshot_download

        snapshot_path = snapshot_download(
            SENTENCE_TRANSFORMER_MODEL,
            revision=SENTENCE_TRANSFORMER_REVISION,
            local_files_only=True,
        )
        return SentenceTransformer(
            snapshot_path,
            local_files_only=True,
        )
    except OSError:
        return SentenceTransformer(
            SENTENCE_TRANSFORMER_MODEL,
            revision=SENTENCE_TRANSFORMER_REVISION,
        )


def output_data_path(filename: str) -> Path:
    OUTPUT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DATA_DIR / filename


def load_experiments(version: str):
    with open(cache_dir() / version / "experiments.pkl", "rb") as f:
        return pickle.load(f)


def write_experiment_variant_prompts(experiments, version: str) -> Path:
    prompts_path = output_data_path(f"experiment_variant_prompts_{version}.txt")
    secondary_prompt = is_secondary_version(version)

    with prompts_path.open("w") as f:
        for experiment in tqdm(experiments, desc="Writing experiment prompts"):
            for row in expand_experiment_rows(experiment, secondary_prompt=secondary_prompt):
                f.write(
                    "\n".join(
                        [
                            "=" * 80,
                            f"experiment_id: {row['experiment_id']}",
                            f"variant_idx: {row['variant_idx']}",
                            f"ascending: {row['ascending']}",
                            "prompt:",
                            row["prompt"],
                            "",
                        ]
                    )
                )

    return prompts_path

def relative_similarity(left: list, right: list, center: list) -> float:

    model = get_embedding_model()
    left_embeds = model.encode(
        left,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    right_embeds = model.encode(
        right,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    center_embeds = model.encode(
        center,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    left_sim = left_embeds @ center_embeds.T
    right_sim = right_embeds @ center_embeds.T

    return float(left_sim.mean() - right_sim.mean())

def perturbed_similarity(left: list, center: list) -> float:
    
    model = get_embedding_model()
    left_embeds = model.encode(
        left,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    center_embeds = model.encode(
        center,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    left_sim = left_embeds @ center_embeds.T

    return float(left_sim.mean())

def normalize_outcome(values: pd.Series, version: str) -> pd.Series:
    numeric = values.astype(float)
    if is_secondary_version(version):
        return (numeric - 1) / 10
    return numeric


def model_phase(model_name: str) -> int:
    return 0 if model_name.endswith("_base") else 1


def model_family(model_name: str) -> str:
    return model_name.split("_")[0]


def aggregate_results(version: str, n_trials: int):
    validate_experiment_cache(version, n_trials=n_trials)
    cache = cache_dir()

    experiments = load_experiments(version)
        
    experiment_rows = []
    for exp in tqdm(experiments, desc="Processing experiments"):
        
        experiment_rows.append({
            "experiment_id": exp.experiment_id,
            "soc_code": exp.vacancies[0].soc_code,
            "soc_name": exp.vacancies[0].soc_name,
            "perturbed_idx": exp.perturbed_idx,
            "prompt": exp.prompt(0, secondary_prompt=is_secondary_version(version)),
            "vacancy_company": exp.vacancies[0].company_name,
            "vacancy_title": exp.vacancies[0].title,
            "vacancy_skills": exp.vacancies[0].skills,
            "vacancy_salary": exp.vacancies[0].salary,
            "perturbed_first_name": exp.profiles[exp.perturbed_idx].first_name,
            "perturbed_last_name": exp.profiles[exp.perturbed_idx].last_name,
            "perturbed_degree": exp.profiles[exp.perturbed_idx].degree_major,
            "perturbed_school": exp.profiles[exp.perturbed_idx].school,
            "perturbed_skills": exp.profiles[exp.perturbed_idx].skills,
            "perturbed_experience": exp.profiles[exp.perturbed_idx].experience,
            "perturbed_grad_year": exp.profiles[exp.perturbed_idx].grad_year,
            "base_first_name": exp.profiles[1- exp.perturbed_idx].first_name,
            "base_last_name": exp.profiles[1- exp.perturbed_idx].last_name,
            "base_degree": exp.profiles[1- exp.perturbed_idx].degree_major,
            "base_school": exp.profiles[1- exp.perturbed_idx].school,
            "base_skills": exp.profiles[1- exp.perturbed_idx].skills,
            "base_experience": exp.profiles[1- exp.perturbed_idx].experience,
            "base_grad_year": exp.profiles[1- exp.perturbed_idx].grad_year,
            "skill_similarity": relative_similarity(
                exp.profiles[exp.perturbed_idx].skills,
                exp.profiles[1- exp.perturbed_idx].skills,
                exp.vacancies[0].skills
            ),
            "title_similarity": relative_similarity(
                [exp.profiles[exp.perturbed_idx].experience],
                [exp.profiles[1- exp.perturbed_idx].experience],
                [exp.vacancies[0].title]
            ),
            "degree_similarity": relative_similarity(
                [exp.profiles[exp.perturbed_idx].degree_major],
                [exp.profiles[1- exp.perturbed_idx].degree_major],
                [exp.vacancies[0].title]
            ),
            "school_similarity": relative_similarity(
                [exp.profiles[exp.perturbed_idx].school],
                [exp.profiles[1- exp.perturbed_idx].school],
                [exp.vacancies[0].title]
            ),
            "perturbed_skills_similarity": perturbed_similarity(
                exp.profiles[exp.perturbed_idx].skills,
                exp.vacancies[0].skills
            ),
            "perturbed_title_similarity": perturbed_similarity(
                [exp.profiles[exp.perturbed_idx].experience],
                [exp.vacancies[0].title]
            ),
            "perturbed_degree_similarity": perturbed_similarity(
                [exp.profiles[exp.perturbed_idx].degree_major],
                [exp.vacancies[0].title]
            ),
            "perturbed_school_similarity": perturbed_similarity(
                [exp.profiles[exp.perturbed_idx].school],
                [exp.vacancies[0].title]
            ),
        })

    experiment_df = pd.DataFrame(experiment_rows)
    experiment_df.head()
    
    result_rows = []
    for m in tqdm(ANALYSIS_MODELS, desc="Loading model results"):
        
        for t in range(n_trials):
            
            results_file = cache / version / m / f"trial_{t + 1}_results.pkl"
            with open(results_file, "rb") as f:
                trial_results = pickle.load(f)

            for exp in trial_results:
                exp["trial"] = t + 1
                exp['model'] = m

            result_rows.extend(trial_results)

    results_df = pd.DataFrame(result_rows)
    
    intersection_map = {
        0: "white_male_young",
        1: "white_male_old",
        2: "white_female_young",
        3: "white_female_old",
        4: "black_male_young",
        5: "black_male_old",
        6: "black_female_young",
        7: "black_female_old",
    }
    
    race_map = {
        0: 0, 1: 1, 2: 1, 3: 1,
        4: 2, 5: 2, 6: 2, 7: 2
    }

    gender_map = {
        0: 0, 1: 0, 2: 1, 3: 1,
        4: 0, 5: 0, 6: 1, 7: 1
    }

    age_map = {
        0: 0, 1: 1, 2: 0, 3: 1,
        4: 0, 5: 1, 6: 0, 7: 1
    }

    results_df2 = (
        results_df
        .groupby(["model", "experiment_id", "variant_idx"])
        .agg(
            {"picked_perturbed": "mean"}
        )
        .reset_index()
        .merge(
            experiment_df,
            on="experiment_id"
        )
        .assign(
            black=lambda df: df["variant_idx"].map(race_map),
            female=lambda df: df["variant_idx"].map(gender_map),
            over40=lambda df: df["variant_idx"].map(age_map),
            post=lambda df: df["model"].map(model_phase),
            family=lambda df: df["model"].map(model_family),
            picked_perturbed=lambda df: df["picked_perturbed"].astype(float),
            # perturbed_first=lambda df: 1 - df["perturbed_idx"] * df["ascending"],
            intersection_group=lambda df: df["variant_idx"].map(intersection_map),
        )
        .assign(
            picked_perturbed=lambda df: normalize_outcome(df["picked_perturbed"], version)
        )
    )

    results_df2
    
    return results_df2


def main(version: str, n_trials: int = N_TRIALS) -> None:
    experiments = load_experiments(version)
    results_df = aggregate_results(version, n_trials)
    prompts_path = write_experiment_variant_prompts(experiments, version)
    
    print(results_df.columns)
    print(results_df.head())
    
    results_path = output_data_path(f"results_{version}.csv.gz")
    results_df.to_csv(
        results_path,
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    print(f"Wrote prompts to {prompts_path}")
    print(f"Wrote results to {results_path}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare model outputs for analysis.")
    add_version_argument(parser)
    parser.add_argument("--n-trials", type=int, default=N_TRIALS)
    args = parser.parse_args()
    main(args.version, args.n_trials)
    
    

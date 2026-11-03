# Data and Artifact Documentation

## Release inventory

The release contains five generator distributions in both canonical pickle and
portable gzip-compressed JSON form.

| Distribution | Top-level records | Contents | Source class |
| --- | ---: | --- | --- |
| Occupation | 867 | Degree, skill, transition, and experience weights by SOC | Aggregate proprietary person histories |
| Vacancy | 260 | Company, title, skill, and log-salary parameters by SOC | Aggregate proprietary vacancy records |
| Degree-major | 350 | University weights conditional on degree major | Aggregate proprietary education histories |
| First-name | 145 | First-name weights by birth year and gender | U.S. Social Security baby names |
| Last-name | 2 fields | Surname choices and frequencies | 2010 U.S. Census surnames |

The JSON schema is in `experiment_cache/distributions/schema.json`. The pickle
files contain dictionaries of primitive values and are the exact pipeline
inputs. The JSON files use string object keys, including birth years.

## Public source files

- `raw_data/names/`: U.S. Social Security baby-name files and their upstream
  readme. Source: <https://www.ssa.gov/oact/babynames/limits.html>.
- `raw_data/Names_2010Census.csv`: 2010 Census surname frequencies. Source:
  <https://www.census.gov/topics/population/genealogy/data/2010_surnames.html>.
- `raw_data/Education, Training, and Experience.xlsx`: O*NET education,
  training, and experience ratings used to identify bachelor-level
  occupations. Source and terms: <https://www.onetcenter.org/database.html>.
- `raw_data/audit_study_names.csv`: 1,000 unique race-, gender-, first-name-,
  and last-name combinations derived from the public replication data for
  Kline, Rose, and Walters, *A Discrimination Report Card*. The row-level
  application and callback data are not redistributed. Replication citation:
  <https://doi.org/10.3886/E198284V1>.

Public-source files remain subject to their upstream terms. Their inclusion in
this repository does not relicense them under MIT.

## Proprietary sources

Stage 1 originally queried proprietary person-history, education, skill,
experience, and job-vacancy tables. Those row-level records and posting text are
not included. The release contains only aggregate choice/frequency lists and
log-salary parameters produced from those records. No person identifiers,
application outcomes, posting descriptions, or source rows are present in the
released distributions.

Rebuilding the aggregate distributions requires valid Snowflake credentials,
the table variables in `.env.example`, and permission to use the underlying
sources. Users without access should begin from the committed distributions and
experiment definitions.

## Experiment and model artifacts

Each version contains one `experiments.pkl`, 22 model directories, and 10 trial
files per model. `v1` stores paired-choice outcomes; `v1_secondary` stores 1-10
ratings. Stage 3 maps secondary ratings using `(rating - 1) / 10`, matching the
published analysis, and stages 5-6 convert coefficients back to rating points
for display.

Model outputs may be subject to the terms of their respective model providers.
They are released for research replication and should not be treated as model
weights or as endorsements by those providers.

## Integrity

`release_manifest.json` records every released data/artifact path, byte size,
SHA-256 checksum, distribution count, model list, trial count, generation-seed
status, and structured provenance and license status. Rebuild it with:

```bash
uv run python scripts/build_release_artifacts.py
```

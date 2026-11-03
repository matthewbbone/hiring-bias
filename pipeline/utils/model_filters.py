from __future__ import annotations

import pandas as pd


OLMO_APPENDIX_ONLY_MODELS = {
    "olmo_32b_sft",
    "olmo_32b_dpo",
}


def exclude_olmo_appendix_only_models(
    df: pd.DataFrame,
    model_column: str = "model",
) -> pd.DataFrame:
    if model_column not in df.columns:
        return df.copy()
    return df.loc[~df[model_column].isin(OLMO_APPENDIX_ONLY_MODELS)].copy()

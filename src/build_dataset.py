"""Build the analysis dataset from raw FRED series."""

from pathlib import Path

import numpy as np
import pandas as pd

from .fred_client import FREDClient


def merge_series(series_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Merge all series into a single DataFrame.

    Args:
        series_data: Dictionary mapping series ID to DataFrame

    Returns:
        Merged DataFrame with date as index
    """
    dfs = list(series_data.values())
    merged = dfs[0]
    for df in dfs[1:]:
        merged = merged.merge(df, on="date", how="outer")
    merged = merged.sort_values("date").reset_index(drop=True)
    return merged


def compute_transformations(df: pd.DataFrame) -> pd.DataFrame:
    """Compute productivity growth and factor share transformations.

    Transformations:
    - prod_yoy_pct: 100 * (ln(OPHNFB_t) - ln(OPHNFB_{t-4}))
    - profit_share_pct: 100 * CPROFIT / GDP
    - wage_share_pct: 100 * COE / GDP
    - d_profit_share_yoy_pp: profit_share_pct - profit_share_pct.shift(4)
    - d_wage_share_yoy_pp: wage_share_pct - wage_share_pct.shift(4)

    Args:
        df: DataFrame with OPHNFB, GDP, CPROFIT, COE columns

    Returns:
        DataFrame with computed transformations
    """
    result = df.copy()

    result["prod_yoy_pct"] = 100 * (
        np.log(result["OPHNFB"]) - np.log(result["OPHNFB"].shift(4))
    )

    result["profit_share_pct"] = 100 * result["CPROFIT"] / result["GDP"]
    result["wage_share_pct"] = 100 * result["COE"] / result["GDP"]

    result["d_profit_share_yoy_pp"] = (
        result["profit_share_pct"] - result["profit_share_pct"].shift(4)
    )
    result["d_wage_share_yoy_pp"] = (
        result["wage_share_pct"] - result["wage_share_pct"].shift(4)
    )

    return result


def build_analysis_dataset(
    no_network: bool = False,
    cache_dir: str = "data/raw",
    output_dir: str = "data/processed",
) -> pd.DataFrame:
    """Build the complete analysis dataset.

    Args:
        no_network: If True, use only cached data
        cache_dir: Directory for raw FRED data cache
        output_dir: Directory for processed output

    Returns:
        DataFrame ready for analysis
    """
    client = FREDClient(cache_dir=cache_dir, no_network=no_network)
    series_data = client.get_all_series()

    for series_id, df in series_data.items():
        client.validate_series(df, series_id)

    merged = merge_series(series_data)
    transformed = compute_transformations(merged)

    output_cols = [
        "date",
        "prod_yoy_pct",
        "d_wage_share_yoy_pp",
        "wage_share_pct",
        "d_profit_share_yoy_pp",
        "profit_share_pct",
    ]
    analysis_df = transformed[output_cols].dropna().reset_index(drop=True)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / "dshares_vs_prod.csv"
    analysis_df.to_csv(output_file, index=False)

    return analysis_df

"""Stationarity tests for time series variables."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss


@dataclass
class StationarityResult:
    """Results from ADF and KPSS stationarity tests."""

    variable: str
    adf_stat: float
    adf_pvalue: float
    adf_critical_1pct: float
    adf_critical_5pct: float
    kpss_stat: float
    kpss_pvalue: float
    kpss_critical_5pct: float
    is_stationary: bool


def run_adf_test(series: pd.Series) -> tuple[float, float, float, float]:
    """Run Augmented Dickey-Fuller test.

    Null hypothesis: The series has a unit root (non-stationary).
    Reject null (p < 0.05) to conclude stationarity.

    Args:
        series: Time series to test

    Returns:
        Tuple of (adf_stat, p_value, critical_1pct, critical_5pct)
    """
    result = adfuller(series.dropna(), autolag="AIC")
    adf_stat = result[0]
    p_value = result[1]
    critical_values = result[4]
    return adf_stat, p_value, critical_values["1%"], critical_values["5%"]


def run_kpss_test(series: pd.Series) -> tuple[float, float, float]:
    """Run KPSS test.

    Null hypothesis: The series is stationary.
    Fail to reject null (p > 0.05) to conclude stationarity.

    Args:
        series: Time series to test

    Returns:
        Tuple of (kpss_stat, p_value, critical_5pct)
    """
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = kpss(series.dropna(), regression="c", nlags="auto")
    kpss_stat = result[0]
    p_value = result[1]
    critical_values = result[3]
    return kpss_stat, p_value, critical_values["5%"]


def run_stationarity_tests(df: pd.DataFrame) -> dict[str, StationarityResult]:
    """Run ADF and KPSS tests on key transformed variables.

    Tests the three main variables used in regressions:
    - prod_yoy_pct: Year-over-year productivity growth (%)
    - d_profit_share_yoy_pp: Y/Y change in profit share (pp)
    - d_wage_share_yoy_pp: Y/Y change in wage share (pp)

    A variable is considered stationary if:
    - ADF rejects null (p < 0.05): rejects unit root
    - KPSS fails to reject null (p > 0.05): fails to reject stationarity

    Args:
        df: DataFrame with the three variables

    Returns:
        Dictionary mapping variable names to StationarityResult
    """
    variables = ["prod_yoy_pct", "d_profit_share_yoy_pp", "d_wage_share_yoy_pp"]
    results = {}

    for var in variables:
        series = df[var]

        adf_stat, adf_pvalue, adf_crit_1, adf_crit_5 = run_adf_test(series)
        kpss_stat, kpss_pvalue, kpss_crit_5 = run_kpss_test(series)

        # Stationary if ADF rejects (p < 0.05) AND KPSS fails to reject (p > 0.05)
        is_stationary = bool((adf_pvalue < 0.05) and (kpss_pvalue > 0.05))

        results[var] = StationarityResult(
            variable=var,
            adf_stat=adf_stat,
            adf_pvalue=adf_pvalue,
            adf_critical_1pct=adf_crit_1,
            adf_critical_5pct=adf_crit_5,
            kpss_stat=kpss_stat,
            kpss_pvalue=kpss_pvalue,
            kpss_critical_5pct=kpss_crit_5,
            is_stationary=is_stationary,
        )

    return results


def export_stationarity_results(
    results: dict[str, StationarityResult],
    output_dir: str = "results",
) -> tuple[Path, Path]:
    """Export stationarity test results to JSON and CSV.

    Args:
        results: Dictionary of StationarityResult
        output_dir: Directory for output files

    Returns:
        Tuple of (json_path, csv_path)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Export to JSON
    results_dict = {k: asdict(v) for k, v in results.items()}
    json_path = output_path / "stationarity_tests.json"
    with open(json_path, "w") as f:
        json.dump(results_dict, f, indent=2)

    # Export to CSV
    rows = [asdict(result) for result in results.values()]
    csv_df = pd.DataFrame(rows)
    csv_path = output_path / "stationarity_tests.csv"
    csv_df.to_csv(csv_path, index=False)

    return json_path, csv_path

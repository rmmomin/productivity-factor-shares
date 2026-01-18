"""OLS regression with HAC standard errors."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


@dataclass
class RegressionResult:
    """Results from OLS regression with HAC standard errors."""

    dependent_var: str
    intercept: float
    slope: float
    t_hac: float
    r2: float
    correlation: float
    n_obs: int
    maxlags: int


def fit_ols_hac(
    x: np.ndarray,
    y: np.ndarray,
    maxlags: int = 4,
    dependent_var: str = "y",
) -> RegressionResult:
    """Fit OLS regression and compute HAC (Newey-West) standard errors.

    Args:
        x: Independent variable (productivity growth)
        y: Dependent variable (factor share change)
        maxlags: Maximum lags for HAC estimation (default 4 = 1 year quarterly)
        dependent_var: Name of dependent variable

    Returns:
        RegressionResult with OLS coefficients and HAC t-statistic
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    X = sm.add_constant(x)
    ols = sm.OLS(y, X).fit()
    hac = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})

    intercept, slope = ols.params
    t_hac = hac.tvalues[1]
    r2 = ols.rsquared
    corr = np.corrcoef(x, y)[0, 1]

    return RegressionResult(
        dependent_var=dependent_var,
        intercept=intercept,
        slope=slope,
        t_hac=t_hac,
        r2=r2,
        correlation=corr,
        n_obs=len(x),
        maxlags=maxlags,
    )


def run_regressions(
    df: pd.DataFrame,
    maxlags: int = 4,
) -> dict[str, RegressionResult]:
    """Run regressions for profit share and wage share.

    Args:
        df: DataFrame with prod_yoy_pct, d_profit_share_yoy_pp, d_wage_share_yoy_pp
        maxlags: Maximum lags for HAC estimation

    Returns:
        Dictionary with 'profit' and 'wage' RegressionResults
    """
    x = df["prod_yoy_pct"].values

    profit_result = fit_ols_hac(
        x,
        df["d_profit_share_yoy_pp"].values,
        maxlags=maxlags,
        dependent_var="d_profit_share_yoy_pp",
    )

    wage_result = fit_ols_hac(
        x,
        df["d_wage_share_yoy_pp"].values,
        maxlags=maxlags,
        dependent_var="d_wage_share_yoy_pp",
    )

    return {"profit": profit_result, "wage": wage_result}


def export_results(
    results: dict[str, RegressionResult],
    output_dir: str = "results",
) -> tuple[Path, Path]:
    """Export regression results to JSON and CSV.

    Args:
        results: Dictionary of RegressionResults
        output_dir: Directory for output files

    Returns:
        Tuple of (json_path, csv_path)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results_dict = {k: asdict(v) for k, v in results.items()}
    json_path = output_path / "regression_summary.json"
    with open(json_path, "w") as f:
        json.dump(results_dict, f, indent=2)

    rows = []
    for name, result in results.items():
        row = {"regression": name, **asdict(result)}
        rows.append(row)
    csv_df = pd.DataFrame(rows)
    csv_path = output_path / "regression_summary.csv"
    csv_df.to_csv(csv_path, index=False)

    return json_path, csv_path

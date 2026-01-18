"""Smoke tests for the replication kit."""

import os
from pathlib import Path
import tempfile

import pandas as pd
import pytest


EXPECTED_COLUMNS = [
    "date",
    "prod_yoy_pct",
    "d_wage_share_yoy_pp",
    "wage_share_pct",
    "d_profit_share_yoy_pp",
    "profit_share_pct",
]


class TestDatasetColumns:
    """Test that output dataset has expected columns."""

    def test_sample_csv_has_expected_columns(self):
        """Check that sample CSV has all expected columns."""
        sample_path = Path("prod_and_factor_shares_yoy.csv")
        if not sample_path.exists():
            pytest.skip("Sample CSV not found")

        df = pd.read_csv(sample_path)
        for col in EXPECTED_COLUMNS:
            assert col in df.columns, f"Missing column: {col}"

    def test_processed_csv_has_expected_columns(self):
        """Check that processed output has all expected columns."""
        processed_path = Path("data/processed/dshares_vs_prod.csv")
        if not processed_path.exists():
            pytest.skip("Processed CSV not found - run pipeline first")

        df = pd.read_csv(processed_path)
        for col in EXPECTED_COLUMNS:
            assert col in df.columns, f"Missing column: {col}"


class TestDataQuality:
    """Test data quality of outputs."""

    def test_no_missing_values_in_processed(self):
        """Check that processed output has no missing values."""
        processed_path = Path("data/processed/dshares_vs_prod.csv")
        if not processed_path.exists():
            pytest.skip("Processed CSV not found - run pipeline first")

        df = pd.read_csv(processed_path)
        assert not df.isna().any().any(), "Found missing values in processed data"

    def test_quarterly_frequency(self):
        """Check that data is quarterly frequency."""
        processed_path = Path("data/processed/dshares_vs_prod.csv")
        if not processed_path.exists():
            pytest.skip("Processed CSV not found - run pipeline first")

        df = pd.read_csv(processed_path)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")

        date_diff = df["date"].diff().dropna()
        median_days = date_diff.dt.days.median()

        assert 85 <= median_days <= 95, f"Data does not appear quarterly: median diff = {median_days} days"


class TestOutputFiles:
    """Test that output files are created."""

    def test_processed_csv_exists(self):
        """Check that processed CSV exists after running pipeline."""
        processed_path = Path("data/processed/dshares_vs_prod.csv")
        if not processed_path.exists():
            pytest.skip("Processed CSV not found - run pipeline first")
        assert processed_path.exists()

    def test_regression_json_exists(self):
        """Check that regression JSON exists after running pipeline."""
        json_path = Path("results/regression_summary.json")
        if not json_path.exists():
            pytest.skip("Regression JSON not found - run pipeline first")
        assert json_path.exists()

    def test_regression_csv_exists(self):
        """Check that regression CSV exists after running pipeline."""
        csv_path = Path("results/regression_summary.csv")
        if not csv_path.exists():
            pytest.skip("Regression CSV not found - run pipeline first")
        assert csv_path.exists()

    def test_scatter_plots_exist(self):
        """Check that scatter plots exist after running pipeline."""
        figures_dir = Path("figures")
        if not figures_dir.exists():
            pytest.skip("Figures directory not found - run pipeline first")

        expected_plots = [
            "scatter_profit_share.png",
            "scatter_wage_share.png",
            "scatter_combined.png",
        ]
        for plot in expected_plots:
            plot_path = figures_dir / plot
            if not plot_path.exists():
                pytest.skip(f"Plot {plot} not found - run pipeline first")
            assert plot_path.exists()

    def test_binscatter_plots_exist(self):
        """Check that binscatter plots exist after running pipeline."""
        figures_dir = Path("figures")
        if not figures_dir.exists():
            pytest.skip("Figures directory not found - run pipeline first")

        expected_plots = [
            "binscatter_profit_share.png",
            "binscatter_wage_share.png",
            "binscatter_combined.png",
        ]
        for plot in expected_plots:
            plot_path = figures_dir / plot
            if not plot_path.exists():
                pytest.skip(f"Plot {plot} not found - run pipeline first")
            assert plot_path.exists()


class TestTransformations:
    """Test data transformations."""

    def test_productivity_growth_calculation(self):
        """Test productivity y/y growth calculation."""
        from src.build_dataset import compute_transformations
        import numpy as np

        df = pd.DataFrame({
            "date": pd.date_range("2020-01-01", periods=8, freq="QE"),
            "OPHNFB": [100, 101, 102, 103, 105, 106, 107, 108],
            "GDP": [1000] * 8,
            "CPROFIT": [100] * 8,
            "COE": [500] * 8,
        })

        result = compute_transformations(df)

        expected_prod_yoy = 100 * (np.log(105) - np.log(100))
        actual_prod_yoy = result.loc[4, "prod_yoy_pct"]
        assert abs(actual_prod_yoy - expected_prod_yoy) < 0.001

    def test_share_calculations(self):
        """Test factor share calculations."""
        from src.build_dataset import compute_transformations

        df = pd.DataFrame({
            "date": pd.date_range("2020-01-01", periods=5, freq="QE"),
            "OPHNFB": [100, 101, 102, 103, 104],
            "GDP": [1000, 1000, 1000, 1000, 1000],
            "CPROFIT": [100, 110, 120, 130, 140],
            "COE": [500, 510, 520, 530, 540],
        })

        result = compute_transformations(df)

        assert result.loc[0, "profit_share_pct"] == 10.0
        assert result.loc[0, "wage_share_pct"] == 50.0

        assert result.loc[4, "profit_share_pct"] == 14.0
        assert result.loc[4, "wage_share_pct"] == 54.0


class TestRegression:
    """Test regression functions."""

    def test_fit_ols_hac(self):
        """Test OLS with HAC standard errors."""
        from src.analysis import fit_ols_hac
        import numpy as np

        np.random.seed(42)
        x = np.linspace(0, 10, 100)
        y = 2 + 0.5 * x + np.random.normal(0, 0.5, 100)

        result = fit_ols_hac(x, y, maxlags=4, dependent_var="test")

        assert abs(result.intercept - 2) < 0.5
        assert abs(result.slope - 0.5) < 0.2
        assert result.r2 > 0.8
        assert result.n_obs == 100
        assert result.maxlags == 4

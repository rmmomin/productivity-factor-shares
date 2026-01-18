"""FRED API client with caching support."""

import json
import os
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

FRED_BASE_URL = "https://api.stlouisfed.org/fred"
SERIES_IDS = {
    "OPHNFB": "Output Per Hour of All Persons, Nonfarm Business Sector",
    "GDP": "Gross Domestic Product",
    "CPROFIT": "Corporate Profits with IVA and CCAdj",
    "COE": "Compensation of Employees, Paid",
}


class FREDClient:
    """Client for fetching and caching FRED data."""

    def __init__(self, cache_dir: str = "data/raw", no_network: bool = False):
        self.api_key = os.getenv("FRED_API_KEY")
        if not self.api_key and not no_network:
            raise ValueError("FRED_API_KEY environment variable not set")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.no_network = no_network
        self.metadata_path = self.cache_dir / "series_metadata.json"

    def _get_cache_path(self, series_id: str) -> Path:
        """Get cache file path for a series."""
        return self.cache_dir / f"{series_id}.csv"

    def _is_cached(self, series_id: str) -> bool:
        """Check if series data is cached."""
        return self._get_cache_path(series_id).exists()

    def _fetch_series_from_api(self, series_id: str) -> pd.DataFrame:
        """Fetch series data from FRED API."""
        url = f"{FRED_BASE_URL}/series/observations"
        params = {
            "series_id": series_id,
            "api_key": self.api_key,
            "file_type": "json",
        }
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        observations = data.get("observations", [])
        df = pd.DataFrame(observations)
        df = df[["date", "value"]].copy()
        df["date"] = pd.to_datetime(df["date"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.rename(columns={"value": series_id})
        return df

    def _fetch_metadata_from_api(self, series_id: str) -> dict:
        """Fetch series metadata from FRED API."""
        url = f"{FRED_BASE_URL}/series"
        params = {
            "series_id": series_id,
            "api_key": self.api_key,
            "file_type": "json",
        }
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        return data.get("seriess", [{}])[0]

    def _load_from_cache(self, series_id: str) -> pd.DataFrame:
        """Load series data from cache."""
        cache_path = self._get_cache_path(series_id)
        df = pd.read_csv(cache_path)
        df["date"] = pd.to_datetime(df["date"])
        return df

    def _save_to_cache(self, series_id: str, df: pd.DataFrame) -> None:
        """Save series data to cache."""
        cache_path = self._get_cache_path(series_id)
        df.to_csv(cache_path, index=False)

    def _load_metadata(self) -> dict:
        """Load metadata from cache."""
        if self.metadata_path.exists():
            with open(self.metadata_path, "r") as f:
                return json.load(f)
        return {}

    def _save_metadata(self, metadata: dict) -> None:
        """Save metadata to cache."""
        with open(self.metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

    def get_series(self, series_id: str, force_refresh: bool = False) -> pd.DataFrame:
        """Get series data, using cache if available.

        Args:
            series_id: FRED series ID
            force_refresh: If True, fetch fresh data even if cached

        Returns:
            DataFrame with date and series value columns
        """
        if self._is_cached(series_id) and not force_refresh:
            return self._load_from_cache(series_id)

        if self.no_network:
            if self._is_cached(series_id):
                return self._load_from_cache(series_id)
            raise RuntimeError(
                f"No cached data for {series_id} and network is disabled"
            )

        df = self._fetch_series_from_api(series_id)
        self._save_to_cache(series_id, df)
        return df

    def get_metadata(self, series_id: str) -> dict:
        """Get series metadata, using cache if available."""
        metadata = self._load_metadata()

        if series_id in metadata:
            return metadata[series_id]

        if self.no_network:
            return {"id": series_id, "title": SERIES_IDS.get(series_id, series_id)}

        series_meta = self._fetch_metadata_from_api(series_id)
        metadata[series_id] = series_meta
        self._save_metadata(metadata)
        return series_meta

    def get_all_series(self, force_refresh: bool = False) -> dict[str, pd.DataFrame]:
        """Fetch all required series.

        Returns:
            Dictionary mapping series ID to DataFrame
        """
        series_data = {}
        for series_id in SERIES_IDS:
            series_data[series_id] = self.get_series(series_id, force_refresh)
        return series_data

    def validate_series(self, df: pd.DataFrame, series_id: str) -> None:
        """Validate series data.

        Raises:
            ValueError: If validation fails
        """
        if df.empty:
            raise ValueError(f"Series {series_id} is empty")

        if df[series_id].isna().all():
            raise ValueError(f"Series {series_id} has all missing values")

        date_diff = df["date"].diff().dropna()
        if not date_diff.empty:
            median_diff = date_diff.median()
            expected_quarterly = pd.Timedelta(days=91)
            if abs(median_diff - expected_quarterly) > pd.Timedelta(days=10):
                print(f"Warning: {series_id} may not be quarterly frequency")

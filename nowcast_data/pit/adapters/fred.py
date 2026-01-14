"""FRED/ALFRED adapter for US macroeconomic data."""

import os
import time
from datetime import date, datetime
from typing import Optional, List
import requests
import pandas as pd

from nowcast_data.pit.adapters.base import PITAdapter
from nowcast_data.pit.core.models import PITObservation
from nowcast_data.pit.exceptions import (
    PITNotSupportedError,
    VintageNotFoundError,
    SourceFetchError,
)
from nowcast_data.pit.core.vintage_logic import select_vintage_for_asof


class FREDALFREDAdapter(PITAdapter):
    """
    Adapter for FRED/ALFRED (Federal Reserve Economic Data).
    
    Uses the FRED API to retrieve point-in-time data using realtime_start/realtime_end.
    """
    
    BASE_URL = "https://api.stlouisfed.org/fred"
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize FRED adapter.
        
        Args:
            api_key: FRED API key. If None, reads from FRED_API_KEY env var.
        """
        self._api_key = api_key or os.getenv("FRED_API_KEY")
        if not self._api_key:
            raise ValueError(
                "FRED API key required. Set FRED_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        self._session = requests.Session()
        self._vintage_cache = {}  # Cache vintage dates per series
        self._cache_ttl = 86400  # 24 hours
    
    @property
    def name(self) -> str:
        return "FRED_ALFRED"
    
    def supports_pit(self, series_id: str) -> bool:
        """Check if series has vintage dates (supports PIT)."""
        try:
            vintages = self.list_vintages(series_id)
            return len(vintages) > 0
        except Exception:
            return False
    
    def list_vintages(self, series_id: str) -> List[date]:
        """
        List available vintage dates from FRED.
        
        Uses the series/vintagedates endpoint.
        """
        # Check cache
        cache_key = series_id
        if cache_key in self._vintage_cache:
            cached_data, cached_time = self._vintage_cache[cache_key]
            if time.time() - cached_time < self._cache_ttl:
                return cached_data
        
        url = f"{self.BASE_URL}/series/vintagedates"
        params = {
            "series_id": series_id,
            "api_key": self._api_key,
            "file_type": "json",
        }
        
        try:
            response = self._request_with_retry(url, params)
            data = response.json()
            
            vintage_dates_str = data.get("vintage_dates", [])
            if not vintage_dates_str:
                # No vintages = not a revised series
                vintages = []
            else:
                vintages = [
                    datetime.strptime(v, "%Y-%m-%d").date()
                    for v in vintage_dates_str
                ]
            
            # Cache result
            self._vintage_cache[cache_key] = (vintages, time.time())
            
            return vintages
            
        except Exception as e:
            raise SourceFetchError("FRED_ALFRED", original_error=e)
    
    def fetch_asof(
        self,
        series_id: str,
        asof_date: date,
        start: Optional[date] = None,
        end: Optional[date] = None
    ) -> List[PITObservation]:
        """
        Fetch observations as of asof_date using FRED realtime API.
        
        Sets realtime_start = realtime_end = asof_date to get the vintage
        that was available on that date.
        """
        # Check if series supports PIT
        vintages = self.list_vintages(series_id)
        if not vintages:
            raise PITNotSupportedError(series_id, "No vintages available (non-revised series)")
        
        # Check if asof_date is valid
        selected_vintage = select_vintage_for_asof(vintages, asof_date)
        if selected_vintage is None:
            raise VintageNotFoundError(series_id, asof_date)
        
        # Fetch observations with realtime parameters
        url = f"{self.BASE_URL}/series/observations"
        params = {
            "series_id": series_id,
            "api_key": self._api_key,
            "file_type": "json",
            "realtime_start": asof_date.strftime("%Y-%m-%d"),
            "realtime_end": asof_date.strftime("%Y-%m-%d"),
        }
        
        if start:
            params["observation_start"] = start.strftime("%Y-%m-%d")
        if end:
            params["observation_end"] = end.strftime("%Y-%m-%d")
        
        try:
            response = self._request_with_retry(url, params)
            data = response.json()
            
            observations = []
            for obs_dict in data.get("observations", []):
                obs_date_str = obs_dict["date"]
                value_str = obs_dict["value"]
                
                # FRED uses "." for missing values
                if value_str == ".":
                    continue
                
                try:
                    value = float(value_str)
                except (ValueError, TypeError):
                    continue
                
                obs = PITObservation(
                    series_key=series_id,
                    source="FRED_ALFRED",
                    source_series_id=series_id,
                    asof_date=asof_date,
                    vintage_date=asof_date,  # For ALFRED, vintage = asof for snapshot queries
                    obs_date=datetime.strptime(obs_date_str, "%Y-%m-%d").date(),
                    value=value,
                    value_raw=value_str,
                    frequency=self._infer_frequency(obs_date_str),
                    realtime_start=asof_date,
                    realtime_end=asof_date,
                    ingested_at=datetime.utcnow(),
                    provenance={"realtime_start": obs_dict.get("realtime_start")},
                )
                observations.append(obs)
            
            return observations
            
        except SourceFetchError:
            raise
        except Exception as e:
            raise SourceFetchError("FRED_ALFRED", original_error=e)
    
    def _request_with_retry(
        self, url: str, params: dict, max_retries: int = 3
    ) -> requests.Response:
        """Make HTTP request with exponential backoff retry."""
        for attempt in range(max_retries):
            try:
                response = self._session.get(url, params=params, timeout=30)
                response.raise_for_status()
                return response
            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    raise
                wait_time = 2 ** attempt
                time.sleep(wait_time)
        
        raise SourceFetchError("FRED_ALFRED", "Max retries exceeded")
    
    def _infer_frequency(self, date_str: str) -> str:
        """Infer frequency from date string format."""
        # Simple heuristic - FRED typically uses period end dates
        # Real implementation should query series metadata
        return "M"  # Default to monthly

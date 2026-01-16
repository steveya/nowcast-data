"""Main PIT data retrieval API facade."""

import os
from datetime import date
from typing import Optional, List, Dict
import pandas as pd
from dotenv import load_dotenv

from nowcast_data.pit.adapters.base import PITAdapter
from nowcast_data.pit.adapters.fred import FREDALFREDAdapter
from nowcast_data.pit.adapters.ecb import ECBRTDBAdapter
from nowcast_data.pit.adapters.boe import BOERTDBAdapter
from nowcast_data.pit.adapters.statcan import StatCanRealTimeAdapter
from nowcast_data.pit.adapters.swiss import SwissAdapter
from nowcast_data.pit.adapters.alphaforge import AlphaForgePITAdapter
from nowcast_data.pit.core.catalog import SeriesCatalog
from nowcast_data.pit.core.models import create_pit_dataframe, create_wide_view
from nowcast_data.pit.exceptions import PITNotSupportedError

load_dotenv()


class PITDataManager:
    """
    Main interface for point-in-time data retrieval.
    
    Provides high-level methods for getting series and panels as-of specific dates.
    """
    
    def __init__(self, catalog: SeriesCatalog, adapters: Optional[Dict[str, PITAdapter]] = None):
        """
        Initialize PIT data manager.
        
        Args:
            catalog: Series catalog with metadata
            adapters: Dictionary mapping adapter names to adapter instances.
                     If None, creates default adapters.
        """
        self.catalog = catalog
        
        if adapters is None:
            # Create default adapters
            self.adapters = {}
            # FRED is the only one requiring API key, others are stubs
            try:
                fred_api_key = os.environ.get("FRED_API_KEY")
                if fred_api_key:
                    self.adapters["FRED_ALFRED"] = FREDALFREDAdapter(api_key=fred_api_key)
                    self.adapters["alphaforge"] = AlphaForgePITAdapter(fred_api_key=fred_api_key)
                else:
                    print("Warning: FRED_API_KEY environment variable not set. FRED and AlphaForge adapters will not be available.")
            except ValueError:
                # API key not available, skip FRED
                pass
            
            self.adapters["ECB_RTDB"] = ECBRTDBAdapter()
            self.adapters["BOE_RTDB"] = BOERTDBAdapter()
            self.adapters["STATCAN_REALTIME"] = StatCanRealTimeAdapter()
            self.adapters["SWISS_SECO"] = SwissAdapter()
        else:
            self.adapters = adapters
    
    def get_series_asof(
        self,
        series_key: str,
        asof_date: date,
        start: Optional[date] = None,
        end: Optional[date] = None
    ) -> pd.DataFrame:
        """
        Get a single series as-of a specific date.
        
        Args:
            series_key: Canonical series identifier
            asof_date: Point-in-time evaluation date
            start: Optional start date for observations
            end: Optional end date for observations
            
        Returns:
            DataFrame with columns: obs_date, value, (and metadata columns)
            
        Raises:
            PITNotSupportedError: If series doesn't support PIT
            VintageNotFoundError: If no vintage available
            SourceFetchError: If fetch fails
        """
        # Get series metadata
        metadata = self.catalog.get(series_key)
        if not metadata:
            raise ValueError(f"Series '{series_key}' not found in catalog")
        
        # Check PIT support
        if metadata.pit_mode == "NO_PIT":
            raise PITNotSupportedError(series_key)
        
        # Get appropriate adapter
        adapter_name = getattr(metadata, 'adapter', metadata.source)
        adapter = self.adapters.get(adapter_name)
        if not adapter:
            raise ValueError(f"No adapter available for source '{adapter_name}'")
        
        # Fetch data
        observations = adapter.fetch_asof(
            metadata.source_series_id,
            asof_date,
            start,
            end
        )
        
        # Convert to DataFrame
        df = create_pit_dataframe(observations)
        
        return df
    
    def get_panel_asof(
        self,
        series_keys: List[str],
        asof_date: date,
        start: Optional[date] = None,
        end: Optional[date] = None,
        wide: bool = True
    ) -> pd.DataFrame:
        """
        Get multiple series as-of a specific date (panel).
        
        Args:
            series_keys: List of canonical series identifiers
            asof_date: Point-in-time evaluation date
            start: Optional start date for observations
            end: Optional end date for observations
            wide: If True, return wide format (obs_date x series_key).
                 If False, return long format.
            
        Returns:
            DataFrame in wide or long format
            
        Raises:
            PITNotSupportedError: If any series doesn't support PIT
            VintageNotFoundError: If no vintage available for any series
            SourceFetchError: If any fetch fails
        """
        all_observations = []
        
        for series_key in series_keys:
            try:
                df = self.get_series_asof(series_key, asof_date, start, end)
                all_observations.append(df)
            except Exception as e:
                # Log error but continue with other series
                print(f"Warning: Failed to fetch {series_key}: {e}")
                continue
        
        if not all_observations:
            return pd.DataFrame()
        
        # Combine all series
        combined = pd.concat(all_observations, ignore_index=True)
        
        if wide:
            return create_wide_view(combined)
        else:
            return combined
    
    def get_series_vintages(self, series_key: str) -> List[date]:
        """
        List available vintages for a series.
        
        Args:
            series_key: Canonical series identifier
            
        Returns:
            List of vintage dates
            
        Raises:
            PITNotSupportedError: If series doesn't support vintages
        """
        metadata = self.catalog.get(series_key)
        if not metadata:
            raise ValueError(f"Series '{series_key}' not found in catalog")
        
        if metadata.pit_mode == "NO_PIT":
            raise PITNotSupportedError(series_key)
        
        adapter = self.adapters.get(metadata.source)
        if not adapter:
            raise ValueError(f"No adapter available for source '{metadata.source}'")
        
        return adapter.list_vintages(metadata.source_series_id)
    
    def get_panel_vintages(self, series_keys: List[str]) -> Dict[str, List[date]]:
        """
        Get vintages for multiple series.
        
        Args:
            series_keys: List of canonical series identifiers
            
        Returns:
            Dictionary mapping series_key to list of vintage dates
        """
        result = {}
        
        for series_key in series_keys:
            try:
                vintages = self.get_series_vintages(series_key)
                result[series_key] = vintages
            except Exception as e:
                print(f"Warning: Failed to get vintages for {series_key}: {e}")
                result[series_key] = []
        
        return result
    
    def build_pit_cube(
        self,
        series_keys: List[str],
        asof_dates: List[date],
        start: date,
        end: date
    ) -> pd.DataFrame:
        """
        Build a PIT cube: multiple series × multiple asof dates.
        
        Returns a tidy/long DataFrame with all combinations.
        
        Args:
            series_keys: List of series to include
            asof_dates: List of asof evaluation dates
            start: Start date for observation period
            end: End date for observation period
            
        Returns:
            Long DataFrame with schema matching macro_pit_observations
        """
        all_observations = []
        
        total = len(series_keys) * len(asof_dates)
        processed = 0
        
        for series_key in series_keys:
            for asof_date in asof_dates:
                try:
                    df = self.get_series_asof(series_key, asof_date, start, end)
                    all_observations.append(df)
                except Exception as e:
                    print(f"Warning: Failed to fetch {series_key} @ {asof_date}: {e}")
                
                processed += 1
                if processed % 10 == 0:
                    print(f"Progress: {processed}/{total}")
        
        if not all_observations:
            return pd.DataFrame()
        
        return pd.concat(all_observations, ignore_index=True)

"""Data models and schemas for PIT data storage."""

from dataclasses import dataclass
from datetime import date, datetime
from typing import Optional, Literal, Dict, Any
import pandas as pd


PITMode = Literal["ALFRED_REALTIME", "DISCRETE_VINTAGES_SNAP", "NO_PIT"]


@dataclass
class SeriesMetadata:
    """Metadata for a macro series."""
    
    series_key: str
    country: str
    source: str
    source_series_id: str
    frequency: str
    pit_mode: PITMode
    seasonal_adjustment: Optional[str] = None
    units: Optional[str] = None
    description: Optional[str] = None
    transforms: Optional[list] = None


@dataclass
class PITObservation:
    """A single point-in-time observation."""
    
    series_key: str
    source: str
    source_series_id: str
    asof_date: date
    vintage_date: date
    obs_date: date
    value: float
    frequency: str
    value_raw: Optional[str] = None
    units: Optional[str] = None
    seasonal_adjustment: Optional[str] = None
    realtime_start: Optional[date] = None
    realtime_end: Optional[date] = None
    ingested_at: Optional[datetime] = None
    provenance: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for DataFrame construction."""
        return {
            'series_key': self.series_key,
            'source': self.source,
            'source_series_id': self.source_series_id,
            'asof_date': self.asof_date,
            'vintage_date': self.vintage_date,
            'obs_date': self.obs_date,
            'value': self.value,
            'value_raw': self.value_raw,
            'frequency': self.frequency,
            'units': self.units,
            'seasonal_adjustment': self.seasonal_adjustment,
            'realtime_start': self.realtime_start,
            'realtime_end': self.realtime_end,
            'ingested_at': self.ingested_at,
            'provenance': self.provenance,
        }


def create_pit_dataframe(observations: list[PITObservation]) -> pd.DataFrame:
    """
    Create a canonical PIT DataFrame from observations.
    
    Schema:
    - series_key, source, source_series_id
    - asof_date, vintage_date, obs_date
    - value, value_raw
    - frequency, units, seasonal_adjustment
    - realtime_start, realtime_end
    - ingested_at, provenance
    """
    if not observations:
        return pd.DataFrame(columns=[
            'series_key', 'source', 'source_series_id',
            'asof_date', 'vintage_date', 'obs_date',
            'value', 'value_raw', 'frequency', 'units', 'seasonal_adjustment',
            'realtime_start', 'realtime_end', 'ingested_at', 'provenance'
        ])
    
    data = [obs.to_dict() for obs in observations]
    df = pd.DataFrame(data)
    
    # Ensure date columns are proper datetime types
    date_cols = ['asof_date', 'vintage_date', 'obs_date', 'realtime_start', 'realtime_end']
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    
    if 'ingested_at' in df.columns:
        df['ingested_at'] = pd.to_datetime(df['ingested_at'])
    
    return df


def create_wide_view(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a wide view for modeling from PIT DataFrame.
    
    Args:
        df: PIT DataFrame with single asof_date
        
    Returns:
        Wide DataFrame with:
        - index: obs_date
        - columns: series_key
        - values: value
    """
    if df.empty:
        return pd.DataFrame()
    
    # Check that all rows have the same asof_date
    asof_dates = df['asof_date'].unique()
    if len(asof_dates) > 1:
        raise ValueError(
            f"Wide view requires single asof_date, got {len(asof_dates)} different dates"
        )
    
    # Pivot to wide format
    wide = df.pivot(index='obs_date', columns='series_key', values='value')
    
    return wide

"""Tests for PIT data models and schema."""

from datetime import date

import pandas as pd
import pytest

from nowcast_data.pit.core.models import (
    PITObservation,
    create_pit_dataframe,
    create_wide_view,
)


class TestPITObservation:
    """Tests for PITObservation dataclass."""
    
    def test_create_observation(self):
        """Should create observation with required fields."""
        obs = PITObservation(
            series_key="US_GDP",
            source="FRED_ALFRED",
            source_series_id="GDP",
            asof_date=date(2020, 1, 1),
            vintage_date=date(2020, 1, 1),
            obs_date=date(2019, 12, 31),
            value=21000.0,
            frequency="Q",
        )
        assert obs.series_key == "US_GDP"
        assert obs.value == 21000.0
    
    def test_to_dict(self):
        """Should convert observation to dictionary."""
        obs = PITObservation(
            series_key="US_GDP",
            source="FRED_ALFRED",
            source_series_id="GDP",
            asof_date=date(2020, 1, 1),
            vintage_date=date(2020, 1, 1),
            obs_date=date(2019, 12, 31),
            value=21000.0,
            frequency="Q",
            units="Billions",
        )
        d = obs.to_dict()
        assert d["series_key"] == "US_GDP"
        assert d["value"] == 21000.0
        assert d["units"] == "Billions"


class TestCreatePITDataFrame:
    """Tests for create_pit_dataframe function."""
    
    def test_empty_observations(self):
        """Should create empty DataFrame with correct columns."""
        df = create_pit_dataframe([])
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
        assert "series_key" in df.columns
        assert "value" in df.columns
        assert "obs_date" in df.columns
    
    def test_single_observation(self):
        """Should create DataFrame from single observation."""
        obs = PITObservation(
            series_key="US_GDP",
            source="FRED_ALFRED",
            source_series_id="GDP",
            asof_date=date(2020, 1, 1),
            vintage_date=date(2020, 1, 1),
            obs_date=date(2019, 12, 31),
            value=21000.0,
            frequency="Q",
        )
        df = create_pit_dataframe([obs])
        assert len(df) == 1
        assert df.iloc[0]["series_key"] == "US_GDP"
        assert df.iloc[0]["value"] == 21000.0
    
    def test_multiple_observations(self):
        """Should create DataFrame from multiple observations."""
        obs1 = PITObservation(
            series_key="US_GDP",
            source="FRED_ALFRED",
            source_series_id="GDP",
            asof_date=date(2020, 1, 1),
            vintage_date=date(2020, 1, 1),
            obs_date=date(2019, 9, 30),
            value=21000.0,
            frequency="Q",
        )
        obs2 = PITObservation(
            series_key="US_GDP",
            source="FRED_ALFRED",
            source_series_id="GDP",
            asof_date=date(2020, 1, 1),
            vintage_date=date(2020, 1, 1),
            obs_date=date(2019, 12, 31),
            value=21500.0,
            frequency="Q",
        )
        df = create_pit_dataframe([obs1, obs2])
        assert len(df) == 2


class TestCreateWideView:
    """Tests for create_wide_view function."""
    
    def test_empty_dataframe(self):
        """Should handle empty DataFrame."""
        df = create_pit_dataframe([])
        wide = create_wide_view(df)
        assert isinstance(wide, pd.DataFrame)
        assert len(wide) == 0
    
    def test_single_series(self):
        """Should create wide view for single series."""
        obs1 = PITObservation(
            series_key="US_GDP",
            source="FRED_ALFRED",
            source_series_id="GDP",
            asof_date=date(2020, 1, 1),
            vintage_date=date(2020, 1, 1),
            obs_date=date(2019, 9, 30),
            value=21000.0,
            frequency="Q",
        )
        obs2 = PITObservation(
            series_key="US_GDP",
            source="FRED_ALFRED",
            source_series_id="GDP",
            asof_date=date(2020, 1, 1),
            vintage_date=date(2020, 1, 1),
            obs_date=date(2019, 12, 31),
            value=21500.0,
            frequency="Q",
        )
        df = create_pit_dataframe([obs1, obs2])
        wide = create_wide_view(df)
        
        assert "US_GDP" in wide.columns
        assert len(wide) == 2
    
    def test_multiple_series(self):
        """Should create wide view for multiple series."""
        obs1 = PITObservation(
            series_key="US_GDP",
            source="FRED_ALFRED",
            source_series_id="GDP",
            asof_date=date(2020, 1, 1),
            vintage_date=date(2020, 1, 1),
            obs_date=date(2019, 12, 31),
            value=21000.0,
            frequency="Q",
        )
        obs2 = PITObservation(
            series_key="US_CPI",
            source="FRED_ALFRED",
            source_series_id="CPIAUCSL",
            asof_date=date(2020, 1, 1),
            vintage_date=date(2020, 1, 1),
            obs_date=date(2019, 12, 31),
            value=255.0,
            frequency="M",
        )
        df = create_pit_dataframe([obs1, obs2])
        wide = create_wide_view(df)
        
        assert "US_GDP" in wide.columns
        assert "US_CPI" in wide.columns
    
    def test_multiple_asof_dates_raises(self):
        """Should raise error when multiple asof dates present."""
        obs1 = PITObservation(
            series_key="US_GDP",
            source="FRED_ALFRED",
            source_series_id="GDP",
            asof_date=date(2020, 1, 1),
            vintage_date=date(2020, 1, 1),
            obs_date=date(2019, 12, 31),
            value=21000.0,
            frequency="Q",
        )
        obs2 = PITObservation(
            series_key="US_GDP",
            source="FRED_ALFRED",
            source_series_id="GDP",
            asof_date=date(2020, 2, 1),  # Different asof_date
            vintage_date=date(2020, 2, 1),
            obs_date=date(2019, 12, 31),
            value=21100.0,
            frequency="Q",
        )
        df = create_pit_dataframe([obs1, obs2])
        
        with pytest.raises(ValueError, match="single asof_date"):
            create_wide_view(df)

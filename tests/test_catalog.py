"""Tests for series catalog."""

from pathlib import Path
import tempfile
import pytest

from nowcast_data.pit.core.catalog import SeriesCatalog
from nowcast_data.pit.core.models import SeriesMetadata


class TestSeriesCatalog:
    """Tests for SeriesCatalog class."""
    
    def test_empty_catalog(self):
        """Should create empty catalog."""
        catalog = SeriesCatalog()
        assert len(catalog.get_all()) == 0
    
    def test_add_series(self):
        """Should add series metadata."""
        catalog = SeriesCatalog()
        metadata = SeriesMetadata(
            series_key="US_GDP",
            country="US",
            source="FRED_ALFRED",
            source_series_id="GDP",
            frequency="Q",
            pit_mode="ALFRED_REALTIME",
        )
        catalog.add(metadata)
        
        retrieved = catalog.get("US_GDP")
        assert retrieved is not None
        assert retrieved.series_key == "US_GDP"
        assert retrieved.country == "US"
    
    def test_get_nonexistent(self):
        """Should return None for nonexistent series."""
        catalog = SeriesCatalog()
        result = catalog.get("NONEXISTENT")
        assert result is None
    
    def test_list_series(self):
        """Should list all series."""
        catalog = SeriesCatalog()
        catalog.add(SeriesMetadata(
            series_key="US_GDP",
            country="US",
            source="FRED_ALFRED",
            source_series_id="GDP",
            frequency="Q",
            pit_mode="ALFRED_REALTIME",
        ))
        catalog.add(SeriesMetadata(
            series_key="US_CPI",
            country="US",
            source="FRED_ALFRED",
            source_series_id="CPIAUCSL",
            frequency="M",
            pit_mode="ALFRED_REALTIME",
        ))
        
        series = catalog.list_series()
        assert len(series) == 2
        assert "US_GDP" in series
        assert "US_CPI" in series
    
    def test_list_series_filtered_by_country(self):
        """Should filter series by country."""
        catalog = SeriesCatalog()
        catalog.add(SeriesMetadata(
            series_key="US_GDP",
            country="US",
            source="FRED_ALFRED",
            source_series_id="GDP",
            frequency="Q",
            pit_mode="ALFRED_REALTIME",
        ))
        catalog.add(SeriesMetadata(
            series_key="EA_GDP",
            country="EA",
            source="ECB_RTDB",
            source_series_id="MNA.Q.Y.I8.W2.S1.S1.B.B1GQ._Z._Z._Z.EUR.V.N",
            frequency="Q",
            pit_mode="DISCRETE_VINTAGES_SNAP",
        ))
        
        us_series = catalog.list_series(country="US")
        assert len(us_series) == 1
        assert "US_GDP" in us_series
        
        ea_series = catalog.list_series(country="EA")
        assert len(ea_series) == 1
        assert "EA_GDP" in ea_series
    
    def test_supports_pit(self):
        """Should check PIT support correctly."""
        catalog = SeriesCatalog()
        catalog.add(SeriesMetadata(
            series_key="US_GDP",
            country="US",
            source="FRED_ALFRED",
            source_series_id="GDP",
            frequency="Q",
            pit_mode="ALFRED_REALTIME",
        ))
        catalog.add(SeriesMetadata(
            series_key="CH_GDP",
            country="CH",
            source="SWISS_SECO",
            source_series_id="GDP_CH",
            frequency="Q",
            pit_mode="NO_PIT",
        ))
        
        assert catalog.supports_pit("US_GDP") is True
        assert catalog.supports_pit("CH_GDP") is False
        assert catalog.supports_pit("NONEXISTENT") is False
    
    def test_load_from_yaml(self):
        """Should load catalog from YAML file."""
        yaml_content = """
US_GDP:
  country: US
  source: FRED_ALFRED
  source_series_id: GDP
  frequency: Q
  pit_mode: ALFRED_REALTIME
  units: Billions

US_CPI:
  country: US
  source: FRED_ALFRED
  source_series_id: CPIAUCSL
  frequency: M
  pit_mode: ALFRED_REALTIME
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name
        
        try:
            catalog = SeriesCatalog(Path(temp_path))
            
            assert len(catalog.get_all()) == 2
            
            gdp = catalog.get("US_GDP")
            assert gdp.country == "US"
            assert gdp.frequency == "Q"
            assert gdp.units == "Billions"
            
            cpi = catalog.get("US_CPI")
            assert cpi.frequency == "M"
        finally:
            Path(temp_path).unlink()

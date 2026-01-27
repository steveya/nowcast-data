# Nowcast Data: Point-in-Time Macro Data Library

A Python library for retrieving and managing point-in-time (PIT) / vintage / "as-of" macroeconomic time series data for G6 countries, built on the [alphaforge](https://github.com/steveya/alphaforge) data framework.

## Overview

This library provides infrastructure for accessing historical snapshots of macroeconomic series as they were known on specific dates, enabling proper backtesting without look-ahead bias.

### Key Features

- **Point-in-Time Data Retrieval**: Access macro series as they appeared on specific dates
- **Vintage Management**: Track and retrieve specific vintages of revised data series
- **G6 Coverage**: Support for US, Euro Area, UK, Canada, Switzerland, and Japan
- **Multiple Data Sources**:
  - FRED/ALFRED (US Federal Reserve)
  - ECB Real-Time Database (Euro Area) - stub
  - Bank of England RTDB (UK) - stub
  - Statistics Canada Real-Time Tables - stub
  - Swiss sources (SECO/SNB) - stub
- **Flexible API**: Single series, panels, and cube building
- **No Look-Ahead**: Strict validation to prevent future data leakage

## Installation

```bash
# Clone the repository
git clone https://github.com/steveya/nowcast-data.git
cd nowcast-data

# Install dependencies
pip install -e .

# Optional: Install storage dependencies
pip install -e ".[storage]"

# Optional: Install development dependencies
pip install -e ".[dev]"
```

## Quick Start

### 1. PIT Store Setup

AlphaForge is used as the PIT storage/query layer. Create a DuckDB-backed store and
DataContext to automatically enable `ctx.pit`:

```python
from alphaforge.data.context import DataContext
from alphaforge.store.duckdb_parquet import DuckDBParquetStore

store = DuckDBParquetStore(root="./pit_store")
ctx = DataContext(sources={}, calendars={}, store=store)
```

### 2. Basic Usage

```python
from datetime import date
from pathlib import Path
from nowcast_data.pit.core.catalog import SeriesCatalog
from nowcast_data.pit.api import PITDataManager

# Load series catalog
catalog = SeriesCatalog(Path("series_catalog.yaml"))

# Initialize PIT data manager
manager = PITDataManager(catalog)

# Get US GDP as it was known on 2020-01-15
df = manager.get_series_asof(
    series_key="US_GDP_SAAR",
    asof_date=date(2020, 1, 15),
    start=date(2019, 1, 1),
    end=date(2019, 12, 31),
)

print(df)
```

### 3. Reference Period Queries

Reference period keys use `YYYY`, `YYYYQq`, `YYYY-MM`, or `YYYY/MM` formats.

```python
from datetime import date
from alphaforge.time.ref_period import RefFreq
from nowcast_data.pit.adapters.alphaforge import AlphaForgePITAdapter

adapter = AlphaForgePITAdapter(ctx=ctx)
adapter.fetch_asof_ref("GDP", date(2025, 5, 15), start_ref="2024Q4", end_ref="2025Q1", freq=RefFreq.Q)
adapter.fetch_revisions_ref("GDP", "2024Q4", freq=RefFreq.Q)
```

AlphaForge’s adapter can optionally ingest from `ctx.sources["fred"]` during
`fetch_asof` (default `True`) before querying the PIT table. Set
`ingest_from_ctx_source=False` to disable the source fetch side-effect.

### 4. Panel Data

```python
# Get multiple series at once
panel = manager.get_panel_asof(
    series_keys=["US_GDP_SAAR", "US_CPI", "US_UNRATE"],
    asof_date=date(2020, 6, 1),
    start=date(2019, 1, 1),
    end=date(2020, 5, 31),
    wide=True,  # Returns wide format (obs_date x series)
)
```

### 5. List Vintages

```python
# See all available vintages for a series
vintages = manager.get_series_vintages("US_GDP_SAAR")
print(f"Available vintages: {len(vintages)}")
```

### 6. Build PIT Cube

```python
# Build cube: multiple series x multiple asof dates
cube = manager.build_pit_cube(
    series_keys=["US_GDP_SAAR", "US_CPI"],
    asof_dates=[
        date(2020, 1, 15),
        date(2020, 3, 15),
        date(2020, 6, 15),
    ],
    start=date(2019, 1, 1),
    end=date(2019, 12, 31),
)
```

## Architecture

### Core Components

1. **PIT Adapters** (`nowcast_data.pit.adapters`): Data source connectors
   - `AlphaForgePITAdapter`: uses the canonical AlphaForge PIT table
   - `FREDALFREDAdapter`: US Federal Reserve data with real-time API
   - Other adapters (ECB, BoE, StatCan, Swiss): Stub implementations

2. **Core Logic** (`nowcast_data.pit.core`):
   - `vintage_logic.py`: Vintage selection algorithms
   - `models.py`: Data models and schemas
   - `catalog.py`: Series metadata management

3. **API** (`nowcast_data.pit.api`):
   - `PITDataManager`: High-level interface for data retrieval

### Data Model

The canonical PIT observation schema:

```
- series_key: Internal identifier
- source: Data source name
- source_series_id: Provider's series ID
- asof_date: Point-in-time evaluation date
- vintage_date: Provider's vintage identifier
- obs_date: Observation period date
- value: Numeric value
- frequency: D/W/M/Q/A
- units, seasonal_adjustment: Metadata
- realtime_start, realtime_end: ALFRED realtime window
- ingested_at: Timestamp
- provenance: Additional metadata (JSON)
```

## PIT Modes

The library supports three PIT modes:

1. **ALFRED_REALTIME**: Uses FRED's realtime API (realtime_start/realtime_end)
   - Used for US series with revision history
   - Continuous vintages available

2. **DISCRETE_VINTAGES_SNAP**: Uses discrete vintage snapshots
   - Selects latest vintage ≤ asof_date
   - Used for ECB, BoE, StatCan data sources

3. **NO_PIT**: Series does not support point-in-time retrieval
   - Will raise `PITNotSupportedError` if attempted

## Series Catalog

Series are defined in `series_catalog.yaml`:

```yaml
US_GDP_SAAR:
  country: US
  source: FRED_ALFRED
  source_series_id: GDP
  frequency: Q
  seasonal_adjustment: SAAR
  units: Billions of Dollars
  description: Gross Domestic Product
  pit_mode: ALFRED_REALTIME
```

## Testing

Run the test suite:

```bash
pytest tests/
```

Run specific test files:

```bash
pytest tests/test_vintage_logic.py
pytest tests/test_models.py
pytest tests/test_catalog.py
```

## Examples

See `examples/demo_pit.py` for comprehensive usage examples:

```bash
python examples/demo_pit.py
```

## Development Status

### Implemented ✅
- Core PIT vintage selection logic
- FRED/ALFRED adapter with full functionality
- Series catalog system
- Data models and schemas
- High-level API (get_series_asof, get_panel_asof, build_pit_cube)
- Unit tests for core functionality

### Stub/TODO 🚧
- ECB Real-Time Database adapter
- Bank of England RTDB adapter
- Statistics Canada adapter
- Switzerland adapter
- Storage layer (parquet/duckdb persistence)
- Caching layer
- Integration with alphaforge DataSource protocol

## Contributing

Contributions are welcome! Key areas for contribution:

1. Implementing stub adapters (ECB, BoE, StatCan, Swiss)
2. Adding more series to the catalog
3. Implementing storage/caching layer
4. Performance optimizations
5. Additional tests

## License

MIT License - see LICENSE file

## Acknowledgments

- Built on [alphaforge](https://github.com/steveya/alphaforge) by Steve Ya
- FRED/ALFRED API by Federal Reserve Bank of St. Louis

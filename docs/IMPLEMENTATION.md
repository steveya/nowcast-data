# How PIT Is Implemented

This document explains the implementation of the point-in-time (PIT) macro data retrieval system in the nowcast-data library.

## Architecture Overview

The PIT system consists of several layers:

```
User API (PITDataManager)
    ↓
Series Catalog (metadata)
    ↓
Adapters (data source connectors)
    ↓
Core Logic (vintage selection, models)
    ↓
External Data Sources (FRED, ECB, BoE, etc.)
```

## Core Concepts

### 1. Point-in-Time (PIT) Data

Point-in-time data represents economic time series as they were known on a specific date. This is critical for backtesting because macroeconomic data is often revised after initial publication.

**Example**: US GDP for Q4 2019 might have been:
- First reported on Jan 30, 2020: $21.5 trillion
- Revised on Feb 27, 2020: $21.7 trillion
- Final on Mar 26, 2020: $21.9 trillion

When backtesting a strategy that uses GDP data, you need to know what value was available on each date to avoid look-ahead bias.

### 2. Vintages

A **vintage** is a snapshot of a time series at a particular point in time. Each vintage contains:
- The vintage date (when this snapshot was created)
- All historical observations as they were known on that date
- Values may differ from other vintages due to revisions

### 3. As-Of Date

The **as-of date** is the evaluation date for which you want to retrieve data. The system selects the appropriate vintage (latest one not after the as-of date) to ensure no look-ahead bias.

## Key Components

### 1. Exceptions (`pit/exceptions.py`)

Custom exceptions for PIT operations:

- `PITNotSupportedError`: Raised when a series doesn't support PIT retrieval
- `VintageNotFoundError`: Raised when no vintage is available for an as-of date
- `SourceFetchError`: Raised when data fetching fails

### 2. Vintage Logic (`pit/core/vintage_logic.py`)

Core algorithm for selecting vintages:

```python
def select_vintage_for_asof(vintages, asof_date):
    """
    Returns the latest vintage not after asof_date.
    
    Rules:
    - If asof before first vintage: None
    - If asof equals vintage: that vintage
    - If asof between vintages: previous vintage
    - If asof after last vintage: last vintage
    """
```

This ensures strict no-lookahead: we only use data that was available on or before the as-of date.

### 3. Data Models (`pit/core/models.py`)

Defines the canonical schema for PIT observations:

```python
@dataclass
class PITObservation:
    series_key: str          # Internal identifier
    source: str              # Data source name
    source_series_id: str    # Provider's ID
    asof_date: date          # Evaluation date
    vintage_date: date       # Vintage identifier
    obs_date: date           # Observation date
    value: float             # Numeric value
    # ... metadata fields
```

Also provides utilities for converting observations to pandas DataFrames in both long and wide formats.

### 4. Series Catalog (`pit/core/catalog.py`)

Manages series metadata from YAML configuration.

### 5. Adapters (`pit/adapters/`)

Each adapter implements the `PITAdapter` protocol for fetching data from external sources.

#### FRED/ALFRED Adapter

The fully-implemented adapter for US Federal Reserve data uses FRED API with `realtime_start` and `realtime_end` parameters to retrieve historical vintages.

## PIT Modes

### ALFRED_REALTIME

Used for FRED/ALFRED data where vintages are continuous.

### DISCRETE_VINTAGES_SNAP

Used for sources with periodic vintage snapshots. Must select the latest vintage not after the as-of date.

### NO_PIT

Series does not support PIT retrieval.

## Summary

The PIT implementation provides:

1. **Correctness**: Strict no-lookahead guarantee
2. **Flexibility**: Supports multiple PIT modes
3. **Extensibility**: Easy to add new sources
4. **Testability**: Comprehensive test coverage
5. **Usability**: Simple, high-level API

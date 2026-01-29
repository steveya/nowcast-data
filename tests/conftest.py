from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

try:
    from alphaforge.data.context import DataContext
    from alphaforge.store.duckdb_parquet import DuckDBParquetStore

    HAS_ALPHAFORGE = True
except ImportError:
    HAS_ALPHAFORGE = False


@pytest.fixture
def pit_context(tmp_path: Path) -> DataContext:
    """Fixture providing an alphaforge DataContext for PIT tests.

    Skips the test if alphaforge is not installed.
    """
    if not HAS_ALPHAFORGE:
        pytest.skip("alphaforge is not installed")

    store = DuckDBParquetStore(root=str(tmp_path))
    ctx = DataContext(sources={}, calendars={}, store=store)
    base_rows = [
        {
            "obs_date": "2024-12-31",
            "asof_utc": "2025-01-10",
            "value": 1.0,
        },
        {
            "obs_date": "2024-12-31",
            "asof_utc": "2025-02-10",
            "value": 1.1,
        },
        {
            "obs_date": "2025-03-31",
            "asof_utc": "2025-04-10",
            "value": 2.0,
        },
        {
            "obs_date": "2025-03-31",
            "asof_utc": "2025-05-10",
            "value": 2.1,
        },
    ]
    data = pd.DataFrame(
        [{"series_key": "BASE_GDP", **row} for row in base_rows]
        + [{"series_key": "US_GDP_SAAR", **row} for row in base_rows]
    )
    gdp_rows = pd.DataFrame(
        [
            {
                "series_key": "GDP",
                "obs_date": "2024-12-31",
                "asof_utc": "2025-01-15",
                "value": 3.0,
            },
            {
                "series_key": "GDP",
                "obs_date": "2024-12-31",
                "asof_utc": "2025-03-01",
                "value": 3.5,
            },
        ]
    )
    daily_rows = pd.DataFrame(
        [
            {
                "series_key": "DAILY_FCI",
                "obs_date": "2025-01-05",
                "asof_utc": "2025-01-06",
                "value": 10.0,
            },
            {
                "series_key": "DAILY_FCI",
                "obs_date": "2025-01-10",
                "asof_utc": "2025-01-11",
                "value": 11.0,
            },
            {
                "series_key": "DAILY_FCI",
                "obs_date": "2025-01-20",
                "asof_utc": "2025-01-21",
                "value": 12.0,
            },
            {
                "series_key": "DAILY_FCI",
                "obs_date": "2025-02-05",
                "asof_utc": "2025-02-06",
                "value": 13.0,
            },
            {
                "series_key": "DAILY_FCI",
                "obs_date": "2025-02-09",
                "asof_utc": "2025-02-10",
                "value": 14.0,
            },
            {
                "series_key": "DAILY_FCI",
                "obs_date": "2025-02-15",  # After as-of date for cutoff tests
                "asof_utc": "2025-02-10",
                "value": 99.0,
            },
        ]
    )
    data = pd.concat([data, gdp_rows, daily_rows], ignore_index=True)
    ctx.pit.upsert_pit_observations(data)
    return ctx

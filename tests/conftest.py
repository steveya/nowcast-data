from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from alphaforge.data.context import DataContext
from alphaforge.store.duckdb_parquet import DuckDBParquetStore


@pytest.fixture
def pit_context(tmp_path: Path) -> DataContext:
    store = DuckDBParquetStore(root=str(tmp_path))
    ctx = DataContext(sources={}, calendars={}, store=store)
    data = pd.DataFrame(
        [
            {
                "series_key": "GDP",
                "obs_date": "2024-12-31",
                "asof_utc": "2025-01-10",
                "value": 1.0,
            },
            {
                "series_key": "GDP",
                "obs_date": "2024-12-31",
                "asof_utc": "2025-02-10",
                "value": 1.1,
            },
            {
                "series_key": "GDP",
                "obs_date": "2025-03-31",
                "asof_utc": "2025-04-10",
                "value": 2.0,
            },
            {
                "series_key": "GDP",
                "obs_date": "2025-03-31",
                "asof_utc": "2025-05-10",
                "value": 2.1,
            },
        ]
    )
    ctx.pit.upsert_pit_observations(data)
    return ctx

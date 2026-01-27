from __future__ import annotations

from datetime import date

import pytest

from alphaforge.data.context import DataContext
from alphaforge.data.query import Query
from alphaforge.store.duckdb_parquet import DuckDBParquetStore
from nowcast_data.pit.adapters.alphaforge import AlphaForgePITAdapter


class _FailingSource:
    name = "fred"

    def schemas(self) -> dict:
        return {}

    def fetch(self, q: Query):
        raise AssertionError("fetch should not be called when ingest is disabled")


def test_fetch_asof_skips_ingestion(tmp_path) -> None:
    store = DuckDBParquetStore(root=str(tmp_path))
    ctx = DataContext(sources={"fred": _FailingSource()}, calendars={}, store=store)
    adapter = AlphaForgePITAdapter(ctx=ctx)

    observations = adapter.fetch_asof(
        series_id="GDP",
        asof_date=date(2025, 1, 15),
        ingest_from_ctx_source=False,
    )

    assert observations == []

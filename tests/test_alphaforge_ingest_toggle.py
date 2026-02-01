from __future__ import annotations

from datetime import date

import pytest

pytest.importorskip("alphaforge")
from alphaforge.data.context import DataContext  # noqa: E402
from alphaforge.store.duckdb_parquet import DuckDBParquetStore  # noqa: E402
from nowcast_data.pit.adapters.alphaforge import AlphaForgePITAdapter  # noqa: E402


def test_fetch_asof_skips_ingestion(tmp_path, monkeypatch) -> None:
    store = DuckDBParquetStore(root=str(tmp_path))
    ctx = DataContext(sources={"fred": object()}, calendars={}, store=store)
    adapter = AlphaForgePITAdapter(ctx=ctx)

    def _fail_fetch_panel(*args, **kwargs):
        raise AssertionError("fetch_panel should not be called when ingest is disabled")

    monkeypatch.setattr(ctx, "fetch_panel", _fail_fetch_panel)

    observations = adapter.fetch_asof(
        series_id="GDP",
        asof_date=date(2025, 1, 15),
        ingest_from_ctx_source=False,
    )

    assert observations == []

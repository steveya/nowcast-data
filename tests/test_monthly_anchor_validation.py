from __future__ import annotations

from datetime import date

import pandas as pd
import pytest
from alphaforge.data.context import DataContext
from alphaforge.store.duckdb_parquet import DuckDBParquetStore

from nowcast_data.models.datasets import _build_predictor_frame
from nowcast_data.pit.adapters.alphaforge import AlphaForgePITAdapter
from nowcast_data.pit.core.catalog import SeriesCatalog
from nowcast_data.pit.core.models import SeriesMetadata


def _make_ctx(tmp_path) -> DataContext:
    store = DuckDBParquetStore(root=str(tmp_path))
    ctx = DataContext(sources={}, calendars={}, store=store)
    return ctx


class TestMonthlyObsDateAnchor:
    def test_monthly_anchor_start_accepts_month_start(self, tmp_path) -> None:
        ctx = _make_ctx(tmp_path)
        pit_rows = pd.DataFrame(
            [
                {
                    "series_key": "M_START",
                    "obs_date": "2025-01-01",
                    "asof_utc": "2025-02-01",
                    "value": 1.0,
                },
                {
                    "series_key": "M_START",
                    "obs_date": "2025-02-01",
                    "asof_utc": "2025-03-01",
                    "value": 2.0,
                },
            ]
        )
        ctx.pit.upsert_pit_observations(pit_rows)
        adapter = AlphaForgePITAdapter(ctx=ctx)

        catalog = SeriesCatalog()
        catalog.add(
            SeriesMetadata(
                series_key="M_START",
                country="US",
                source="TEST",
                source_series_id="M_START",
                frequency="M",
                pit_mode="NO_PIT",
                obs_date_anchor="start",
            )
        )

        _build_predictor_frame(
            adapter,
            catalog,
            predictor_series_keys=["M_START"],
            agg_spec={"M_START": "mean"},
            asof_date=date(2025, 3, 15),
            include_partial_quarters=True,
            ingest_from_ctx_source=False,
        )

    def test_monthly_anchor_end_accepts_month_end(self, tmp_path) -> None:
        ctx = _make_ctx(tmp_path)
        pit_rows = pd.DataFrame(
            [
                {
                    "series_key": "M_END",
                    "obs_date": "2025-01-31",
                    "asof_utc": "2025-02-10",
                    "value": 1.0,
                },
                {
                    "series_key": "M_END",
                    "obs_date": "2025-02-28",
                    "asof_utc": "2025-03-10",
                    "value": 2.0,
                },
            ]
        )
        ctx.pit.upsert_pit_observations(pit_rows)
        adapter = AlphaForgePITAdapter(ctx=ctx)

        catalog = SeriesCatalog()
        catalog.add(
            SeriesMetadata(
                series_key="M_END",
                country="US",
                source="TEST",
                source_series_id="M_END",
                frequency="M",
                pit_mode="NO_PIT",
                obs_date_anchor="end",
            )
        )

        _build_predictor_frame(
            adapter,
            catalog,
            predictor_series_keys=["M_END"],
            agg_spec={"M_END": "mean"},
            asof_date=date(2025, 3, 15),
            include_partial_quarters=True,
            ingest_from_ctx_source=False,
        )

    def test_monthly_anchor_none_rejects_mid_month(self, tmp_path) -> None:
        ctx = _make_ctx(tmp_path)
        pit_rows = pd.DataFrame(
            [
                {
                    "series_key": "M_NONE",
                    "obs_date": "2025-01-01",
                    "asof_utc": "2025-02-01",
                    "value": 1.0,
                },
                {
                    "series_key": "M_NONE",
                    "obs_date": "2025-01-15",
                    "asof_utc": "2025-02-01",
                    "value": 2.0,
                },
            ]
        )
        ctx.pit.upsert_pit_observations(pit_rows)
        adapter = AlphaForgePITAdapter(ctx=ctx)

        catalog = SeriesCatalog()
        catalog.add(
            SeriesMetadata(
                series_key="M_NONE",
                country="US",
                source="TEST",
                source_series_id="M_NONE",
                frequency="M",
                pit_mode="NO_PIT",
            )
        )

        with pytest.raises(ValueError, match="month-start or month-end"):
            _build_predictor_frame(
                adapter,
                catalog,
                predictor_series_keys=["M_NONE"],
                agg_spec={"M_NONE": "mean"},
                asof_date=date(2025, 3, 15),
                include_partial_quarters=True,
                ingest_from_ctx_source=False,
            )

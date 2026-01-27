from __future__ import annotations

from datetime import date

import pandas as pd

from alphaforge.data.context import DataContext
from alphaforge.store.duckdb_parquet import DuckDBParquetStore

from nowcast_data.benchmark.pit_builder import (
    SeriesSpec,
    apply_benchmark_transforms,
    build_monthly_panel_asof,
    compute_vintage_grid,
)


def test_benchmark_pit_builder_e2e(tmp_path) -> None:
    store = DuckDBParquetStore(root=str(tmp_path))
    ctx = DataContext(sources={}, calendars={}, store=store)

    specs = [
        SeriesSpec(series_id="M1", series_key="m1", frequency="m", transform="pctchange"),
        SeriesSpec(series_id="Q1", series_key="q1", frequency="q", transform="pctchange"),
    ]

    pit_rows = pd.DataFrame(
        [
            {
                "series_key": "m1",
                "obs_date": pd.Timestamp("2024-12-31", tz="UTC"),
                "asof_utc": pd.Timestamp("2025-02-01", tz="UTC"),
                "value": 100.0,
            },
            {
                "series_key": "m1",
                "obs_date": pd.Timestamp("2025-01-31", tz="UTC"),
                "asof_utc": pd.Timestamp("2025-02-01", tz="UTC"),
                "value": 110.0,
            },
            {
                "series_key": "m1",
                "obs_date": pd.Timestamp("2024-12-31", tz="UTC"),
                "asof_utc": pd.Timestamp("2025-02-10", tz="UTC"),
                "value": 105.0,
            },
            {
                "series_key": "m1",
                "obs_date": pd.Timestamp("2025-01-31", tz="UTC"),
                "asof_utc": pd.Timestamp("2025-02-10", tz="UTC"),
                "value": 115.0,
            },
            {
                "series_key": "q1",
                "obs_date": pd.Timestamp("2024-12-31", tz="UTC"),
                "asof_utc": pd.Timestamp("2025-02-05", tz="UTC"),
                "value": 200.0,
            },
            {
                "series_key": "q1",
                "obs_date": pd.Timestamp("2024-12-31", tz="UTC"),
                "asof_utc": pd.Timestamp("2025-03-15", tz="UTC"),
                "value": 205.0,
            },
            {
                "series_key": "q1",
                "obs_date": pd.Timestamp("2025-03-31", tz="UTC"),
                "asof_utc": pd.Timestamp("2025-03-15", tz="UTC"),
                "value": 210.0,
            },
        ]
    )

    ctx.pit.upsert_pit_observations(pit_rows)

    vintages = compute_vintage_grid(ctx, specs)
    expected = [
        pd.Timestamp("2025-02-01", tz="UTC"),
        pd.Timestamp("2025-02-05", tz="UTC"),
        pd.Timestamp("2025-02-10", tz="UTC"),
        pd.Timestamp("2025-03-15", tz="UTC"),
    ]
    assert vintages == expected

    panel = build_monthly_panel_asof(ctx, specs, pd.Timestamp("2025-03-15", tz="UTC"))
    assert pd.Timestamp("2024-12-31", tz="UTC") in panel.index
    assert pd.Timestamp("2025-03-31", tz="UTC") in panel.index

    tf_early = apply_benchmark_transforms(
        build_monthly_panel_asof(ctx, specs, pd.Timestamp("2025-02-01", tz="UTC")),
        specs,
    )
    tf_late = apply_benchmark_transforms(
        build_monthly_panel_asof(ctx, specs, pd.Timestamp("2025-02-10", tz="UTC")),
        specs,
    )

    obs_date = pd.Timestamp("2025-01-31", tz="UTC")
    assert tf_early.loc[obs_date, "m1"] != tf_late.loc[obs_date, "m1"]

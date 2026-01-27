from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from alphaforge.data.context import DataContext
from alphaforge.store.duckdb_parquet import DuckDBParquetStore

from nowcast_data.benchmark.manifest import MANIFEST_PATH, load_manifest


def _load_pit_frame(manifest: dict, pit_context: DataContext) -> pd.DataFrame:
    data_locations = manifest.get("data_locations", {})
    pit_store_root = Path(data_locations.get("pit_store_root", "pit_store"))
    if pit_store_root.exists():
        store = DuckDBParquetStore(root=str(pit_store_root))
        ctx = DataContext(sources={}, calendars={}, store=store)
        return ctx.pit.conn.execute(
            "SELECT series_key, obs_date, asof_utc FROM pit_observations"
        ).df()

    benchmark_raw = Path(data_locations.get("benchmark_data_raw", "benchmark_pit/data_raw.parquet"))
    if benchmark_raw.exists():
        conn = duckdb.connect()
        df = conn.execute("SELECT * FROM parquet_scan(?)", [str(benchmark_raw)]).df()
        conn.close()

        if "asof_date" in df.columns and "asof_utc" not in df.columns:
            df = df.rename(columns={"asof_date": "asof_utc"})
        id_vars = [col for col in ["asof_utc", "obs_date"] if col in df.columns]
        series_cols = [col for col in df.columns if col not in id_vars]
        if id_vars and series_cols:
            long_df = df.melt(id_vars=id_vars, var_name="series_key", value_name="value")
            return long_df[["series_key", "obs_date", "asof_utc"]]

    return pit_context.pit.conn.execute(
        "SELECT series_key, obs_date, asof_utc FROM pit_observations"
    ).df()


def test_manifest_exists_and_parses() -> None:
    assert MANIFEST_PATH.exists()
    manifest = load_manifest()
    assert isinstance(manifest, dict)


def test_manifest_contract_keys() -> None:
    manifest = load_manifest()
    for key in ["dataset_name", "created_at", "timezone_conventions", "series", "target", "vintages"]:
        assert key in manifest
    for key in ["series_key", "ref_period_convention", "frequency"]:
        assert key in manifest["target"]


def test_manifest_series_unique() -> None:
    manifest = load_manifest()
    series = manifest["series"]
    series_keys = [entry["series_key"] for entry in series]
    assert len(series_keys) == len(set(series_keys))


def test_pit_data_contract(pit_context: DataContext) -> None:
    manifest = load_manifest()
    series_keys = {entry["series_key"] for entry in manifest["series"]}
    pit_df = _load_pit_frame(manifest, pit_context)

    assert "series_key" in pit_df.columns
    assert pit_df["series_key"].notna().all()
    assert set(pit_df["series_key"]).issubset(series_keys)

    obs_dates = pd.to_datetime(pit_df["obs_date"], utc=True, errors="coerce")
    assert obs_dates.notna().all()

    asof_dates = pd.to_datetime(pit_df["asof_utc"], utc=True, errors="coerce")
    assert asof_dates.notna().all()
    assert asof_dates.dt.tz is not None
    assert str(asof_dates.dt.tz) == "UTC"

    vintages = pd.DatetimeIndex(asof_dates.unique()).sort_values()
    assert vintages.notna().all()
    assert vintages.is_unique
    assert vintages.is_monotonic_increasing

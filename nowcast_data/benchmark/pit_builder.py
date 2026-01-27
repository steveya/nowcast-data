from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import logging
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from alphaforge.data.context import DataContext
from alphaforge.data.query import Query

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SeriesSpec:
    series_id: str
    series_key: str
    frequency: str
    transform: str
    name: Optional[str] = None
    meta: Optional[dict] = None


def _coerce_utc_timestamp(value: pd.Timestamp | str | date) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


def _to_month_end(ts: pd.Timestamp) -> pd.Timestamp:
    ts = _coerce_utc_timestamp(ts)
    naive = ts.tz_localize(None)
    month_end = naive.to_period("M").end_time.normalize()
    return month_end.tz_localize("UTC")


def _to_quarter_end(ts: pd.Timestamp) -> pd.Timestamp:
    ts = _coerce_utc_timestamp(ts)
    naive = ts.tz_localize(None)
    quarter_end = naive.to_period("Q").end_time.normalize()
    return quarter_end.tz_localize("UTC")


def load_benchmark_metadata(
    path: str | Path = "data/meta_data.csv",
) -> list[SeriesSpec]:
    meta_path = Path(path)
    df = pd.read_csv(meta_path)
    df.columns = [c.strip().lower() for c in df.columns]

    used_columns = {"series", "name", "freq", "transform"}
    logger.info("Loading benchmark metadata from %s", meta_path)
    logger.info("Using columns: %s", sorted(used_columns & set(df.columns)))

    specs: list[SeriesSpec] = []
    for _, row in df.iterrows():
        series = str(row.get("series", "")).strip()
        if not series:
            continue
        series_id = series.upper()
        series_key = series.lower()
        freq = str(row.get("freq", "m")).strip().lower() or "m"
        transform = str(row.get("transform", "pctchange")).strip().lower() or "pctchange"
        name = str(row.get("name", "")).strip() or None

        meta = {k: row[k] for k in df.columns if k not in {"series", "name", "freq", "transform"}}
        specs.append(
            SeriesSpec(
                series_id=series_id,
                series_key=series_key,
                frequency=freq,
                transform=transform,
                name=name,
                meta=meta,
            )
        )
    return specs


def ingest_alfred_full_history(
    ctx: DataContext,
    spec: SeriesSpec,
    *,
    fred_source_name: str = "fred",
) -> None:
    if ctx.pit is None:
        raise ValueError("DataContext does not have PIT enabled")
    source = ctx.sources.get(fred_source_name)
    if source is None:
        raise ValueError(f"Missing data source '{fred_source_name}' in DataContext")

    fred_client = getattr(source, "_fred", None)
    if fred_client is None or not hasattr(fred_client, "get_series_vintage_dates"):
        raise ValueError(
            "FRED source does not expose get_series_vintage_dates; cannot ingest full history"
        )

    vintage_dates = fred_client.get_series_vintage_dates(spec.series_id)
    if vintage_dates is None:
        return

    vintages = pd.to_datetime(vintage_dates, utc=True)
    vintages = pd.DatetimeIndex(vintages).sort_values().unique()

    for vintage in vintages:
        q = Query(
            table="fred_series",
            columns=["value"],
            entities=[spec.series_id],
            asof=pd.Timestamp(vintage, tz="UTC"),
        )
        panel = ctx.fetch_panel(fred_source_name, q)
        panel_df = panel.df.reset_index()
        required = {"entity_id", "ts_utc", "asof_utc", "value"}
        missing = required - set(panel_df.columns)
        if missing:
            raise ValueError("Unexpected alphaforge panel schema; missing " f"{sorted(missing)}")

        pit_df = pd.DataFrame(
            {
                "series_key": [spec.series_key] * len(panel_df),
                "obs_date": pd.to_datetime(panel_df["ts_utc"], utc=True).dt.floor("D"),
                "asof_utc": pd.to_datetime(panel_df["asof_utc"], utc=True),
                "value": panel_df["value"],
                "source": "alfred",
                "revision_id": pd.NA,
                "meta_json": pd.NA,
                "release_time_utc": pd.NaT,
            }
        )
        ctx.pit.upsert_pit_observations(pit_df)


def compute_vintage_grid(ctx: DataContext, specs: list[SeriesSpec]) -> list[pd.Timestamp]:
    if ctx.pit is None:
        raise ValueError("DataContext does not have PIT enabled")
    conn = ctx.pit.conn
    vintages: set[pd.Timestamp] = set()
    for spec in specs:
        rows = conn.execute(
            "SELECT DISTINCT asof_utc FROM pit_observations WHERE series_key = ?",
            [spec.series_key],
        ).fetchall()
        for (asof,) in rows:
            if asof is None:
                continue
            vintages.add(_coerce_utc_timestamp(asof))
    return sorted(vintages)


def build_monthly_panel_asof(
    ctx: DataContext,
    specs: list[SeriesSpec],
    asof: pd.Timestamp,
    *,
    start: date | None = None,
    end: date | None = None,
) -> pd.DataFrame:
    if ctx.pit is None:
        raise ValueError("DataContext does not have PIT enabled")

    asof_ts = _coerce_utc_timestamp(asof)
    start_ts = _coerce_utc_timestamp(start) if start is not None else None
    end_ts = _coerce_utc_timestamp(end) if end is not None else None

    series_by_key: dict[str, pd.Series] = {}
    for spec in specs:
        snap = ctx.pit.get_snapshot(
            spec.series_key,
            asof_ts,
            start=start_ts,
            end=end_ts,
        )
        if snap.empty:
            series_by_key[spec.series_key] = pd.Series(dtype="float64")
            continue

        if spec.frequency == "q":
            aligned_index = snap.index.map(_to_quarter_end)
        else:
            aligned_index = snap.index.map(_to_month_end)

        aligned = pd.Series(snap.values, index=pd.DatetimeIndex(aligned_index))
        aligned = aligned.groupby(level=0).last().sort_index()
        aligned.index.name = "obs_date"
        series_by_key[spec.series_key] = aligned

    panel = pd.DataFrame(series_by_key)
    panel = panel.reindex(columns=[spec.series_key for spec in specs])
    panel.index = pd.DatetimeIndex(panel.index).tz_convert("UTC")
    panel.index.name = "obs_date"
    return panel.sort_index()


def apply_benchmark_transforms(
    panel_raw: pd.DataFrame,
    specs: list[SeriesSpec],
) -> pd.DataFrame:
    spec_map = {spec.series_key: spec for spec in specs}
    transformed = pd.DataFrame(index=panel_raw.index)

    for col in panel_raw.columns:
        series = panel_raw[col]
        spec = spec_map.get(col)
        transform = (spec.transform if spec else "pctchange").lower()

        if transform == "level":
            out = series.copy()
        elif transform == "diff":
            out = series.diff()
        elif transform == "logdiff":
            out = np.log(series).diff() * 100.0
        elif transform == "pctchange":
            out = series.pct_change(fill_method=None) * 100.0
        else:
            logger.warning("Unknown transform '%s' for %s; defaulting to pctchange", transform, col)
            out = series.pct_change(fill_method=None) * 100.0

        transformed[col] = out

    transformed.index = panel_raw.index
    transformed.index.name = panel_raw.index.name
    return transformed


def build_and_export_benchmark_pit(
    ctx: DataContext,
    specs: list[SeriesSpec],
    out_dir: str | Path = "benchmark_pit",
    *,
    start: date | None = None,
    end: date | None = None,
) -> None:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    vintage_grid = compute_vintage_grid(ctx, specs)
    vintages_df = pd.DataFrame({"asof_date": vintage_grid})
    vintages_df.to_parquet(out_path / "vintages.parquet", index=False)

    raw_frames: list[pd.DataFrame] = []
    tf_frames: list[pd.DataFrame] = []

    for asof in vintage_grid:
        raw = build_monthly_panel_asof(ctx, specs, asof, start=start, end=end)
        tf = apply_benchmark_transforms(raw, specs)

        raw_with_asof = raw.copy()
        raw_with_asof["asof_date"] = _coerce_utc_timestamp(asof)
        raw_with_asof = raw_with_asof.set_index("asof_date", append=True)
        raw_with_asof = raw_with_asof.reorder_levels(["asof_date", "obs_date"]).sort_index()

        tf_with_asof = tf.copy()
        tf_with_asof["asof_date"] = _coerce_utc_timestamp(asof)
        tf_with_asof = tf_with_asof.set_index("asof_date", append=True)
        tf_with_asof = tf_with_asof.reorder_levels(["asof_date", "obs_date"]).sort_index()

        raw_frames.append(raw_with_asof)
        tf_frames.append(tf_with_asof)

    if raw_frames:
        pd.concat(raw_frames).to_parquet(out_path / "data_raw.parquet")
    else:
        pd.DataFrame().to_parquet(out_path / "data_raw.parquet")

    if tf_frames:
        pd.concat(tf_frames).to_parquet(out_path / "data_tf.parquet")
    else:
        pd.DataFrame().to_parquet(out_path / "data_tf.parquet")

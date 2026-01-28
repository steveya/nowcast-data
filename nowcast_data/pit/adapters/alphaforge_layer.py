"""Thin adapter wrapper around AlphaForge PIT APIs."""

from datetime import date
from typing import Optional

import pandas as pd
from alphaforge.data.context import DataContext
from alphaforge.time.ref_period import RefPeriod, RefFreq


def _coerce_utc_timestamp(value: date | pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


def _normalize_utc_day(value: date | pd.Timestamp) -> date:
    return _coerce_utc_timestamp(value).normalize().date()


class AlphaForgePITLayer:
    """Wrapper for AlphaForge PIT accessors."""

    def __init__(self, ctx: DataContext) -> None:
        if ctx.pit is None:
            raise ValueError("PIT requires DuckDBParquetStore-backed DataContext")
        self._ctx = ctx

    def snapshot(
        self,
        series_key: str,
        asof: pd.Timestamp,
        start: Optional[pd.Timestamp] = None,
        end: Optional[pd.Timestamp] = None,
    ) -> pd.Series:
        return self._ctx.pit.get_snapshot(series_key, asof=asof, start=start, end=end)

    def snapshot_ref(
        self,
        series_key: str,
        asof: pd.Timestamp,
        start_ref: str | RefPeriod | None = None,
        end_ref: str | RefPeriod | None = None,
        *,
        freq: Optional[RefFreq] = None,
    ) -> pd.Series:
        if isinstance(start_ref, str):
            start_ref = RefPeriod.parse(start_ref)
        if isinstance(end_ref, str):
            end_ref = RefPeriod.parse(end_ref)
        return self._ctx.pit.get_snapshot_ref(
            series_key,
            asof=asof,
            start_ref=start_ref,
            end_ref=end_ref,
            freq=freq,
        )

    def revisions(
        self,
        series_key: str,
        obs_date: pd.Timestamp,
        start_asof: Optional[pd.Timestamp] = None,
        end_asof: Optional[pd.Timestamp] = None,
    ) -> pd.Series:
        return self._ctx.pit.get_revision_timeline(
            series_key, obs_date=obs_date, start_asof=start_asof, end_asof=end_asof
        )

    def revisions_ref(
        self,
        series_key: str,
        ref: str | RefPeriod,
        start_asof: Optional[pd.Timestamp] = None,
        end_asof: Optional[pd.Timestamp] = None,
        *,
        freq: Optional[RefFreq] = None,
    ) -> pd.Series:
        if isinstance(ref, str):
            ref = RefPeriod.parse(ref)
        return self._ctx.pit.get_revision_timeline_ref(
            series_key, ref=ref, start_asof=start_asof, end_asof=end_asof, freq=freq
        )

    def upsert(self, df: pd.DataFrame) -> None:
        self._ctx.pit.upsert_pit_observations(df)

    def list_pit_observations_asof(
        self,
        *,
        series_key: str,
        obs_date: date,
        asof_date: date,
    ) -> pd.DataFrame:
        conn = self._ctx.pit.conn
        obs_day = _normalize_utc_day(obs_date)
        asof_cutoff = _coerce_utc_timestamp(asof_date).normalize() + pd.Timedelta(days=1)
        df = conn.execute(
            """
            SELECT series_key, obs_date, asof_utc, value
            FROM pit_observations
            WHERE series_key = ?
              AND DATE(obs_date) = ?
              AND asof_utc < ?
            ORDER BY asof_utc ASC
            """,
            [series_key, obs_day, asof_cutoff],
        ).fetchdf()
        if df.empty:
            return pd.DataFrame(
                {
                    "series_key": pd.Series(dtype="object"),
                    "obs_date": pd.Series(dtype="datetime64[ns, UTC]"),
                    "asof_utc": pd.Series(dtype="datetime64[ns, UTC]"),
                    "value": pd.Series(dtype="float64"),
                }
            )

        df["obs_date"] = pd.to_datetime(df["obs_date"], utc=True).dt.floor("D")
        df["asof_utc"] = pd.to_datetime(df["asof_utc"], utc=True)
        return df

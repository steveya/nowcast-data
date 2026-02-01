"""Adapter for fetching data from AlphaForge."""

from __future__ import annotations

from datetime import date, datetime
from typing import List, Optional

import pandas as pd
try:  # pragma: no cover - optional dependency
    from alphaforge.data.context import DataContext
    from alphaforge.data.query import Query
    from alphaforge.time.ref_period import RefPeriod, RefFreq
except ImportError:  # pragma: no cover - optional dependency
    DataContext = None  # type: ignore[assignment]
    Query = None  # type: ignore[assignment]
    RefPeriod = None  # type: ignore[assignment]
    RefFreq = None  # type: ignore[assignment]

from nowcast_data.pit.adapters.base import PITAdapter
from nowcast_data.pit.adapters.alphaforge_layer import AlphaForgePITLayer
from nowcast_data.pit.core.models import PITObservation, SeriesMetadata


def _normalize_obs_date_key(value) -> date:
    """Normalize AlphaForge obs_date timestamps to canonical date keys.

    AlphaForge may return obs_date timestamps with non-midnight UTC times.
    We round by adding 12 hours before taking the date to recover period-end keys.
    """
    if isinstance(value, date) and not isinstance(value, datetime):
        return value

    ts = pd.Timestamp(value)
    if pd.isna(ts):
        raise ValueError("obs_date is missing or NaT")
    if ts.tz is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    if ts != ts.normalize():
        ts = ts + pd.Timedelta(hours=12)
    return ts.normalize().date()


class AlphaForgePITAdapter(PITAdapter):
    """Point-in-time data adapter for AlphaForge."""

    def __init__(self, ctx: DataContext):
        if DataContext is None or RefPeriod is None or Query is None:
            raise ImportError("alphaforge must be installed to use AlphaForgePITAdapter")
        self._ctx = ctx
        self._layer = AlphaForgePITLayer(ctx)

    @property
    def name(self) -> str:
        """Adapter name/identifier."""
        return "alphaforge"

    def supports_pit(self, series_id: str) -> bool:
        """
        Check if a series supports point-in-time retrieval.

        Args:
            series_id: Source-specific series identifier

        Returns:
            True if PIT is supported, False otherwise
        """
        # For now, assume all series from AlphaForge support PIT
        return True

    def list_vintages(self, query_series_key: str) -> List[date]:
        """
        List available vintage dates for a series.

        Args:
            query_series_key: PIT series key stored in pit_observations

        Returns:
            List of vintage dates (sorted)

        Raises:
            PITNotSupportedError: If series doesn't support vintages
            SourceFetchError: If fetching fails
        """
        conn = self._ctx.pit.conn
        rows = conn.execute(
            "SELECT DISTINCT asof_utc FROM pit_observations WHERE series_key = ?",
            [query_series_key],
        ).fetchall()
        if not rows:
            return []
        vintages = sorted(
            {pd.Timestamp(row[0], tz="UTC").date() for row in rows if row[0] is not None}
        )
        return vintages

    def list_pit_observations_asof(
        self,
        *,
        series_key: str,
        obs_date: date,
        asof_date: date,
    ) -> pd.DataFrame:
        """
        List all PIT observations for a series/obs_date up to an as-of date.

        Args:
            series_key: PIT series key stored in pit_observations.
            obs_date: Observation date (day-granularity).
            asof_date: As-of date treated as end-of-day UTC.

        Returns:
            DataFrame with columns: series_key, obs_date, asof_utc, value
        """
        return self._layer.list_pit_observations_asof(
            series_key=series_key,
            obs_date=obs_date,
            asof_date=asof_date,
        )

    def fetch_asof(
        self,
        series_id: str,
        asof_date: date,
        start: Optional[date] = None,
        end: Optional[date] = None,
        *,
        metadata: Optional[SeriesMetadata] = None,
        ingest_from_ctx_source: bool = True,
    ) -> List[PITObservation]:
        """
        Fetch observations as they were known on asof_date.

        Args:
            series_id: Source-specific series identifier
            asof_date: Point-in-time evaluation date
            start: Optional start date for observation period
            end: Optional end date for observation period

        Returns:
            List of PIT observations

        Raises:
            PITNotSupportedError: If series doesn't support PIT
            VintageNotFoundError: If no vintage available at asof_date
            SourceFetchError: If fetching fails
        """
        source_series_id = metadata.source_series_id if metadata else series_id
        query_series_key = metadata.series_key if metadata else series_id

        asof_ts = pd.Timestamp(asof_date, tz="UTC")
        start_ts = pd.Timestamp(start, tz="UTC") if start else None
        end_ts = pd.Timestamp(end, tz="UTC") if end else None

        if ingest_from_ctx_source and "fred" in self._ctx.sources:
            query = Query(
                table="fred_series",
                columns=["value"],
                entities=[source_series_id],
                start=start_ts,
                end=end_ts,
                asof=asof_ts,
            )
            panel = self._ctx.fetch_panel("fred", query)
            panel_df = panel.df.reset_index()
            required = {"entity_id", "ts_utc", "asof_utc", "value"}
            missing = required - set(panel_df.columns)
            if missing:
                raise ValueError(
                    "Unexpected alphaforge panel schema; missing " f"{sorted(missing)}"
                )

            if metadata is None:
                series_keys_from_df = panel_df["entity_id"]
                series_key_values = series_keys_from_df
            else:
                series_keys_repeated = [query_series_key] * len(panel_df)
                series_key_values = series_keys_repeated
            pit_df = pd.DataFrame(
                {
                    "series_key": series_key_values,
                    "obs_date": pd.to_datetime(panel_df["ts_utc"], utc=True).dt.floor("D"),
                    "asof_utc": pd.to_datetime(panel_df["asof_utc"], utc=True),
                    "value": panel_df["value"],
                    "source": pd.NA,
                    "revision_id": pd.NA,
                    "meta_json": pd.NA,
                    "release_time_utc": pd.NaT,
                }
            )
            self._ctx.pit.upsert_pit_observations(pit_df)

        snap = self._layer.snapshot(query_series_key, asof=asof_ts, start=start_ts, end=end_ts)

        observations = []
        series_key = metadata.series_key if metadata else series_id
        source_series_id = metadata.source_series_id if metadata else series_id
        frequency = metadata.frequency if metadata else ""
        source = metadata.source if metadata else "alphaforge"
        for obs_date, value in snap.items():
            obs = PITObservation(
                series_key=series_key,
                source=source,
                source_series_id=source_series_id,
                asof_date=asof_date,
                vintage_date=asof_date,
                obs_date=_normalize_obs_date_key(obs_date),
                value=float(value),
                frequency=frequency,
            )
            observations.append(obs)
        return observations

    def fetch_asof_ref(
        self,
        series_id: str,
        asof_date: date,
        start_ref: str | RefPeriod | None = None,
        end_ref: str | RefPeriod | None = None,
        *,
        freq: Optional[RefFreq] = None,
        metadata: Optional[SeriesMetadata] = None,
    ) -> List[PITObservation]:
        query_series_key = metadata.series_key if metadata else series_id
        asof_ts = pd.Timestamp(asof_date, tz="UTC")
        snap = self._layer.snapshot_ref(
            query_series_key, asof=asof_ts, start_ref=start_ref, end_ref=end_ref, freq=freq
        )
        observations = []
        series_key = metadata.series_key if metadata else series_id
        source_series_id = metadata.source_series_id if metadata else series_id
        source = metadata.source if metadata else "alphaforge"
        frequency = metadata.frequency if metadata else (freq.value if freq else "")
        for obs_date, value in snap.items():
            obs = PITObservation(
                series_key=series_key,
                source=source,
                source_series_id=source_series_id,
                asof_date=asof_date,
                vintage_date=asof_date,
                obs_date=_normalize_obs_date_key(obs_date),
                value=float(value),
                frequency=frequency,
            )
            observations.append(obs)
        return observations

    def fetch_revisions_ref(
        self,
        series_id: str,
        ref: str | RefPeriod,
        start_asof: Optional[date] = None,
        end_asof: Optional[date] = None,
        *,
        freq: Optional[RefFreq] = None,
        metadata: Optional[SeriesMetadata] = None,
    ) -> pd.Series:
        start_ts = pd.Timestamp(start_asof, tz="UTC") if start_asof else None
        end_ts = pd.Timestamp(end_asof, tz="UTC") if end_asof else None
        query_series_key = metadata.series_key if metadata else series_id
        series = self._layer.revisions_ref(
            query_series_key, ref, start_asof=start_ts, end_asof=end_ts, freq=freq
        )
        if metadata is not None:
            series = series.rename(metadata.series_key)
        return series

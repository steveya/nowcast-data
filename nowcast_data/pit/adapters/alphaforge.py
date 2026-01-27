"""Adapter for fetching data from AlphaForge."""

from __future__ import annotations

from datetime import date
from typing import List, Optional

import pandas as pd
from alphaforge.data.context import DataContext
from alphaforge.data.query import Query
from alphaforge.time.ref_period import RefPeriod, RefFreq

from nowcast_data.pit.adapters.base import PITAdapter
from nowcast_data.pit.adapters.alphaforge_layer import AlphaForgePITLayer
from nowcast_data.pit.core.models import PITObservation, SeriesMetadata


class AlphaForgePITAdapter(PITAdapter):
    """Point-in-time data adapter for AlphaForge."""

    def __init__(self, ctx: DataContext):
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

    def list_vintages(self, series_id: str) -> List[date]:
        """
        List available vintage dates for a series.
        
        Args:
            series_id: Source-specific series identifier
            
        Returns:
            List of vintage dates (sorted)
            
        Raises:
            PITNotSupportedError: If series doesn't support vintages
            SourceFetchError: If fetching fails
        """
        conn = self._ctx.pit.conn
        rows = conn.execute(
            "SELECT DISTINCT asof_utc FROM pit_observations WHERE series_key = ?",
            [series_id],
        ).fetchall()
        if not rows:
            return []
        vintages = sorted(
            {pd.Timestamp(row[0], tz="UTC").date() for row in rows if row[0] is not None}
        )
        return vintages

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
        asof_ts = pd.Timestamp(asof_date, tz="UTC")
        start_ts = pd.Timestamp(start, tz="UTC") if start else None
        end_ts = pd.Timestamp(end, tz="UTC") if end else None

        if ingest_from_ctx_source and "fred" in self._ctx.sources:
            query = Query(
                table="fred_series",
                columns=["value"],
                entities=[series_id],
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
                    "Unexpected alphaforge panel schema; missing "
                    f"{sorted(missing)}"
                )

            pit_df = pd.DataFrame(
                {
                    "series_key": panel_df["entity_id"],
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

        query_series_key = series_id
        snap = self._layer.snapshot(
            query_series_key, asof=asof_ts, start=start_ts, end=end_ts
        )

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
                obs_date=pd.Timestamp(obs_date).date(),
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
        asof_ts = pd.Timestamp(asof_date, tz="UTC")
        snap = self._layer.snapshot_ref(
            series_id, asof=asof_ts, start_ref=start_ref, end_ref=end_ref, freq=freq
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
                obs_date=pd.Timestamp(obs_date).date(),
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
        query_series_key = series_id
        series = self._layer.revisions_ref(
            query_series_key, ref, start_asof=start_ts, end_asof=end_ts, freq=freq
        )
        if metadata is not None:
            series = series.rename(metadata.series_key)
        return series

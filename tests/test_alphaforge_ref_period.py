from __future__ import annotations

from datetime import date

import pandas as pd

from alphaforge.time.ref_period import RefFreq
from nowcast_data.pit.adapters.alphaforge import AlphaForgePITAdapter


def test_snapshot_ref_ranges(pit_context) -> None:
    adapter = AlphaForgePITAdapter(ctx=pit_context)
    snap = adapter.fetch_asof_ref(
        series_id="BASE_GDP",
        asof_date=date(2025, 5, 15),
        start_ref="2024Q4",
        end_ref="2025Q1",
        freq=RefFreq.Q,
    )

    obs_dates = [obs.obs_date for obs in snap]
    assert obs_dates == [date(2024, 12, 31), date(2025, 3, 31)]


def test_snapshot_ref_normalizes_non_midnight_timestamps(pit_context, monkeypatch) -> None:
    adapter = AlphaForgePITAdapter(ctx=pit_context)
    index = pd.to_datetime(["2024-12-30 19:00:00+00:00", "2025-03-30 20:00:00+00:00"], utc=True)
    fake_snap = pd.Series([1.0, 2.0], index=index)

    monkeypatch.setattr(adapter._layer, "snapshot_ref", lambda *args, **kwargs: fake_snap)

    snap = adapter.fetch_asof_ref(
        series_id="BASE_GDP",
        asof_date=date(2025, 5, 15),
        start_ref="2024Q4",
        end_ref="2025Q1",
        freq=RefFreq.Q,
    )

    obs_dates = [obs.obs_date for obs in snap]
    assert obs_dates == [date(2024, 12, 31), date(2025, 3, 31)]


def test_revisions_ref_matches_obs_date(pit_context) -> None:
    adapter = AlphaForgePITAdapter(ctx=pit_context)
    revs = adapter.fetch_revisions_ref("BASE_GDP", "2024Q4", freq=RefFreq.Q)
    assert isinstance(revs, pd.Series)
    assert list(revs.values) == [1.0, 1.1]

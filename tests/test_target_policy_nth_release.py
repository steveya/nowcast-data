from datetime import date

import pandas as pd
import pytest

from nowcast_data.models.target_policy import (
    TargetPolicy,
    list_quarterly_target_releases_asof,
    resolve_target_from_releases,
)
from nowcast_data.pit.adapters.alphaforge import AlphaForgePITAdapter


def test_nth_release_selects_expected(pit_context) -> None:
    pit_context.pit.upsert_pit_observations(
        pd.DataFrame(
            [
                {
                    "series_key": "GDPC1",
                    "obs_date": "2025-03-31",
                    "asof_utc": "2025-04-25",
                    "value": 1.0,
                },
                {
                    "series_key": "GDPC1",
                    "obs_date": "2025-03-31",
                    "asof_utc": "2025-05-28",
                    "value": 1.2,
                },
                {
                    "series_key": "GDPC1",
                    "obs_date": "2025-03-31",
                    "asof_utc": "2025-06-27",
                    "value": 1.3,
                },
            ]
        )
    )

    adapter = AlphaForgePITAdapter(ctx=pit_context)
    releases = list_quarterly_target_releases_asof(
        adapter,
        series_key="GDPC1",
        ref_quarter="2025Q1",
        asof_date=date(2025, 7, 1),
    )
    value, meta = resolve_target_from_releases(releases, TargetPolicy(mode="nth_release", nth=2))
    assert value == 1.2
    assert meta["selected_release_rank"] == 2


def test_nth_release_missing_returns_none(pit_context) -> None:
    pit_context.pit.upsert_pit_observations(
        pd.DataFrame(
            [
                {
                    "series_key": "GDPC1",
                    "obs_date": "2025-03-31",
                    "asof_utc": "2025-04-25",
                    "value": 1.0,
                },
            ]
        )
    )

    adapter = AlphaForgePITAdapter(ctx=pit_context)
    releases = list_quarterly_target_releases_asof(
        adapter,
        series_key="GDPC1",
        ref_quarter="2025Q1",
        asof_date=date(2025, 6, 1),
    )
    value, meta = resolve_target_from_releases(releases, TargetPolicy(mode="nth_release", nth=2))
    assert value is None
    assert meta["selected_release_rank"] == 2


@pytest.mark.parametrize("nth", [0, -1])
def test_nth_release_invalid_nth_raises(nth: int) -> None:
    with pytest.raises(ValueError, match="TargetPolicy.nth must be >= 1"):
        resolve_target_from_releases(
            pd.DataFrame(columns=["obs_date", "asof_utc", "value"]),
            TargetPolicy(mode="nth_release", nth=nth),
        )

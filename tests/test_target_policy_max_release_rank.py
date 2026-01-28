from datetime import date

import pandas as pd

from nowcast_data.models.target_policy import (
    TargetPolicy,
    list_quarterly_target_releases_asof,
    resolve_target_from_releases,
)
from nowcast_data.pit.adapters.alphaforge import AlphaForgePITAdapter


def test_max_release_rank_caps_latest_available(pit_context) -> None:
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
    value, meta = resolve_target_from_releases(
        releases, TargetPolicy(mode="latest_available", max_release_rank=2)
    )
    assert value == 1.2
    assert meta["selected_release_rank"] == 2
    assert meta["n_releases_available"] == 2
    assert meta["available_release_ranks"] == [1, 2]

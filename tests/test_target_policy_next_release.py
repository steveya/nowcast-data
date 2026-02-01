from datetime import date

import pandas as pd
import pytest

pytest.importorskip("alphaforge")
from nowcast_data.models.target_policy import (  # noqa: E402
    TargetPolicy,
    list_quarterly_target_releases_asof,
    resolve_target_from_releases,
)
from nowcast_data.pit.adapters.alphaforge import AlphaForgePITAdapter  # noqa: E402


def test_next_release_returns_none(pit_context) -> None:
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
    value, meta = resolve_target_from_releases(releases, TargetPolicy(mode="next_release"))
    assert value is None
    assert meta["selected_release_rank"] is None
    assert meta["target_release_rank"] == 3

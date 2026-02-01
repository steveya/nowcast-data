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


def test_latest_available_changes_with_asof(pit_context) -> None:
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

    early_releases = list_quarterly_target_releases_asof(
        adapter,
        series_key="GDPC1",
        ref_quarter="2025Q1",
        asof_date=date(2025, 5, 1),
    )
    early_value, early_meta = resolve_target_from_releases(
        early_releases, TargetPolicy(mode="latest_available")
    )
    assert early_value == 1.0
    assert early_meta["selected_release_rank"] == 1

    late_releases = list_quarterly_target_releases_asof(
        adapter,
        series_key="GDPC1",
        ref_quarter="2025Q1",
        asof_date=date(2025, 6, 1),
    )
    late_value, late_meta = resolve_target_from_releases(
        late_releases, TargetPolicy(mode="latest_available")
    )
    assert late_value == 1.2
    assert late_meta["selected_release_rank"] == 2

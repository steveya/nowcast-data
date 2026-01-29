from datetime import date

import pandas as pd

from nowcast_data.models.target_policy import (
    TargetPolicy,
    resolve_quarterly_final_target,
)
from nowcast_data.pit.adapters.alphaforge import AlphaForgePITAdapter


def test_resolve_quarterly_final_target_defaults_to_third_release(pit_context) -> None:
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
    value, meta = resolve_quarterly_final_target(
        adapter,
        series_key="GDPC1",
        ref_quarter="2025Q1",
        evaluation_asof_date=date(2025, 7, 15),
    )
    assert value == 1.3
    assert meta["selected_release_rank"] == 3
    assert meta["target_release_rank"] is None


def test_resolve_quarterly_final_target_no_cap_uses_last(pit_context) -> None:
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
    value, meta = resolve_quarterly_final_target(
        adapter,
        series_key="GDPC1",
        ref_quarter="2025Q1",
        evaluation_asof_date=date(2025, 7, 15),
        policy=TargetPolicy(mode="latest_available", max_release_rank=None),
    )
    assert value == 1.3
    assert meta["selected_release_rank"] == 3
    assert meta["target_release_rank"] is None

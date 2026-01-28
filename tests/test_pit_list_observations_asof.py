from __future__ import annotations

from datetime import date

import pandas as pd

from nowcast_data.pit.adapters.alphaforge import AlphaForgePITAdapter


def test_list_pit_observations_asof(pit_context) -> None:
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
                    "value": 1.1,
                },
            ]
        )
    )

    adapter = AlphaForgePITAdapter(ctx=pit_context)

    early = adapter.list_pit_observations_asof(
        series_key="GDPC1",
        obs_date=date(2025, 3, 31),
        asof_date=date(2025, 5, 1),
    )
    assert len(early) == 1
    assert early.iloc[0]["value"] == 1.0

    late = adapter.list_pit_observations_asof(
        series_key="GDPC1",
        obs_date=date(2025, 3, 31),
        asof_date=date(2025, 6, 1),
    )
    assert len(late) == 2
    assert late["value"].tolist() == [1.0, 1.1]

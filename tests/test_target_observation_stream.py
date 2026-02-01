from datetime import date

import pandas as pd

from nowcast_data.models.target_policy import get_quarterly_release_observation_stream
from nowcast_data.pit.adapters.alphaforge import AlphaForgePITAdapter


def test_quarterly_release_observation_stream_ranks(pit_context) -> None:
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
    stream = get_quarterly_release_observation_stream(
        adapter,
        series_key="GDPC1",
        ref_quarter="2025Q1",
        asof_date=date(2025, 6, 1),
    )
    assert stream["release_rank"].tolist() == [1, 2]
    assert stream["value"].tolist() == [1.0, 1.2]
    assert stream["ref_quarter"].tolist() == ["2025Q1", "2025Q1"]

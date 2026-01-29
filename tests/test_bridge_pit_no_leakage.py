from datetime import date

import pandas as pd

from nowcast_data.models.bridge import build_rt_quarterly_dataset
from nowcast_data.pit.adapters.alphaforge import AlphaForgePITAdapter


def test_bridge_builder_no_leakage(pit_context) -> None:
    adapter = AlphaForgePITAdapter(ctx=pit_context)
    asof_date = date(2025, 5, 15)

    pit_rows = [
        {
            "series_key": "BASE_GDP",
            "obs_date": "2025-03-31",
            "asof_utc": "2025-05-01",
            "value": 1.0,
        },
        {
            "series_key": "P1",
            "obs_date": "2025-04-30",
            "asof_utc": "2025-05-10",
            "value": 100.0,
        },
        {
            "series_key": "P1",
            "obs_date": "2025-05-31",
            "asof_utc": "2025-05-10",
            "value": 101.0,
        },
        {
            "series_key": "P1",
            "obs_date": "2025-04-30",
            "asof_utc": "2025-06-10",
            "value": 999.0,
        },
        {
            "series_key": "P1",
            "obs_date": "2025-05-31",
            "asof_utc": "2025-06-10",
            "value": 998.0,
        },
    ]

    pit_context.pit.upsert_pit_observations(pd.DataFrame(pit_rows))

    dataset, _, _ = build_rt_quarterly_dataset(
        adapter,
        None,
        target_series_key="BASE_GDP",
        predictor_series_keys=["P1"],
        agg_spec={"P1": "last"},
        asof_date=asof_date,
        include_partial_quarters=True,
    )

    current_quarter = pd.Period("2025Q2", freq="Q")
    # 2025-05-31 is after asof_date (2025-05-15) and should not be included.
    assert dataset.loc[current_quarter, "P1"] == 100.0

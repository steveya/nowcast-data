from datetime import date

import pandas as pd

from alphaforge.time.ref_period import RefFreq
from nowcast_data.pit.adapters.alphaforge import AlphaForgePITAdapter
from nowcast_data.time.nowcast_calendar import (
    infer_current_quarter,
    infer_previous_quarter,
    refperiod_to_quarter_end,
    get_target_asof_ref,
)


def test_infer_current_quarter_boundaries() -> None:
    assert str(infer_current_quarter(date(2025, 1, 1))) == "2025Q1"
    assert str(infer_current_quarter(date(2025, 3, 31))) == "2025Q1"
    assert str(infer_current_quarter(date(2025, 4, 1))) == "2025Q2"


def test_infer_previous_quarter_boundaries() -> None:
    assert str(infer_previous_quarter(date(2025, 1, 1))) == "2024Q4"
    assert str(infer_previous_quarter(date(2025, 3, 31))) == "2024Q4"
    assert str(infer_previous_quarter(date(2025, 4, 1))) == "2025Q1"


def test_refperiod_to_quarter_end() -> None:
    assert refperiod_to_quarter_end(infer_current_quarter(date(2025, 1, 1))) == date(
        2025, 3, 31
    )


def test_get_target_asof_ref_matches_vintage(pit_context) -> None:
    adapter = AlphaForgePITAdapter(ctx=pit_context)
    df = pd.DataFrame(
        [
            {
                "series_key": "GDP",
                "obs_date": "2024-12-31",
                "asof_utc": "2025-01-15",
                "value": 3.0,
            },
            {
                "series_key": "GDP",
                "obs_date": "2024-12-31",
                "asof_utc": "2025-03-01",
                "value": 3.5,
            },
        ]
    )
    pit_context.pit.upsert_pit_observations(df)

    ref = infer_previous_quarter(date(2025, 2, 15))
    value = get_target_asof_ref(
        adapter,
        "GDP",
        asof_date=date(2025, 2, 15),
        ref=ref,
        freq=RefFreq.Q,
    )
    assert value == 3.0

    value_latest = get_target_asof_ref(
        adapter,
        "GDP",
        asof_date=date(2025, 3, 15),
        ref=ref,
        freq=RefFreq.Q,
    )
    assert value_latest == 3.5

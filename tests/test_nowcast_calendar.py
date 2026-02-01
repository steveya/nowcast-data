from datetime import date

import pandas as pd
import pytest
from alphaforge.time.ref_period import RefFreq

from nowcast_data.pit.adapters.alphaforge import AlphaForgePITAdapter
from nowcast_data.pit.adapters.base import PITAdapter
from nowcast_data.time.nowcast_calendar import (
    get_target_asof_ref,
    infer_current_quarter,
    infer_previous_quarter,
    refperiod_to_quarter_end,
)


def test_infer_current_quarter_boundaries() -> None:
    assert str(infer_current_quarter(date(2025, 1, 1))) == "2025Q1"
    assert str(infer_current_quarter(date(2025, 3, 31))) == "2025Q1"
    assert str(infer_current_quarter(date(2025, 4, 1))) == "2025Q2"
    assert str(infer_current_quarter(date(2025, 6, 30))) == "2025Q2"
    assert str(infer_current_quarter(date(2025, 9, 30))) == "2025Q3"
    assert str(infer_current_quarter(date(2025, 12, 31))) == "2025Q4"
    assert str(infer_current_quarter(date(2025, 8, 15))) == "2025Q3"


def test_infer_previous_quarter_boundaries() -> None:
    assert str(infer_previous_quarter(date(2025, 1, 1))) == "2024Q4"
    assert str(infer_previous_quarter(date(2025, 3, 31))) == "2024Q4"
    assert str(infer_previous_quarter(date(2025, 4, 1))) == "2025Q1"
    assert str(infer_previous_quarter(date(2025, 12, 31))) == "2025Q3"


def test_refperiod_to_quarter_end() -> None:
    assert refperiod_to_quarter_end(infer_current_quarter(date(2025, 1, 1))) == date(
        2025, 3, 31
    )
    assert refperiod_to_quarter_end(infer_current_quarter(date(2025, 4, 1))) == date(
        2025, 6, 30
    )
    assert refperiod_to_quarter_end(infer_current_quarter(date(2025, 7, 1))) == date(
        2025, 9, 30
    )
    assert refperiod_to_quarter_end(infer_current_quarter(date(2025, 10, 1))) == date(
        2025, 12, 31
    )


def test_refperiod_to_quarter_end_accepts_string() -> None:
    assert refperiod_to_quarter_end("2025Q1") == date(2025, 3, 31)
    assert refperiod_to_quarter_end("2025Q2") == date(2025, 6, 30)
    assert refperiod_to_quarter_end("2025Q3") == date(2025, 9, 30)
    assert refperiod_to_quarter_end("2025Q4") == date(2025, 12, 31)


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


def test_get_target_asof_ref_missing_returns_none(pit_context) -> None:
    adapter = AlphaForgePITAdapter(ctx=pit_context)
    ref = infer_previous_quarter(date(2025, 2, 15))
    value = get_target_asof_ref(
        adapter,
        "GDP",
        asof_date=date(2024, 12, 1),
        ref=ref,
        freq=RefFreq.Q,
    )
    assert value is None


def test_get_target_asof_ref_multiple_observations_raises(pit_context, monkeypatch) -> None:
    adapter = AlphaForgePITAdapter(ctx=pit_context)
    ref = infer_previous_quarter(date(2025, 2, 15))

    def _multi_snap(*args, **kwargs):
        return pd.Series([1.0, 2.0], index=pd.to_datetime(["2024-12-31", "2025-03-31"]))

    monkeypatch.setattr(adapter._layer, "snapshot_ref", _multi_snap)

    with pytest.raises(ValueError, match="Expected single observation"):
        get_target_asof_ref(
            adapter,
            "GDP",
            asof_date=date(2025, 2, 15),
            ref=ref,
            freq=RefFreq.Q,
        )


def test_refperiod_to_quarter_end_rejects_invalid() -> None:
    with pytest.raises(ValueError, match="Expected quarterly RefPeriod"):
        refperiod_to_quarter_end("2025M01")
    with pytest.raises(ValueError, match="Invalid quarter"):
        refperiod_to_quarter_end("2025Q5")


def test_get_target_asof_ref_requires_adapter_override() -> None:
    class _DummyAdapter(PITAdapter):
        @property
        def name(self) -> str:
            return "Dummy"

        def supports_pit(self, series_id: str) -> bool:
            return False

        def list_vintages(self, series_id: str):  # type: ignore[override]
            return []

        def fetch_asof(  # type: ignore[override]
            self,
            series_id: str,
            asof_date: date,
            start: date | None = None,
            end: date | None = None,
            *,
            metadata=None,
        ):
            return []

    adapter = _DummyAdapter()

    with pytest.raises(NotImplementedError, match="Dummy"):
        get_target_asof_ref(
            adapter,
            "GDP",
            asof_date=date(2025, 1, 1),
            ref="2025Q1",
        )

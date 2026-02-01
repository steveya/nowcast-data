from datetime import date

from nowcast_data.pit.adapters.base import PITAdapter
from nowcast_data.time.nowcast_calendar import get_target_asof_ref


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

    def fetch_asof_ref(  # type: ignore[override]
        self,
        series_id: str,
        asof_date: date,
        start_ref=None,
        end_ref=None,
        *,
        freq=None,
        metadata=None,
    ):
        return []


def test_get_target_asof_ref_fallback_handles_missing_method() -> None:
    adapter = _DummyAdapter()
    value = get_target_asof_ref(
        adapter,
        "GDP",
        asof_date=date(2025, 1, 15),
        ref="2025Q1",
    )
    assert value is None

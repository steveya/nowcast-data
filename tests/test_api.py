from datetime import date
from pathlib import Path

import pytest

pytest.importorskip("alphaforge")
from nowcast_data.pit.api import PITDataManager  # noqa: E402
from nowcast_data.pit.adapters.alphaforge import AlphaForgePITAdapter  # noqa: E402
from nowcast_data.pit.core.catalog import SeriesCatalog  # noqa: E402


def test_get_series_asof_with_alphaforge_adapter(pit_context, monkeypatch) -> None:
    """Ensure manager passthrough uses PIT snapshot without ingestion side-effects."""
    catalog_path = Path(__file__).parent.parent / "series_catalog.yaml"
    catalog = SeriesCatalog(catalog_path)
    mock_fred_source = object()
    pit_context.sources["fred"] = mock_fred_source  # placeholder to exercise ingest toggle

    def _fail_fetch_panel(*args, **kwargs):
        raise AssertionError("fetch_panel should not be called when ingest is disabled")

    monkeypatch.setattr(pit_context, "fetch_panel", _fail_fetch_panel)
    manager = PITDataManager(
        catalog=catalog,
        adapters={"alphaforge": AlphaForgePITAdapter(ctx=pit_context)},
    )

    df = manager.get_series_asof(
        series_key="US_GDP_SAAR",
        asof_date=date(2025, 1, 15),
        start=date(2024, 12, 31),
        end=date(2024, 12, 31),
        ingest_from_ctx_source=False,
    )

    assert df.empty is False
    assert df["value"].iloc[0] == 1.0

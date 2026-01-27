from datetime import date
from pathlib import Path
from nowcast_data.pit.api import PITDataManager
from nowcast_data.pit.adapters.alphaforge import AlphaForgePITAdapter
from nowcast_data.pit.core.catalog import SeriesCatalog



def test_get_series_asof_with_alphaforge_adapter(pit_context) -> None:
    catalog_path = Path(__file__).parent.parent / "series_catalog.yaml"
    catalog = SeriesCatalog(catalog_path)
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

from datetime import date
from pathlib import Path

import pandas as pd
from nowcast_data.pit.api import PITDataManager
from nowcast_data.pit.adapters.alphaforge import AlphaForgePITAdapter
from nowcast_data.pit.core.catalog import SeriesCatalog



def test_get_usgdp_point_in_time(pit_context) -> None:
    series_key = "US_GDP_SAAR"
    catalog_path = Path(__file__).parent.parent / "series_catalog.yaml"
    catalog = SeriesCatalog(catalog_path)
    data_manager = PITDataManager(
        catalog=catalog,
        adapters={"alphaforge": AlphaForgePITAdapter(ctx=pit_context)},
    )

    vintage_dates = data_manager.get_series_vintages(series_key)
    assert vintage_dates == [
        date(2025, 1, 10),
        date(2025, 2, 10),
        date(2025, 4, 10),
        date(2025, 5, 10),
    ]

    all_series = []
    for vintage in vintage_dates[:2]:
        df = data_manager.get_series_asof(series_key, vintage)
        df = df.set_index(["asof_date", "obs_date"])["value"]
        all_series.append(df)

    result_series = pd.concat(all_series)

    assert isinstance(result_series, pd.Series)
    assert isinstance(result_series.index, pd.MultiIndex)
    assert result_series.index.nlevels == 2
    assert result_series.index.names == ["asof_date", "obs_date"]
    assert len(result_series.index.get_level_values(0).unique()) == 2

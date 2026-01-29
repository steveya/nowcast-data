from datetime import date

import numpy as np
import pandas as pd

from nowcast_data.models.bridge import BridgeConfig, BridgeNowcaster
from nowcast_data.pit.adapters.alphaforge import AlphaForgePITAdapter
from nowcast_data.time.nowcast_calendar import infer_current_quarter


def test_bridge_nowcaster_ols_smoke(pit_context) -> None:
    adapter = AlphaForgePITAdapter(ctx=pit_context)
    asof_date = date(2025, 5, 15)

    pit_rows = [
        {
            "series_key": "BASE_GDP",
            "obs_date": "2024-09-30",
            "asof_utc": "2024-11-10",
            "value": 0.8,
        },
        {
            "series_key": "BASE_GDP",
            "obs_date": "2024-12-31",
            "asof_utc": "2025-02-10",
            "value": 1.0,
        },
        {
            "series_key": "BASE_GDP",
            "obs_date": "2025-03-31",
            "asof_utc": "2025-05-01",
            "value": 1.2,
        },
    ]
    for series_key, base_value in [("P1", 100.0), ("P2", 200.0)]:
        for idx, obs_date in enumerate(
            [
                "2024-10-31",
                "2024-11-30",
                "2024-12-31",
                "2025-01-31",
                "2025-02-28",
                "2025-03-31",
                "2025-04-30",
            ]
        ):
            pit_rows.append(
                {
                    "series_key": series_key,
                    "obs_date": obs_date,
                    "asof_utc": "2025-05-10",
                    "value": base_value + idx,
                }
            )

    pit_context.pit.upsert_pit_observations(pd.DataFrame(pit_rows))

    config = BridgeConfig(
        target_series_key="BASE_GDP",
        predictor_series_keys=["P1", "P2"],
        agg_spec={"P1": "mean", "P2": "mean"},
        min_train_quarters=2,
        model="ols",
    )
    nowcaster = BridgeNowcaster(config, adapter)

    result = nowcaster.fit_predict_one(asof_date)

    assert pd.notna(result["y_pred"])
    assert result["ref_quarter"] == str(infer_current_quarter(asof_date))
    assert np.isnan(result["alpha_selected"])

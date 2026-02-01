from datetime import date

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("alphaforge")
from nowcast_data.models.target_features import (  # noqa: E402
    QuarterlyTargetFeatureSpec,
    get_quarterly_target_release_features,
)
from nowcast_data.pit.adapters.alphaforge import AlphaForgePITAdapter  # noqa: E402


def test_quarterly_target_release_features_between_releases(pit_context) -> None:
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
    spec = QuarterlyTargetFeatureSpec()
    features, meta = get_quarterly_target_release_features(
        adapter,
        series_key="GDPC1",
        ref_quarter="2025Q1",
        asof_date=date(2025, 5, 1),
        spec=spec,
    )
    assert features["GDPC1.rel1"] == 1.0
    assert np.isnan(features["GDPC1.rel2"])
    assert features["GDPC1.latest"] == 1.0
    assert features["GDPC1.n_releases"] == 1.0
    assert np.isnan(features["GDPC1.rel2_minus_rel1"])
    assert meta["n_releases_available"] == 1

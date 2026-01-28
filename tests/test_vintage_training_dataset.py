from datetime import date

import numpy as np
import pandas as pd

from nowcast_data.models.datasets import (
    VintageTrainingDatasetConfig,
    build_vintage_training_dataset,
)
from nowcast_data.models.target_features import QuarterlyTargetFeatureSpec
from nowcast_data.models.target_policy import TargetPolicy
from nowcast_data.pit.adapters.alphaforge import AlphaForgePITAdapter


def test_build_vintage_training_dataset_with_target_features(pit_context) -> None:
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
                {
                    "series_key": "P1",
                    "obs_date": "2025-01-31",
                    "asof_utc": "2025-04-30",
                    "value": 101.0,
                },
                {
                    "series_key": "P1",
                    "obs_date": "2025-02-28",
                    "asof_utc": "2025-04-30",
                    "value": 102.0,
                },
                {
                    "series_key": "P1",
                    "obs_date": "2025-03-31",
                    "asof_utc": "2025-04-30",
                    "value": 103.0,
                },
                {
                    "series_key": "P1",
                    "obs_date": "2025-04-30",
                    "asof_utc": "2025-04-30",
                    "value": 104.0,
                },
            ]
        )
    )

    adapter = AlphaForgePITAdapter(ctx=pit_context)
    config = VintageTrainingDatasetConfig(
        target_series_key="GDPC1",
        predictor_series_keys=["P1"],
        agg_spec={"P1": "last"},
        ref_offsets=[-1, 0, 1],
        evaluation_asof_date=date(2025, 6, 1),
        final_target_policy=TargetPolicy(mode="latest_available", max_release_rank=3),
        target_feature_spec=QuarterlyTargetFeatureSpec(),
    )

    dataset, meta = build_vintage_training_dataset(
        adapter,
        None,
        config=config,
        asof_date=date(2025, 5, 1),
    )

    q1 = pd.Period("2025Q1", freq="Q")
    q2 = pd.Period("2025Q2", freq="Q")
    assert q1 in dataset.index
    assert q2 in dataset.index
    assert dataset.loc[q1, "y_asof_latest"] == 1.0
    assert dataset.loc[q1, "y_final"] == 1.2
    assert dataset.loc[q1, "GDPC1.rel1"] == 1.0
    assert np.isnan(dataset.loc[q1, "GDPC1.rel2"])
    assert dataset.loc[q1, "GDPC1.latest"] == 1.0
    assert dataset.loc[q1, "GDPC1.n_releases"] == 1.0
    assert dataset.loc[q1, "P1"] == 103.0
    assert dataset.loc[q2, "P1"] == 104.0
    assert meta["current_ref_quarter"] == "2025Q2"

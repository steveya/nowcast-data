from datetime import date

import pandas as pd

from nowcast_data.models.bridge import build_rt_quarterly_dataset
from nowcast_data.models.datasets import (
    VintageTrainingDatasetConfig,
    build_vintage_training_dataset,
)
from nowcast_data.pit.adapters.alphaforge import AlphaForgePITAdapter


def test_weekly_predictor_without_metadata_does_not_raise(pit_context) -> None:
    pit_context.pit.upsert_pit_observations(
        pd.DataFrame(
            [
                {
                    "series_key": "BASE_GDP",
                    "obs_date": "2024-12-31",
                    "asof_utc": "2025-02-10",
                    "value": 1.0,
                },
                {
                    "series_key": "W1",
                    "obs_date": "2025-02-07",
                    "asof_utc": "2025-02-10",
                    "value": 10.0,
                },
                {
                    "series_key": "W1",
                    "obs_date": "2025-02-14",
                    "asof_utc": "2025-02-17",
                    "value": 12.0,
                },
            ]
        )
    )

    adapter = AlphaForgePITAdapter(ctx=pit_context)
    dataset, _, _ = build_rt_quarterly_dataset(
        adapter,
        None,
        target_series_key="BASE_GDP",
        predictor_series_keys=["W1"],
        agg_spec={"W1": "mean"},
        asof_date=date(2025, 2, 20),
        include_partial_quarters=True,
    )
    assert "W1" in dataset.columns

    config = VintageTrainingDatasetConfig(
        target_series_key="BASE_GDP",
        predictor_series_keys=["W1"],
        agg_spec={"W1": "mean"},
        evaluation_asof_date=date(2025, 3, 1),
    )
    training, _ = build_vintage_training_dataset(
        adapter,
        None,
        config=config,
        asof_date=date(2025, 2, 20),
    )
    assert "W1" in training.columns

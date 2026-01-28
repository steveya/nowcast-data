from __future__ import annotations

from datetime import date
from pathlib import Path

from nowcast_data.models.datasets import (
    VintageTrainingDatasetConfig,
    build_vintage_training_dataset,
)
from nowcast_data.models.target_features import QuarterlyTargetFeatureSpec
from nowcast_data.models.target_policy import TargetPolicy
from nowcast_data.pit.api import PITDataManager
from nowcast_data.pit.core.catalog import SeriesCatalog


def main() -> None:
    catalog_path = Path(__file__).parent.parent / "series_catalog.yaml"
    catalog = SeriesCatalog(catalog_path)
    manager = PITDataManager(catalog)

    asof_date = date(2025, 5, 15)
    evaluation_asof_date = date(2025, 7, 15)

    config = VintageTrainingDatasetConfig(
        target_series_key="US_GDP_SAAR",
        predictor_series_keys=["US_CPI", "US_UNRATE"],
        agg_spec={"US_CPI": "mean", "US_UNRATE": "mean"},
        ref_offsets=[-1, 0, 1],
        evaluation_asof_date=evaluation_asof_date,
        final_target_policy=TargetPolicy(mode="latest_available", max_release_rank=3),
        target_feature_spec=QuarterlyTargetFeatureSpec(),
    )

    dataset, meta = build_vintage_training_dataset(
        manager.adapters["alphaforge"],
        catalog,
        config=config,
        asof_date=asof_date,
        ingest_from_ctx_source=True,
    )

    print("Vintage training dataset:")
    print(dataset)
    print("\nMetadata:")
    print(meta)


if __name__ == "__main__":
    main()

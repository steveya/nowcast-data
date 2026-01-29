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

    print("=" * 80)
    print("VINTAGE TRAINING DATASET")
    print("=" * 80)
    print(f"\nAsof date (vintage): {asof_date}")
    print(f"Evaluation asof date (for final target): {evaluation_asof_date}")
    print(f"\nDataset shape: {dataset.shape}")
    print(f"\nTarget series key: {config.target_series_key}")
    print(f"Reference quarter offsets: {config.ref_offsets}")
    print(f"Includes target release features: {config.target_feature_spec is not None}")

    print("\n" + "-" * 80)
    print("FULL DATASET (all columns)")
    print("-" * 80)
    print(dataset)

    print("\n" + "-" * 80)
    print("TARGET LABEL COMPARISON: y_asof_latest vs y_final")
    print("-" * 80)
    label_cols = ["y_asof_latest", "y_final"]
    if all(col in dataset.columns for col in label_cols):
        label_df = dataset[label_cols].copy()
        label_df["difference"] = label_df["y_final"] - label_df["y_asof_latest"]
        print(label_df)
        print(f"\nMean difference (y_final - y_asof_latest): {label_df['difference'].mean():.4f}")

    print("\n" + "-" * 80)
    print("SAMPLE COLUMNS (ref_offsets -1, 0, 1)")
    print("-" * 80)
    display_cols = ["y_asof_latest", "y_final"] + [
        col
        for col in dataset.columns
        if col not in ["y_asof_latest", "y_final"]
        and not col.startswith(f"{config.target_series_key}.")
    ][:3]
    display_cols = [col for col in display_cols if col in dataset.columns]
    print(dataset[display_cols])

    if any(
        col.startswith(f"{config.target_series_key}.") and col not in ["y_asof_latest", "y_final"]
        for col in dataset.columns
    ):
        print("\n" + "-" * 80)
        print("TARGET RELEASE FEATURES")
        print("-" * 80)
        trf_cols = [
            col
            for col in dataset.columns
            if col.startswith(f"{config.target_series_key}.")
            and col not in ["y_asof_latest", "y_final"]
        ]
        print(f"Number of target release features: {len(trf_cols)}")
        print(f"Feature names: {trf_cols[:5]}{'...' if len(trf_cols) > 5 else ''}")
        print(f"\nSample feature values:")
        print(dataset[trf_cols[:3]])

    print("\n" + "-" * 80)
    print("METADATA")
    print("-" * 80)
    print(f"\nCurrent reference quarter: {meta['current_ref_quarter']}")
    print(f"Observations in current quarter: {meta['nobs_current']}")
    print(f"Last observation dates (current quarter): {meta['last_obs_date_current_quarter']}")


if __name__ == "__main__":
    main()

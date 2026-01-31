from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Iterable

import pandas as pd

from nowcast_data.models.datasets import VintageTrainingDatasetConfig
from nowcast_data.models.panel import build_vintage_panel_dataset
from nowcast_data.models.target_features import QuarterlyTargetFeatureSpec
from nowcast_data.models.target_policy import TargetPolicy
from nowcast_data.pit.api import PITDataManager
from nowcast_data.pit.core.catalog import SeriesCatalog


def _collect_event_dates(
    manager: PITDataManager,
    series_keys: Iterable[str],
    *,
    asof_start: date,
    asof_end: date,
) -> list[date]:
    if "alphaforge" not in manager.adapters:
        raise ValueError("alphaforge adapter is required to list vintages")

    adapter = manager.adapters["alphaforge"]
    event_dates: set[date] = set()
    for series_key in series_keys:
        try:
            vintages = adapter.list_vintages(series_key)
        except Exception as exc:
            print(f"Warning: failed to list vintages for {series_key}: {exc}")
            continue
        for v in vintages:
            if asof_start <= v <= asof_end:
                event_dates.add(v)
    return sorted(event_dates)


def _expand_to_daily_grid(
    event_df: pd.DataFrame,
    *,
    asof_start: date,
    asof_end: date,
) -> pd.DataFrame:
    daily_index = pd.bdate_range(start=asof_start, end=asof_end).date
    event_dates = pd.Index(event_df.index)
    daily = event_df.reindex(daily_index).ffill()
    daily["is_event_date"] = daily.index.isin(event_dates)

    event_id = pd.Series(index=daily.index, dtype="float64")
    event_id.loc[daily["is_event_date"]] = range(1, int(daily["is_event_date"].sum()) + 1)
    event_id = event_id.ffill().fillna(0).astype(int)
    daily["event_id"] = event_id

    last_event_date = pd.Series(index=daily.index, dtype="datetime64[ns]")
    last_event_date.loc[daily["is_event_date"]] = pd.to_datetime(daily.index)
    last_event_date = last_event_date.ffill()
    daily["since_event_days"] = (
        (pd.to_datetime(daily.index) - last_event_date).dt.days.fillna(0).astype(int)
    )

    block_sizes = daily.groupby("event_id").size()
    daily["sample_weight"] = daily["event_id"].map(block_sizes).rdiv(1.0)
    return daily


def main() -> None:
    catalog_path = Path(__file__).parent.parent / "series_catalog.yaml"
    catalog = SeriesCatalog(catalog_path)
    manager = PITDataManager(catalog)

    asof_start = date(2025, 1, 1)
    asof_end = date(2025, 7, 15)
    grid_mode = "event"  # "event" or "daily"
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

    series_keys = [config.target_series_key, *config.predictor_series_keys]
    event_dates = _collect_event_dates(
        manager,
        series_keys,
        asof_start=asof_start,
        asof_end=asof_end,
    )
    if not event_dates:
        raise ValueError("No event dates found in the requested window")

    dataset, meta = build_vintage_panel_dataset(
        manager.adapters["alphaforge"],
        catalog,
        config=config,
        vintages=event_dates,
        ingest_from_ctx_source=True,
    )

    dropped = 0
    if "y_asof_latest" in dataset.columns:
        non_null_mask = dataset["y_asof_latest"].notna()
        if non_null_mask.sum() == 0:
            print(
                "Warning: y_asof_latest is NaN for all event dates; "
                "dropping rows with NaN targets."
            )
        dropped = int((~non_null_mask).sum())
        dataset = dataset.loc[non_null_mask]

    if grid_mode == "daily":
        dataset = _expand_to_daily_grid(
            dataset,
            asof_start=asof_start,
            asof_end=asof_end,
        )

    print("=" * 80)
    print("VINTAGE TRAINING DATASET")
    print("=" * 80)
    print(f"\nAs-of window: {asof_start} to {asof_end}")
    print(f"Grid mode: {grid_mode}")
    print(f"Evaluation asof date (for final target): {evaluation_asof_date}")
    print(f"\nDataset shape: {dataset.shape}")
    print(f"Event rows: {len(event_dates)}")
    if grid_mode == "daily":
        print(f"Daily rows: {len(dataset)}")
        print(
            "Warning: daily rows are forward-filled; do not treat as independent samples. "
            "Use is_event_date or sample_weight."
        )
    if dropped > 0:
        print(f"Dropped {dropped} rows with NaN targets")
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
    if not meta.empty:
        meta_row = meta.iloc[-1]
        print(f"\nCurrent reference quarter: {meta_row['current_ref_quarter']}")
        print(f"Observations in current quarter: {meta_row['nobs_current_json']}")
        print("Last observation dates (current quarter): " f"{meta_row['last_obs_date_json']}")


if __name__ == "__main__":
    main()

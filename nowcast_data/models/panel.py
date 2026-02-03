"""Vintage panel dataset construction for walk-forward backtesting.

This module builds panel datasets indexed by vintage date (asof_date), where each
row contains the features and target values available at that vintage. This enables
proper walk-forward backtesting where training happens on historical vintages.
"""

from __future__ import annotations

import json
from datetime import date
from typing import Iterable

import numpy as np
import pandas as pd

from nowcast_data.models.datasets import (
    VintageTrainingDatasetConfig,
    build_vintage_training_dataset,
)
from nowcast_data.pit.adapters.base import PITAdapter
from nowcast_data.pit.core.catalog import SeriesCatalog


def build_vintage_panel_dataset(
    adapter: PITAdapter,
    catalog: SeriesCatalog | None,
    config: VintageTrainingDatasetConfig,
    vintages: Iterable[date],
    *,
    ingest_from_ctx_source: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a panel dataset indexed by vintage date for walk-forward backtesting.

    This function calls build_vintage_training_dataset once per vintage date and
    stacks the results into a panel dataset. Each row represents a vintage point-in-time
    snapshot with predictor features and target values. Daily predictors (frequency "d"
    or "b") are expanded into multiple quarterly features via the dataset builder.

    Args:
        adapter: PIT adapter for data access.
        catalog: Optional series catalog for metadata lookup.
        config: Vintage training dataset configuration. Note: ref_offsets should
            typically be [0] to get only the current quarter's row per vintage.
        vintages: Iterable of vintage dates to process.
        ingest_from_ctx_source: Whether to allow ingestion from context sources.
    Returns:
        Tuple of (Xy_panel, meta_panel):

        Xy_panel: DataFrame indexed by asof_date containing:
            - ref_quarter: Reference quarter for this vintage
            - Predictor feature columns
            - Expanded daily feature columns (e.g., "DAILY_FCI.last")
            - y_asof_latest: Latest target value known at vintage
            - y_final: Final target value (from evaluation_asof_date)

        meta_panel: DataFrame indexed by asof_date with diagnostic columns:
            - current_ref_quarter: Current quarter at this vintage
            - nobs_current_json: JSON string of observation counts per predictor
            - last_obs_date_json: JSON string of last obs dates per predictor

    Raises:
        ValueError: If config.evaluation_asof_date is None (required for y_final).

    Note:
        Features NEVER include y_final to prevent leakage. The y_asof_latest column
        remains a label-only column and is not used as a model feature.
    """
    vintages = sorted(set(vintages))
    if not vintages:
        # Return empty DataFrames with expected structure
        return pd.DataFrame(), pd.DataFrame()

    if config.evaluation_asof_date is None:
        raise ValueError(
            "config.evaluation_asof_date must be provided for vintage panel construction"
        )

    # Collect rows from each vintage
    xy_rows: list[dict] = []
    meta_rows: list[dict] = []

    for vintage_date in vintages:
        try:
            dataset, meta = build_vintage_training_dataset(
                adapter,
                catalog,
                config=config,
                asof_date=vintage_date,
                ingest_from_ctx_source=ingest_from_ctx_source,
            )
        except Exception:
            # Skip vintages that fail (e.g., no data)
            continue

        if dataset.empty:
            continue

        # Get the current quarter row (offset=0)
        current_quarter = pd.Period(meta["current_ref_quarter"], freq="Q")
        if current_quarter not in dataset.index:
            continue

        row = dataset.loc[current_quarter].to_dict()
        row["asof_date"] = vintage_date
        row["ref_quarter"] = str(current_quarter)

        xy_rows.append(row)

        # Build meta row
        meta_row = {
            "asof_date": vintage_date,
            "current_ref_quarter": meta["current_ref_quarter"],
            "nobs_current_json": json.dumps(meta.get("nobs_current", {})),
            "last_obs_date_json": json.dumps(
                {
                    k: str(v) if v else None
                    for k, v in meta.get("last_obs_date_current_quarter", {}).items()
                }
            ),
        }
        meta_rows.append(meta_row)

    if not xy_rows:
        return pd.DataFrame(), pd.DataFrame()

    # Build DataFrames
    xy_panel = pd.DataFrame(xy_rows)
    meta_panel = pd.DataFrame(meta_rows)

    # Set index to asof_date
    xy_panel = xy_panel.set_index("asof_date").sort_index()
    meta_panel = meta_panel.set_index("asof_date").sort_index()

    # Ensure column alignment: union of all columns, stable order
    # Move label columns to the end for clarity
    label_cols = ["y_asof_latest", "y_final"]
    meta_cols = ["ref_quarter"]

    feature_cols = [col for col in xy_panel.columns if col not in label_cols + meta_cols]
    feature_cols = sorted(feature_cols)

    # Order: meta_cols, feature_cols, indicator_cols (if present), label_cols
    final_col_order = meta_cols.copy()
    final_col_order.extend(feature_cols)
    final_col_order.extend([c for c in label_cols if c in xy_panel.columns])

    xy_panel = xy_panel[[c for c in final_col_order if c in xy_panel.columns]]

    return xy_panel, meta_panel


def get_feature_columns(xy_panel: pd.DataFrame) -> list[str]:
    """Extract feature column names from a vintage panel dataset.

    Args:
        xy_panel: Panel dataset from build_vintage_panel_dataset.
    Returns:
        List of feature column names (excludes labels and metadata columns).
    """
    exclude_cols = {"ref_quarter", "y_asof_latest", "y_final"}
    return [col for col in xy_panel.columns if col not in exclude_cols]


def preprocess_panel_for_training(
    xy_panel: pd.DataFrame,
    train_vintages: list[date],
    test_vintage: date,
    *,
    feature_cols: list[str],
    label_col: str,
    max_nan_fraction: float = 0.5,
    standardize: bool = True,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series | None, dict]:
    """Preprocess panel data for walk-forward training.

    This function:
    1. Drops features with too many NaNs in the training set
    2. Imputes remaining NaNs using training set means
    3. Optionally standardizes features using training set statistics

    Args:
        xy_panel: Full panel dataset from build_vintage_panel_dataset.
        train_vintages: List of vintage dates for training.
        test_vintage: Vintage date for testing.
        feature_cols: List of feature column names.
        label_col: Target column name ("y_asof_latest" or "y_final").
        max_nan_fraction: Drop features with more than this fraction NaNs in training.
        standardize: Whether to standardize features.
    Returns:
        Tuple of (train_X, train_y, test_X, test_y, stats):
        - train_X: Preprocessed training features
        - train_y: Training labels
        - test_X: Preprocessed test features (single row)
        - test_y: Test label (may be NaN) or None if test_vintage not in panel
        - stats: Dict with preprocessing statistics (means, stds, dropped_cols)
    """
    # Get training and test data
    train_mask = xy_panel.index.isin(train_vintages)
    train_data = xy_panel.loc[train_mask].copy()

    if test_vintage in xy_panel.index:
        test_data = xy_panel.loc[[test_vintage]].copy()
    else:
        test_data = None

    # Extract features and labels
    available_feature_cols = [c for c in feature_cols if c in train_data.columns]

    train_X = train_data[available_feature_cols].copy()
    train_y = (
        train_data[label_col].copy() if label_col in train_data.columns else pd.Series(dtype=float)
    )

    # Drop high-NaN features based on training set
    nan_frac = train_X.isna().mean()
    keep_cols = nan_frac[nan_frac <= max_nan_fraction].index.tolist()
    dropped_cols = [c for c in available_feature_cols if c not in keep_cols]

    train_X = train_X[keep_cols]

    # Compute training means for imputation
    means = train_X.mean()

    # Impute training set
    train_X = train_X.fillna(means)

    # Standardization
    if standardize and not train_X.empty:
        stds = train_X.std(ddof=0).replace(0.0, 1.0)
        train_X = (train_X - train_X.mean()) / stds
    else:
        stds = pd.Series(1.0, index=train_X.columns)

    stats = {
        "means": means.to_dict(),
        "stds": stds.to_dict(),
        "dropped_cols": dropped_cols,
        "kept_cols": keep_cols,
    }

    # Process test data
    if test_data is not None:
        test_X = test_data[keep_cols].copy()
        test_X = test_X.fillna(means)
        if standardize and not test_X.empty:
            test_X = (
                test_X - pd.Series(stats["means"]).reindex(test_X.columns).fillna(0)
            ) / stds.reindex(test_X.columns).fillna(1)
        test_y = test_data[label_col].iloc[0] if label_col in test_data.columns else np.nan
    else:
        test_X = pd.DataFrame(columns=keep_cols)
        test_y = None

    return train_X, train_y, test_X, test_y, stats

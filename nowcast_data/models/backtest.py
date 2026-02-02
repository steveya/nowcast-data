"""Backtest runner for nowcasting models across multiple vintage dates."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, RidgeCV

from nowcast_data.models.datasets import VintageTrainingDatasetConfig
from nowcast_data.models.panel import (
    build_vintage_panel_dataset,
    get_feature_columns,
    preprocess_panel_for_training,
)
from nowcast_data.models.target_policy import TargetPolicy
from nowcast_data.pit.adapters.base import PITAdapter
from nowcast_data.pit.api import PITDataManager
from nowcast_data.pit.core.catalog import SeriesCatalog


def make_vintage_dates(
    start_date: date, end_date: date, freq: Literal["D", "W", "B"]
) -> list[date]:
    """Generate a list of vintage dates from start_date to end_date (inclusive).

    Args:
        start_date: Start date.
        end_date: End date (inclusive).
        freq: Frequency - "D" (daily), "W" (weekly), "B" (business-day).

    Returns:
        Sorted list of dates.
    """
    if freq == "D":
        freq_td = timedelta(days=1)
    elif freq == "W":
        freq_td = timedelta(weeks=1)
    elif freq == "B":
        # Business days: skip weekends
        freq_td = None
    else:
        raise ValueError(f"freq must be 'D', 'W', or 'B', got {freq}")

    dates = []
    current = start_date
    while current <= end_date:
        dates.append(current)
        if freq_td is not None:
            current += freq_td
        else:
            # Business day: advance by 1 day until we hit a weekday
            current += timedelta(days=1)
            while current.weekday() >= 5:  # 5=Saturday, 6=Sunday
                current += timedelta(days=1)

    return sorted(dates)


@dataclass
class BacktestConfig:
    """Configuration for walk-forward backtesting.

    Attributes:
        target_series_key: Series key for the nowcasting target.
        predictor_series_keys: Series keys for predictors.
        agg_spec: Aggregation method per series.
        start_date: Start date for vintage dates.
        end_date: End date for vintage dates.
        freq: Frequency for vintage dates ("D", "W", "B").
        label: Which label to use for training ("y_asof_latest" or "y_final").
        evaluation_asof_date: Asof date for final target labels (required if label="y_final").
        model: Regression model ("ridge" or "ols").
        alphas: Ridge CV alphas.
        train_min_periods: Minimum number of vintages required for training.
        rolling_window: If set, use rolling window of this size; else expanding window.
        standardize: Whether to standardize features.
        max_nan_fraction: Drop features with more than this fraction NaNs.
        include_y_asof_latest_as_feature: Include y_asof_latest as a feature with indicator.
        output_csv: Optional path to save results CSV.
        compute_metrics: Whether to compute summary metrics.
        ingest_from_ctx_source: Allow ingestion from context source.
        final_target_policy: Policy for resolving final target values.
    """

    target_series_key: str
    predictor_series_keys: list[str]
    agg_spec: dict[str, str]
    start_date: date
    end_date: date
    freq: Literal["D", "W", "B"] = "W"
    label: Literal["y_asof_latest", "y_final"] = "y_asof_latest"
    evaluation_asof_date: date | None = None
    model: Literal["ridge", "ols"] = "ridge"
    alphas: Sequence[float] = field(default_factory=lambda: (0.01, 0.1, 1.0, 10.0, 100.0))
    train_min_periods: int = 40
    rolling_window: int | None = None
    standardize: bool = True
    max_nan_fraction: float = 0.5
    include_y_asof_latest_as_feature: bool = False
    use_real_time_target_as_feature: bool = True
    real_time_feature_cols: list[str] = field(
        default_factory=lambda: ["y_asof_latest_growth", "y_asof_latest_level"]
    )
    training_label_mode: Literal["stable", "revision"] = "stable"
    stable_label_col: str = "y_final_3rd_growth"
    real_time_label_col: str = "y_asof_latest_growth"
    output_csv: str | None = None
    compute_metrics: bool = True
    ingest_from_ctx_source: bool = False
    final_target_policy: TargetPolicy = field(
        default_factory=lambda: TargetPolicy(mode="latest_available", max_release_rank=3)
    )


def run_backtest(
    adapter: PITAdapter | PITDataManager,
    config: BacktestConfig,
    catalog: SeriesCatalog | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Run walk-forward backtest over multiple vintage dates.

    This implements proper walk-forward backtesting:
    1. Build a panel dataset indexed by vintage date
    2. For each test vintage t, train on historical vintages < t
    3. Predict for vintage t
    4. Compute metrics across all predictions

    Args:
        adapter: PIT adapter or PITDataManager for data access.
        config: Backtest configuration.
        catalog: Optional series catalog for metadata.

    Returns:
        Tuple of (results_dataframe, metrics_dict).

        results_dataframe columns:
        - asof_date: Vintage date
        - ref_quarter: Reference quarter
        - y_pred: Model prediction
        - y_true_asof_latest: Latest target value at vintage
        - y_true_final: Final target value
        - label_used: Which label was used for training
        - n_train: Number of training vintages
        - n_features: Number of features used
        - alpha_selected: Selected alpha (for ridge)
        - error: y_pred - y_true (using label_used)
        - abs_error: |error|

        metrics_dict keys:
        - rmse: Root mean squared error (overall)
        - mae: Mean absolute error (overall)
        - count: Number of valid predictions
        - by_ref_quarter: Dict[str, dict] with RMSE/MAE per quarter
    """
    # Validate config
    if config.label == "y_final" and config.evaluation_asof_date is None:
        raise ValueError("evaluation_asof_date is required when label='y_final'")

    # Extract adapter and catalog
    if isinstance(adapter, PITDataManager):
        pit_adapter = adapter.adapters.get("alphaforge")
        if pit_adapter is None:
            raise ValueError("PITDataManager missing alphaforge adapter")
        effective_adapter = pit_adapter
        effective_catalog = adapter.catalog
    else:
        effective_adapter = adapter
        effective_catalog = catalog

    # Generate vintage dates
    vintages = make_vintage_dates(config.start_date, config.end_date, config.freq)
    if not vintages:
        return pd.DataFrame(), {"rmse": np.nan, "mae": np.nan, "count": 0}

    # Build vintage training config
    # Use ref_offsets=[0] to get current quarter row for each vintage
    vintage_config = VintageTrainingDatasetConfig(
        target_series_key=config.target_series_key,
        predictor_series_keys=config.predictor_series_keys,
        agg_spec=config.agg_spec,
        include_partial_quarters=True,
        ref_offsets=[0],  # Only current quarter
        evaluation_asof_date=config.evaluation_asof_date or config.end_date,
        final_target_policy=config.final_target_policy,
        target_feature_spec=None,
    )

    # Build the full panel dataset
    xy_panel, meta_panel = build_vintage_panel_dataset(
        effective_adapter,
        effective_catalog,
        config=vintage_config,
        vintages=vintages,
        ingest_from_ctx_source=config.ingest_from_ctx_source,
        include_y_asof_latest_as_feature=config.include_y_asof_latest_as_feature,
    )

    if xy_panel.empty:
        return pd.DataFrame(), {"rmse": np.nan, "mae": np.nan, "count": 0}

    # Get feature columns
    feature_cols = get_feature_columns(
        xy_panel, include_y_asof_latest_as_feature=config.include_y_asof_latest_as_feature
    )

    if not config.use_real_time_target_as_feature and config.real_time_feature_cols:
        feature_cols = [col for col in feature_cols if col not in config.real_time_feature_cols]

    # Determine label column
    label_col = config.label  # "y_asof_latest" or "y_final"

    # Walk-forward backtest
    results = []
    panel_vintages = sorted(xy_panel.index.tolist())

    for i, test_vintage in enumerate(panel_vintages):
        # Determine training vintages
        if config.rolling_window is not None:
            # Rolling window
            start_idx = max(0, i - config.rolling_window)
            train_vintages = panel_vintages[start_idx:i]
        else:
            # Expanding window
            train_vintages = panel_vintages[:i]

        # Filter to vintages with non-null labels for training
        train_data = xy_panel.loc[train_vintages]
        if label_col in train_data.columns:
            valid_train_mask = train_data[label_col].notna()
            train_vintages = train_data[valid_train_mask].index.tolist()

        # Check minimum training periods
        if len(train_vintages) < config.train_min_periods:
            # Not enough training data, skip this vintage
            row = {
                "asof_date": test_vintage,
                "ref_quarter": (
                    xy_panel.loc[test_vintage, "ref_quarter"]
                    if "ref_quarter" in xy_panel.columns
                    else ""
                ),
                "y_pred": np.nan,
                "y_true_asof_latest": (
                    xy_panel.loc[test_vintage, "y_asof_latest"]
                    if "y_asof_latest" in xy_panel.columns
                    else np.nan
                ),
                "y_true_final": (
                    xy_panel.loc[test_vintage, "y_final"]
                    if "y_final" in xy_panel.columns
                    else np.nan
                ),
                "label_used": label_col,
                "n_train": len(train_vintages),
                "n_features": 0,
                "alpha_selected": np.nan,
            }
            results.append(row)
            continue

        # Preprocess data
        train_X, train_y, test_X, test_y, stats = preprocess_panel_for_training(
            xy_panel,
            train_vintages,
            test_vintage,
            feature_cols=feature_cols,
            label_col=label_col,
            max_nan_fraction=config.max_nan_fraction,
            standardize=config.standardize,
            include_y_asof_latest_as_feature=config.include_y_asof_latest_as_feature,
        )

        if config.training_label_mode == "revision":
            train_y = (
                xy_panel.loc[train_vintages, config.stable_label_col]
                - xy_panel.loc[train_vintages, config.real_time_label_col]
            )

        # Drop NaN labels from training
        valid_mask = train_y.notna()
        train_X_clean = train_X.loc[valid_mask]
        train_y_clean = train_y.loc[valid_mask]

        if train_X_clean.empty or len(train_y_clean) < 2:
            row = {
                "asof_date": test_vintage,
                "ref_quarter": (
                    xy_panel.loc[test_vintage, "ref_quarter"]
                    if "ref_quarter" in xy_panel.columns
                    else ""
                ),
                "y_pred": np.nan,
                "y_true_asof_latest": (
                    xy_panel.loc[test_vintage, "y_asof_latest"]
                    if "y_asof_latest" in xy_panel.columns
                    else np.nan
                ),
                "y_true_final": (
                    xy_panel.loc[test_vintage, "y_final"]
                    if "y_final" in xy_panel.columns
                    else np.nan
                ),
                "label_used": label_col,
                "n_train": len(train_y_clean),
                "n_features": train_X_clean.shape[1] if not train_X_clean.empty else 0,
                "alpha_selected": np.nan,
            }
            results.append(row)
            continue

        # Fit model
        alpha_selected = np.nan
        if config.model == "ridge":
            model = RidgeCV(alphas=config.alphas)
            model.fit(train_X_clean.to_numpy(), train_y_clean.to_numpy())
            alpha_selected = float(model.alpha_)
        elif config.model == "ols":
            model = LinearRegression(fit_intercept=True)
            model.fit(train_X_clean.to_numpy(), train_y_clean.to_numpy())
        else:
            raise ValueError(f"model must be 'ridge' or 'ols', got {config.model}")

        # Predict
        if test_X.empty:
            y_pred = np.nan
        else:
            y_pred = float(model.predict(test_X.to_numpy().reshape(1, -1))[0])

        y_pred_revision = np.nan
        y_pred_stable = y_pred
        if config.training_label_mode == "revision":
            y_pred_revision = y_pred
            if test_vintage in xy_panel.index:
                rt_value = xy_panel.loc[test_vintage, config.real_time_label_col]
                y_pred_stable = (
                    float(rt_value) + y_pred_revision if pd.notna(rt_value) else np.nan
                )
            else:
                y_pred_stable = np.nan

        # Build result row
        row = {
            "asof_date": test_vintage,
            "ref_quarter": (
                xy_panel.loc[test_vintage, "ref_quarter"]
                if "ref_quarter" in xy_panel.columns
                else ""
            ),
            "y_pred": y_pred,
            "y_pred_stable": y_pred_stable,
            "y_pred_revision": y_pred_revision,
            "y_true_asof_latest": (
                xy_panel.loc[test_vintage, "y_asof_latest"]
                if "y_asof_latest" in xy_panel.columns
                else np.nan
            ),
            "y_true_final": (
                xy_panel.loc[test_vintage, "y_final"] if "y_final" in xy_panel.columns else np.nan
            ),
            "y_true_stable": (
                xy_panel.loc[test_vintage, config.stable_label_col]
                if config.stable_label_col in xy_panel.columns
                else np.nan
            ),
            "y_true_real_time": (
                xy_panel.loc[test_vintage, config.real_time_label_col]
                if config.real_time_label_col in xy_panel.columns
                else np.nan
            ),
            "label_used": label_col,
            "training_label_mode": config.training_label_mode,
            "n_train": len(train_y_clean),
            "n_features": train_X_clean.shape[1],
            "alpha_selected": alpha_selected,
        }
        results.append(row)

    # Build DataFrame
    df = pd.DataFrame(results)
    if df.empty:
        return df, {"rmse": np.nan, "mae": np.nan, "count": 0}

    # Add y_true and error columns based on label_used
    df["y_true"] = (
        df[f"y_true_{label_col.replace('y_', '')}"]
        if f"y_true_{label_col.replace('y_', '')}" in df.columns
        else df["y_true_asof_latest"]
    )
    df["error"] = df["y_pred"] - df["y_true"]
    df["abs_error"] = np.abs(df["error"])

    # Compute metrics
    metrics: dict = {"rmse": np.nan, "mae": np.nan, "count": 0}
    if config.compute_metrics:
        valid_mask = df["y_true"].notna() & df["y_pred"].notna()
        valid_df = df[valid_mask]

        if not valid_df.empty:
            mse = (valid_df["error"] ** 2).mean()
            metrics["rmse"] = float(np.sqrt(mse)) if not np.isnan(mse) else np.nan
            metrics["mae"] = float(valid_df["abs_error"].mean())
            metrics["count"] = len(valid_df)

            # Metrics by ref_quarter
            metrics["by_ref_quarter"] = {}
            for quarter in valid_df["ref_quarter"].unique():
                qdf = valid_df[valid_df["ref_quarter"] == quarter]
                qmse = (qdf["error"] ** 2).mean()
                metrics["by_ref_quarter"][str(quarter)] = {
                    "rmse": float(np.sqrt(qmse)) if not np.isnan(qmse) else np.nan,
                    "mae": float(qdf["abs_error"].mean()),
                    "count": len(qdf),
                }

    if config.compute_metrics:
        metrics["stable_vs_final_3rd_growth"] = _compute_metrics(
            df, pred_col="y_pred_stable", truth_col=config.stable_label_col
        )
        metrics["stable_vs_real_time_growth"] = _compute_metrics(
            df, pred_col="y_pred_stable", truth_col=config.real_time_label_col
        )
        if config.training_label_mode == "revision":
            df["y_true_revision"] = df[config.stable_label_col] - df[config.real_time_label_col]
            metrics["revision_metrics"] = _compute_metrics(
                df, pred_col="y_pred_revision", truth_col="y_true_revision"
            )

    # Save to CSV if requested
    if config.output_csv is not None:
        df.to_csv(config.output_csv, index=False)

    return df, metrics

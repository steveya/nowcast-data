"""Example: Run walk-forward backtest over a date range.

This example demonstrates the proper walk-forward backtesting approach:
1. Build a panel dataset indexed by vintage date (asof_date)
2. For each test vintage t, train on historical vintages < t
3. Predict for vintage t
4. Compute metrics across all predictions

Two label modes are supported:
- "y_asof_latest": Online learning - train on latest available target values
- "y_final": Offline learning - train on final target values (requires evaluation_asof_date)
"""

from __future__ import annotations

from datetime import date

from nowcast_data.models.backtest import BacktestConfig, run_backtest
from nowcast_data.pit.api import PITDataManager
from nowcast_data.pit.core.catalog import SeriesCatalog
from nowcast_data.models.target_policy import TargetPolicy
from pathlib import Path


def main() -> None:
    """Run example walk-forward backtest."""
    # Set up data access
    catalog_path = Path(__file__).parent.parent / "series_catalog.yaml"
    catalog = SeriesCatalog(catalog_path)
    manager = PITDataManager(catalog)

    # Define backtest configuration
    # Example 1: Online learning (label="y_asof_latest")
    # Uses latest available target values at each vintage
    # Note: DAILY_FCI must be declared as frequency "d" (or "b") in series_catalog.yaml
    config = BacktestConfig(
        target_series_key="US_GDP_SAAR",
        predictor_series_keys=["US_CPI", "US_UNRATE", "DAILY_FCI"],
        agg_spec={"US_CPI": "mean", "US_UNRATE": "mean", "DAILY_FCI": "mean"},
        start_date=date(2025, 1, 1),
        end_date=date(2025, 6, 30),
        freq="W",  # Weekly vintages
        label="y_asof_latest",  # Online learning mode
        train_min_periods=4,  # Require at least 4 training vintages
        rolling_window=None,  # Expanding window (use all history)
        include_y_asof_latest_as_feature=False,
        compute_metrics=True,
        output_csv=None,  # Set to a path to save results
    )

    print("=" * 80)
    print("WALK-FORWARD BACKTEST (Online Learning)")
    print("=" * 80)
    print(f"\nTarget: {config.target_series_key}")
    print(f"Predictors: {config.predictor_series_keys}")
    print(f"Period: {config.start_date} to {config.end_date} ({config.freq})")
    print(f"Label: {config.label}")
    print(f"Min training periods: {config.train_min_periods}")

    # Run backtest
    print("\nRunning backtest...")
    df, metrics = run_backtest(manager.adapters["alphaforge"], config, catalog)

    print_results(df, metrics)

    # Example 2: Offline learning (label="y_final")
    # Uses final target values from evaluation_asof_date
    print("\n" + "=" * 80)
    print("WALK-FORWARD BACKTEST (Offline Learning)")
    print("=" * 80)

    config_offline = BacktestConfig(
        target_series_key="US_GDP_SAAR",
        predictor_series_keys=["US_CPI", "US_UNRATE", "DAILY_FCI"],
        agg_spec={"US_CPI": "mean", "US_UNRATE": "mean", "DAILY_FCI": "mean"},
        start_date=date(2025, 1, 1),
        end_date=date(2025, 6, 30),
        freq="W",
        label="y_final",  # Offline learning mode
        evaluation_asof_date=date(2025, 12, 31),  # Required for offline mode
        final_target_policy=TargetPolicy(mode="latest_available", max_release_rank=3),
        train_min_periods=4,
        rolling_window=None,
        include_y_asof_latest_as_feature=True,  # Use real-time target as feature
        compute_metrics=True,
    )

    print(f"\nTarget: {config_offline.target_series_key}")
    print(f"Predictors: {config_offline.predictor_series_keys}")
    print(f"Period: {config_offline.start_date} to {config_offline.end_date}")
    print(f"Label: {config_offline.label}")
    print(f"Evaluation date: {config_offline.evaluation_asof_date}")
    print(f"Include y_asof_latest as feature: {config_offline.include_y_asof_latest_as_feature}")

    print("\nRunning backtest...")
    df_offline, metrics_offline = run_backtest(
        manager.adapters["alphaforge"], config_offline, catalog
    )

    print_results(df_offline, metrics_offline)


def print_results(df, metrics) -> None:
    """Print backtest results."""
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"\nTotal vintages: {len(df)}")
    print(f"Valid predictions: {metrics['count']}")

    if metrics["count"] > 0:
        print(f"\nMetrics (overall):")
        print(f"  RMSE: {metrics['rmse']:.4f}")
        print(f"  MAE:  {metrics['mae']:.4f}")

        if "by_ref_quarter" in metrics:
            print(f"\nMetrics by reference quarter:")
            for quarter, qmetrics in sorted(metrics["by_ref_quarter"].items()):
                print(
                    f"  {quarter}: RMSE={qmetrics['rmse']:.4f}, "
                    f"MAE={qmetrics['mae']:.4f}, count={qmetrics['count']}"
                )

    print("\n" + "-" * 80)
    print("RESULTS HEAD (first 5 rows)")
    print("-" * 80)
    display_cols = [
        "asof_date",
        "ref_quarter",
        "y_pred",
        "y_true",
        "n_train",
        "n_features",
    ]
    display_cols = [col for col in display_cols if col in df.columns]
    if len(df) > 0:
        print(df[display_cols].head())
    else:
        print("(No results)")

    print("\n" + "-" * 80)
    print("N_TRAIN PROGRESSION (walk-forward check)")
    print("-" * 80)
    if "n_train" in df.columns and len(df) > 0:
        valid_df = df[df["n_train"] > 0]
        if len(valid_df) > 0:
            print(f"  First vintage: n_train = {valid_df['n_train'].iloc[0]}")
            print(f"  Last vintage:  n_train = {valid_df['n_train'].iloc[-1]}")
            print(f"  Max n_train:   {valid_df['n_train'].max()}")
            print(f"  (n_train should increase in expanding window mode)")


if __name__ == "__main__":
    main()

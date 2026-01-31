"""Example backtest runner demonstrating BridgeNowcaster with different label modes."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from nowcast_data.models.backtest import BacktestConfig, run_backtest
from nowcast_data.pit.api import PITDataManager
from nowcast_data.pit.core.catalog import SeriesCatalog


def main() -> None:
    """Run a backtest with both online and offline labels."""
    catalog_path = Path(__file__).parent.parent / "series_catalog.yaml"
    catalog = SeriesCatalog(catalog_path)
    manager = PITDataManager(catalog)

    # Define backtest period: last 10 business days ending 2024-12-01
    end_date = date(2024, 12, 1)
    start_date = end_date - timedelta(days=30)

    print("=" * 80)
    print("NOWCASTING MODEL BACKTEST")
    print("=" * 80)
    print(f"\nBacktest period: {start_date} to {end_date}")
    print("Vintage frequency: B (business days)")
    print(f"Target series: US_GDP_SAAR")
    print(f"Predictors: US_CPI, US_UNRATE")

    # Configuration shared across both label modes
    agg_spec = {"US_CPI": "mean", "US_UNRATE": "mean"}

    # ========================================================================
    # BACKTEST 1: Online label (y_asof_latest)
    # ========================================================================
    print("\n" + "=" * 80)
    print("BACKTEST 1: Online Label (y_asof_latest)")
    print("=" * 80)
    print("Training on latest available values as-of each vintage date.")

    online_config = BacktestConfig(
        target_series_key="US_GDP_SAAR",
        predictor_series_keys=["US_CPI", "US_UNRATE"],
        agg_spec=agg_spec,
        start_date=start_date,
        end_date=end_date,
        freq="B",
        label="y_asof_latest",
        model="ridge",
        train_min_periods=20,
        evaluation_asof_date=None,  # Not needed for online label
    )

    results_online, metrics_online = run_backtest(manager, online_config)

    print(f"\nResults shape: {results_online.shape}")
    print(f"Valid predictions: {metrics_online['count']}")
    print("\nSummary statistics:")
    print(results_online[["y_pred", "y_true_asof_latest", "n_train", "n_features"]].describe())

    print("\nFirst 5 predictions:")
    print(
        results_online[
            ["asof_date", "ref_quarter", "y_pred", "y_true_asof_latest", "label_used"]
        ].head()
    )

    # ========================================================================
    # BACKTEST 2: Offline label (y_final)
    # ========================================================================
    print("\n" + "=" * 80)
    print("BACKTEST 2: Offline Label (y_final)")
    print("=" * 80)
    print("Training on final target values (evaluated at a later date).")

    # For offline labels, evaluation_asof_date should be much later than vintages
    evaluation_date = end_date + timedelta(days=120)

    offline_config = BacktestConfig(
        target_series_key="US_GDP_SAAR",
        predictor_series_keys=["US_CPI", "US_UNRATE"],
        agg_spec=agg_spec,
        start_date=start_date,
        end_date=end_date,
        freq="B",
        label="y_final",
        model="ridge",
        train_min_periods=20,
        evaluation_asof_date=evaluation_date,
    )

    results_offline, metrics_offline = run_backtest(manager, offline_config)

    print(f"\nResults shape: {results_offline.shape}")
    print(f"Valid predictions: {metrics_offline['count']}")
    print(f"Evaluation asof date: {evaluation_date}")
    print("\nSummary statistics:")
    print(results_offline[["y_pred", "y_true_final", "n_train", "n_features"]].describe())

    print("\nFirst 5 predictions:")
    print(
        results_offline[["asof_date", "ref_quarter", "y_pred", "y_true_final", "label_used"]].head()
    )

    # ========================================================================
    # COMPARISON
    # ========================================================================
    print("\n" + "=" * 80)
    print("BACKTEST COMPARISON")
    print("=" * 80)

    # Merge results for comparison
    comparison = pd.merge(
        results_online[["asof_date", "ref_quarter", "y_pred", "y_true_asof_latest"]].rename(
            columns={"y_pred": "y_pred_online", "y_true_asof_latest": "y_online"}
        ),
        results_offline[["asof_date", "ref_quarter", "y_pred", "y_true_final"]].rename(
            columns={"y_pred": "y_pred_offline", "y_true_final": "y_offline"}
        ),
        on=["asof_date", "ref_quarter"],
        how="inner",
    )

    if not comparison.empty:
        comparison["pred_diff"] = comparison["y_pred_offline"] - comparison["y_pred_online"]
        comparison["true_diff"] = comparison["y_offline"] - comparison["y_online"]

        print("\nPrediction differences (offline - online):")
        print(f"  Mean: {comparison['pred_diff'].mean():.4f}")
        print(f"  Std:  {comparison['pred_diff'].std():.4f}")
        print(f"  Min:  {comparison['pred_diff'].min():.4f}")
        print(f"  Max:  {comparison['pred_diff'].max():.4f}")

        print("\nTarget value differences (offline - online):")
        print(f"  Mean: {comparison['true_diff'].mean():.4f}")
        print(f"  Std:  {comparison['true_diff'].std():.4f}")

        print("\nSample comparison (first 5 rows):")
        print(
            comparison[
                ["asof_date", "ref_quarter", "y_pred_online", "y_pred_offline", "pred_diff"]
            ].head()
        )

    print("\n" + "=" * 80)
    print("END OF BACKTEST")
    print("=" * 80)


if __name__ == "__main__":
    main()

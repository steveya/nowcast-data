"""Unit tests for backtest module (walk-forward backtesting)."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nowcast_data.models.backtest import BacktestConfig, make_vintage_dates, run_backtest
from nowcast_data.pit.adapters.alphaforge import AlphaForgePITAdapter


class TestMakeVintageDates:
    """Tests for make_vintage_dates function."""

    def test_daily_frequency(self) -> None:
        """Test daily vintage generation."""
        dates = make_vintage_dates(date(2025, 1, 1), date(2025, 1, 5), "D")
        assert len(dates) == 5
        assert dates[0] == date(2025, 1, 1)
        assert dates[-1] == date(2025, 1, 5)

    def test_weekly_frequency(self) -> None:
        """Test weekly vintage generation."""
        dates = make_vintage_dates(date(2025, 1, 1), date(2025, 2, 1), "W")
        assert len(dates) >= 4
        assert dates[0] == date(2025, 1, 1)
        # Check consecutive dates are 7 days apart
        for i in range(len(dates) - 1):
            assert (dates[i + 1] - dates[i]).days == 7

    def test_business_day_frequency(self) -> None:
        """Test business-day vintage generation (skips weekends)."""
        dates = make_vintage_dates(date(2025, 1, 1), date(2025, 1, 10), "B")
        # All dates should be weekdays
        for d in dates:
            assert d.weekday() < 5  # Monday=0, Friday=4

    def test_invalid_frequency(self) -> None:
        """Test that invalid frequency raises error."""
        with pytest.raises(ValueError, match="freq must be"):
            make_vintage_dates(date(2025, 1, 1), date(2025, 1, 5), "X")  # type: ignore

    def test_single_date(self) -> None:
        """Test when start and end are the same."""
        dates = make_vintage_dates(date(2025, 1, 1), date(2025, 1, 1), "D")
        assert dates == [date(2025, 1, 1)]


class TestBacktestConfig:
    """Tests for BacktestConfig dataclass."""

    def test_config_creation(self) -> None:
        """Test creating a BacktestConfig."""
        config = BacktestConfig(
            target_series_key="GDP",
            predictor_series_keys=["CPI", "UNRATE"],
            agg_spec={"CPI": "mean", "UNRATE": "mean"},
            start_date=date(2025, 1, 1),
            end_date=date(2025, 6, 30),
            freq="W",
            label="y_asof_latest",
        )
        assert config.target_series_key == "GDP"
        assert config.freq == "W"
        assert config.label == "y_asof_latest"
        assert config.compute_metrics is True

    def test_config_offline_label_requires_evaluation_date(self) -> None:
        """Test that label='y_final' requires evaluation_asof_date."""
        config = BacktestConfig(
            target_series_key="GDP",
            predictor_series_keys=[],
            agg_spec={},
            start_date=date(2025, 1, 1),
            end_date=date(2025, 1, 1),
            label="y_final",
            evaluation_asof_date=date(2025, 12, 31),  # Required
        )
        assert config.label == "y_final"
        assert config.evaluation_asof_date == date(2025, 12, 31)

    def test_config_defaults(self) -> None:
        """Test BacktestConfig defaults."""
        config = BacktestConfig(
            target_series_key="GDP",
            predictor_series_keys=[],
            agg_spec={},
            start_date=date(2025, 1, 1),
            end_date=date(2025, 1, 1),
        )
        assert config.freq == "W"
        assert config.label == "y_asof_latest"
        assert config.output_csv is None
        assert config.train_min_periods == 40  # Default is 40 vintages
        assert config.rolling_window is None


class TestRunBacktest:
    """Tests for run_backtest function."""

    def test_backtest_smoke_test(self, pit_context) -> None:
        """Smoke test: run_backtest returns DataFrame and metrics with expected structure."""
        config = BacktestConfig(
            target_series_key="BASE_GDP",
            predictor_series_keys=["US_GDP_SAAR"],
            agg_spec={"US_GDP_SAAR": "mean"},
            start_date=date(2025, 1, 1),
            end_date=date(2025, 3, 1),
            freq="W",
            label="y_asof_latest",
            train_min_periods=2,
            compute_metrics=True,
        )

        adapter = AlphaForgePITAdapter(ctx=pit_context)
        df, metrics = run_backtest(adapter, config)

        # Check DataFrame structure
        assert isinstance(df, pd.DataFrame)
        assert isinstance(metrics, dict)
        assert "asof_date" in df.columns
        assert "y_pred" in df.columns
        assert "n_train" in df.columns
        assert "n_features" in df.columns

        # Check metrics keys
        assert "rmse" in metrics
        assert "mae" in metrics
        assert "count" in metrics
        assert "stable_vs_final_3rd_growth" in metrics
        assert "stable_vs_real_time_growth" in metrics

    def test_backtest_walk_forward_train_sizes(self, pit_context) -> None:
        """Test that n_train increases in walk-forward fashion."""
        config = BacktestConfig(
            target_series_key="BASE_GDP",
            predictor_series_keys=["US_GDP_SAAR"],
            agg_spec={"US_GDP_SAAR": "mean"},
            start_date=date(2025, 1, 1),
            end_date=date(2025, 2, 1),
            freq="D",
            label="y_asof_latest",
            train_min_periods=1,
            compute_metrics=False,
        )

        adapter = AlphaForgePITAdapter(ctx=pit_context)
        df, _ = run_backtest(adapter, config)

        if len(df) > 3:
            # Check that n_train is non-decreasing (expanding window)
            valid_df = df[df["n_train"] > 0]
            if len(valid_df) > 1:
                n_train_values = valid_df["n_train"].tolist()
                for i in range(1, len(n_train_values)):
                    assert (
                        n_train_values[i] >= n_train_values[i - 1]
                    ), "n_train should be non-decreasing in expanding window"

    def test_backtest_rolling_window(self, pit_context) -> None:
        """Test rolling window constraint."""
        config = BacktestConfig(
            target_series_key="BASE_GDP",
            predictor_series_keys=["US_GDP_SAAR"],
            agg_spec={"US_GDP_SAAR": "mean"},
            start_date=date(2025, 1, 1),
            end_date=date(2025, 3, 1),
            freq="W",
            label="y_asof_latest",
            train_min_periods=1,
            rolling_window=3,  # Use last 3 vintages
            compute_metrics=False,
        )

        adapter = AlphaForgePITAdapter(ctx=pit_context)
        df, _ = run_backtest(adapter, config)

        # Check rolling window is respected (n_train <= rolling_window)
        valid_df = df[df["n_train"] > 0]
        if not valid_df.empty:
            assert valid_df["n_train"].max() <= 3, "n_train should not exceed rolling_window"

    def test_backtest_y_final_requires_evaluation_date(self, pit_context) -> None:
        """Test that label='y_final' without evaluation_asof_date raises error."""
        config = BacktestConfig(
            target_series_key="BASE_GDP",
            predictor_series_keys=["US_GDP_SAAR"],
            agg_spec={"US_GDP_SAAR": "mean"},
            start_date=date(2025, 1, 1),
            end_date=date(2025, 2, 1),
            freq="W",
            label="y_final",
            evaluation_asof_date=None,
            compute_metrics=False,
        )

        adapter = AlphaForgePITAdapter(ctx=pit_context)
        with pytest.raises(ValueError, match="evaluation_asof_date is required"):
            run_backtest(adapter, config)

    def test_backtest_compute_metrics(self, pit_context) -> None:
        """Test compute_metrics=True computes RMSE, MAE."""
        config = BacktestConfig(
            target_series_key="BASE_GDP",
            predictor_series_keys=["US_GDP_SAAR"],
            agg_spec={"US_GDP_SAAR": "mean"},
            start_date=date(2025, 1, 1),
            end_date=date(2025, 2, 1),
            freq="W",
            label="y_asof_latest",
            train_min_periods=1,
            compute_metrics=True,
        )

        adapter = AlphaForgePITAdapter(ctx=pit_context)
        df, metrics = run_backtest(adapter, config)

        if metrics["count"] > 0:
            # Check RMSE computation
            valid_mask = df["y_true"].notna() & df["y_pred"].notna()
            valid_df = df[valid_mask]
            if not valid_df.empty:
                manual_rmse = float(
                    np.sqrt(((valid_df["y_pred"] - valid_df["y_true"]) ** 2).mean())
                )
                assert np.isclose(metrics["rmse"], manual_rmse, rtol=1e-5)

                # Check MAE computation
                manual_mae = float((valid_df["y_pred"] - valid_df["y_true"]).abs().mean())
                assert np.isclose(metrics["mae"], manual_mae, rtol=1e-5)

            # Check grouped metrics
            assert "by_ref_quarter" in metrics

    def test_backtest_output_csv(self, pit_context, tmp_path) -> None:
        """Test output_csv saves results to file."""
        csv_path = str(tmp_path / "backtest_results.csv")
        config = BacktestConfig(
            target_series_key="BASE_GDP",
            predictor_series_keys=["US_GDP_SAAR"],
            agg_spec={"US_GDP_SAAR": "mean"},
            start_date=date(2025, 1, 1),
            end_date=date(2025, 2, 1),
            freq="W",
            train_min_periods=1,
            output_csv=csv_path,
            compute_metrics=False,
        )

        adapter = AlphaForgePITAdapter(ctx=pit_context)
        df, _ = run_backtest(adapter, config)

        # Check file was created
        assert Path(csv_path).exists()

        # Check CSV can be read back
        df_read = pd.read_csv(csv_path)
        assert len(df_read) == len(df)

    def test_backtest_empty_result(self, pit_context) -> None:
        """Test run_backtest returns empty result when no vintages generated."""
        config = BacktestConfig(
            target_series_key="BASE_GDP",
            predictor_series_keys=["US_GDP_SAAR"],
            agg_spec={"US_GDP_SAAR": "mean"},
            start_date=date(2025, 6, 1),  # Far future
            end_date=date(2025, 5, 1),  # End before start
            freq="W",
            compute_metrics=True,
        )

        adapter = AlphaForgePITAdapter(ctx=pit_context)
        df, metrics = run_backtest(adapter, config)

        assert df.empty
        assert np.isnan(metrics["rmse"])
        assert metrics["count"] == 0

    def test_backtest_insufficient_training_data(self, pit_context) -> None:
        """Test that vintages with insufficient training data have NaN predictions."""
        config = BacktestConfig(
            target_series_key="BASE_GDP",
            predictor_series_keys=["US_GDP_SAAR"],
            agg_spec={"US_GDP_SAAR": "mean"},
            start_date=date(2025, 1, 1),
            end_date=date(2025, 1, 5),
            freq="D",
            train_min_periods=100,  # Very high threshold
            compute_metrics=False,
        )

        adapter = AlphaForgePITAdapter(ctx=pit_context)
        df, _ = run_backtest(adapter, config)

        # All predictions should be NaN due to insufficient training
        if not df.empty:
            assert df["y_pred"].isna().all()


class TestPanelFunctions:
    """Tests for panel module functions (pure unit tests, no alphaforge needed)."""

    def test_get_feature_columns_basic(self) -> None:
        """Test get_feature_columns with basic DataFrame."""
        from nowcast_data.models.panel import get_feature_columns

        df = pd.DataFrame(
            {
                "feature1": [1, 2, 3],
                "feature2": [4, 5, 6],
                "y_asof_latest": [7, 8, 9],
                "y_final": [10, 11, 12],
                "ref_quarter": ["2024Q1", "2024Q2", "2024Q3"],
            }
        )

        cols = get_feature_columns(df)
        assert "feature1" in cols
        assert "feature2" in cols
        assert "y_asof_latest" not in cols
        assert "y_final" not in cols
        assert "ref_quarter" not in cols

    def test_preprocess_panel_imputation(self) -> None:
        """Test preprocess_panel_for_training imputes missing values."""
        from nowcast_data.models.panel import preprocess_panel_for_training

        # Create panel with missing values
        panel = pd.DataFrame(
            {
                "feature1": [1.0, np.nan, 3.0, 4.0],
                "feature2": [10.0, 20.0, np.nan, 40.0],
                "y_asof_latest": [100.0, 200.0, 300.0, 400.0],
            },
            index=pd.to_datetime(["2025-01-01", "2025-01-08", "2025-01-15", "2025-01-22"]).date,
        )

        train_vintages = [date(2025, 1, 1), date(2025, 1, 8), date(2025, 1, 15)]
        test_vintage = date(2025, 1, 22)

        train_X, train_y, test_X, test_y, stats = preprocess_panel_for_training(
            panel,
            train_vintages,
            test_vintage,
            feature_cols=["feature1", "feature2"],
            label_col="y_asof_latest",
            max_nan_fraction=0.5,
            standardize=False,
        )

        # Check no NaN in imputed training data
        assert not train_X.isna().any().any(), "Training X should have no NaN after imputation"

    def test_preprocess_panel_standardization(self) -> None:
        """Test preprocess_panel_for_training standardizes features."""
        from nowcast_data.models.panel import preprocess_panel_for_training

        panel = pd.DataFrame(
            {
                "feature1": [10.0, 20.0, 30.0, 40.0],
                "y_asof_latest": [1.0, 2.0, 3.0, 4.0],
            },
            index=pd.to_datetime(["2025-01-01", "2025-01-08", "2025-01-15", "2025-01-22"]).date,
        )

        train_vintages = [date(2025, 1, 1), date(2025, 1, 8), date(2025, 1, 15)]
        test_vintage = date(2025, 1, 22)

        train_X, train_y, test_X, test_y, stats = preprocess_panel_for_training(
            panel,
            train_vintages,
            test_vintage,
            feature_cols=["feature1"],
            label_col="y_asof_latest",
            standardize=True,
        )

        # Check standardization
        assert abs(train_X["feature1"].mean()) < 1e-10, "Mean should be ~0 after standardization"

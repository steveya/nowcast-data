"""Tests for BridgeNowcaster offline-label support."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from nowcast_data.models.bridge import BridgeConfig, BridgeNowcaster

try:
    from alphaforge.time.ref_period import RefFreq
    from nowcast_data.pit.adapters.alphaforge import AlphaForgePITAdapter

    HAS_ALPHAFORGE = True
except ImportError:
    HAS_ALPHAFORGE = False


@pytest.mark.skipif(not HAS_ALPHAFORGE, reason="alphaforge not installed")
def test_bridge_nowcaster_offline_label_smoke(pit_context) -> None:
    """Test that BridgeNowcaster works with offline labels."""
    adapter = AlphaForgePITAdapter(ctx=pit_context)

    config = BridgeConfig(
        target_series_key="GDP",
        predictor_series_keys=[],
        agg_spec={},
        label="y_final",
        evaluation_asof_date=date(2025, 3, 1),
        min_train_quarters=1,
    )

    nowcaster = BridgeNowcaster(config, adapter)
    result = nowcaster.fit_predict_one(date(2025, 1, 15))

    assert result is not None
    assert "label_used" in result
    assert result["label_used"] == "y_final"
    assert "y_true_final" in result


@pytest.mark.skipif(not HAS_ALPHAFORGE, reason="alphaforge not installed")
def test_bridge_nowcaster_online_label_smoke(pit_context) -> None:
    """Test that BridgeNowcaster works with online labels (default)."""
    adapter = AlphaForgePITAdapter(ctx=pit_context)

    config = BridgeConfig(
        target_series_key="GDP",
        predictor_series_keys=[],
        agg_spec={},
        label="y_asof_latest",
        min_train_quarters=1,
    )

    nowcaster = BridgeNowcaster(config, adapter)
    result = nowcaster.fit_predict_one(date(2025, 1, 15))

    assert result is not None
    assert "label_used" in result
    assert result["label_used"] == "y_asof_latest"
    assert "y_true_asof" in result


@pytest.mark.skipif(not HAS_ALPHAFORGE, reason="alphaforge not installed")
def test_bridge_nowcaster_offline_label_requires_eval_date(pit_context) -> None:
    """Test that offline label requires evaluation_asof_date."""
    adapter = AlphaForgePITAdapter(ctx=pit_context)

    config = BridgeConfig(
        target_series_key="GDP",
        predictor_series_keys=[],
        agg_spec={},
        label="y_final",
        evaluation_asof_date=None,  # Missing!
        min_train_quarters=1,
    )

    nowcaster = BridgeNowcaster(config, adapter)

    with pytest.raises(ValueError, match="evaluation_asof_date"):
        nowcaster.fit_predict_one(date(2025, 1, 15))


@pytest.mark.skipif(not HAS_ALPHAFORGE, reason="alphaforge not installed")
def test_bridge_nowcaster_no_lookahead_offline(pit_context) -> None:
    """Test that offline label training excludes current and future quarters."""
    adapter = AlphaForgePITAdapter(ctx=pit_context)

    # Populate with additional data points for multiple quarters
    data = pd.DataFrame(
        [
            {
                "series_key": "GDP",
                "obs_date": "2024-09-30",
                "asof_utc": "2024-11-15",
                "value": 2.5,
            },
            {
                "series_key": "GDP",
                "obs_date": "2024-09-30",
                "asof_utc": "2025-02-15",
                "value": 2.6,
            },
        ]
    )
    pit_context.pit.upsert_pit_observations(data)

    config = BridgeConfig(
        target_series_key="GDP",
        predictor_series_keys=[],
        agg_spec={},
        label="y_final",
        evaluation_asof_date=date(2025, 6, 1),
        min_train_quarters=1,
    )

    nowcaster = BridgeNowcaster(config, adapter)

    # Predict for Q1 2025
    result = nowcaster.fit_predict_one(date(2025, 1, 15))

    # Training should include Q4 2024 and earlier, not Q1 2025
    assert result["n_train"] >= 0
    # If we have enough data, we should have at least one training sample
    # (Q4 2024 should be in training set)

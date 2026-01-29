"""Pure unit tests that don't require alphaforge."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from nowcast_data.models.target_policy import quarter_end_date


class TestQuarterEndDate:
    """Tests for quarter_end_date function - pure pandas logic, no alphaforge needed."""

    def test_q1_end_date(self) -> None:
        """Q1 ends on March 31."""
        result = quarter_end_date("2025Q1")
        assert result == date(2025, 3, 31)

    def test_q2_end_date(self) -> None:
        """Q2 ends on June 30."""
        result = quarter_end_date("2025Q2")
        assert result == date(2025, 6, 30)

    def test_q3_end_date(self) -> None:
        """Q3 ends on September 30."""
        result = quarter_end_date("2025Q3")
        assert result == date(2025, 9, 30)

    def test_q4_end_date(self) -> None:
        """Q4 ends on December 31."""
        result = quarter_end_date("2025Q4")
        assert result == date(2025, 12, 31)

    def test_period_input(self) -> None:
        """Works with pandas Period input."""
        period = pd.Period("2025Q2", freq="Q")
        result = quarter_end_date(period)
        assert result == date(2025, 6, 30)

    def test_different_year(self) -> None:
        """Works with different year."""
        result = quarter_end_date("2024Q1")
        assert result == date(2024, 3, 31)

    def test_invalid_format_raises(self) -> None:
        """Invalid format raises ValueError."""
        with pytest.raises(ValueError, match="Expected quarterly"):
            quarter_end_date("2025M01")

    def test_invalid_quarter_raises(self) -> None:
        """Invalid quarter number raises ValueError."""
        with pytest.raises(ValueError, match="Invalid quarter"):
            quarter_end_date("2025Q5")


class TestDataFrameSelectionLogic:
    """Tests for pure DataFrame selection logic without PIT adapter."""

    def test_train_test_split_by_quarter(self) -> None:
        """Test that we correctly split training and test by quarter."""
        # Create a simple DataFrame with quarterly data
        df = pd.DataFrame(
            {
                "ref_quarter": pd.PeriodIndex(
                    ["2025Q1", "2025Q2", "2025Q3"],
                    freq="Q",
                ),
                "value": [1.0, 2.0, 3.0],
            }
        )
        df = df.set_index("ref_quarter")

        current_quarter = pd.Period("2025Q3", freq="Q")
        train_mask = df.index < current_quarter
        test_mask = df.index == current_quarter

        assert train_mask.sum() == 2  # Q1 and Q2
        assert test_mask.sum() == 1  # Q3
        assert df.loc[train_mask].shape[0] == 2
        assert df.loc[test_mask].shape[0] == 1

    def test_feature_nan_filtering(self) -> None:
        """Test filtering features by NaN fraction."""
        df = pd.DataFrame(
            {
                "feat1": [1.0, 2.0, 3.0, 4.0, 5.0],
                "feat2": [1.0, 2.0, None, None, None],  # 60% NaN
                "feat3": [None, None, None, None, None],  # 100% NaN
            }
        )

        max_nan_fraction = 0.5
        nan_frac = df.isna().mean()
        kept_features = df.loc[:, nan_frac <= max_nan_fraction]

        assert "feat1" in kept_features.columns
        assert "feat2" not in kept_features.columns
        assert "feat3" not in kept_features.columns

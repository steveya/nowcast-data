"""Tests for vintage selection logic."""

from datetime import date
import pytest

from nowcast_data.pit.core.vintage_logic import (
    select_vintage_for_asof,
    validate_no_lookahead,
)


class TestSelectVintageForAsof:
    """Tests for select_vintage_for_asof function."""
    
    def test_empty_vintages(self):
        """Should return None for empty vintage list."""
        result = select_vintage_for_asof([], date(2020, 1, 1))
        assert result is None
    
    def test_asof_before_first_vintage(self):
        """Should return None when asof is before first vintage."""
        vintages = [date(2020, 1, 1), date(2020, 2, 1), date(2020, 3, 1)]
        result = select_vintage_for_asof(vintages, date(2019, 12, 1))
        assert result is None
    
    def test_asof_exact_vintage(self):
        """Should return exact vintage when asof matches."""
        vintages = [date(2020, 1, 1), date(2020, 2, 1), date(2020, 3, 1)]
        result = select_vintage_for_asof(vintages, date(2020, 2, 1))
        assert result == date(2020, 2, 1)
    
    def test_asof_between_vintages(self):
        """Should return previous vintage when asof is between vintages."""
        vintages = [date(2020, 1, 1), date(2020, 2, 1), date(2020, 3, 1)]
        result = select_vintage_for_asof(vintages, date(2020, 1, 15))
        assert result == date(2020, 1, 1)
    
    def test_asof_after_last_vintage(self):
        """Should return last vintage when asof is after all vintages."""
        vintages = [date(2020, 1, 1), date(2020, 2, 1), date(2020, 3, 1)]
        result = select_vintage_for_asof(vintages, date(2020, 4, 1))
        assert result == date(2020, 3, 1)
    
    def test_unsorted_vintages(self):
        """Should work correctly even with unsorted vintages."""
        vintages = [date(2020, 3, 1), date(2020, 1, 1), date(2020, 2, 1)]
        result = select_vintage_for_asof(vintages, date(2020, 1, 15))
        assert result == date(2020, 1, 1)
    
    def test_single_vintage_before(self):
        """Should work with single vintage - asof before."""
        vintages = [date(2020, 2, 1)]
        result = select_vintage_for_asof(vintages, date(2020, 1, 1))
        assert result is None
    
    def test_single_vintage_after(self):
        """Should work with single vintage - asof after."""
        vintages = [date(2020, 2, 1)]
        result = select_vintage_for_asof(vintages, date(2020, 3, 1))
        assert result == date(2020, 2, 1)


class TestValidateNoLookahead:
    """Tests for validate_no_lookahead function."""
    
    def test_vintage_before_asof(self):
        """Should be valid when vintage is before asof."""
        result = validate_no_lookahead(
            vintage_date=date(2020, 1, 1),
            asof_date=date(2020, 2, 1)
        )
        assert result is True
    
    def test_vintage_equals_asof(self):
        """Should be valid when vintage equals asof."""
        result = validate_no_lookahead(
            vintage_date=date(2020, 1, 1),
            asof_date=date(2020, 1, 1)
        )
        assert result is True
    
    def test_vintage_after_asof(self):
        """Should be invalid when vintage is after asof (lookahead)."""
        result = validate_no_lookahead(
            vintage_date=date(2020, 2, 1),
            asof_date=date(2020, 1, 1)
        )
        assert result is False

"""Core vintage selection and PIT logic."""

from datetime import date, datetime

import pandas as pd


def select_vintage_for_asof(
    vintages: list[date | datetime | pd.Timestamp], asof_date: date | datetime | pd.Timestamp
) -> date | datetime | pd.Timestamp | None:
    """
    Select the appropriate vintage for a given asof_date.
    
    Rules:
    - If asof_date is before the first vintage, return None
    - If asof_date equals a vintage, return that vintage
    - If asof_date is between vintages, return the most recent vintage before asof_date
    - Returns the latest vintage not after asof_date
    
    Args:
        vintages: List of vintage dates (must be sorted or will be sorted)
        asof_date: The as-of evaluation date
        
    Returns:
        The selected vintage date, or None if no vintage is available
        
    Examples:
        >>> vintages = [date(2020, 1, 1), date(2020, 2, 1), date(2020, 3, 1)]
        >>> select_vintage_for_asof(vintages, date(2019, 12, 1))  # Before first
        None
        >>> select_vintage_for_asof(vintages, date(2020, 1, 1))   # Exact match
        datetime.date(2020, 1, 1)
        >>> select_vintage_for_asof(vintages, date(2020, 1, 15))  # Between vintages
        datetime.date(2020, 1, 1)
        >>> select_vintage_for_asof(vintages, date(2020, 4, 1))   # After last
        datetime.date(2020, 3, 1)
    """
    if not vintages:
        return None
    
    # Normalize all dates to pd.Timestamp for consistent comparison
    norm_vintages = [_normalize_date(v) for v in vintages]
    norm_asof = _normalize_date(asof_date)
    
    # Sort vintages to ensure correct ordering
    sorted_vintages = sorted(zip(norm_vintages, vintages))
    
    # Find the latest vintage not after asof_date
    selected = None
    selected_original = None
    
    for norm_v, orig_v in sorted_vintages:
        if norm_v <= norm_asof:
            selected = norm_v
            selected_original = orig_v
        else:
            break
    
    return selected_original


def _normalize_date(d: date | datetime | pd.Timestamp) -> pd.Timestamp:
    """Normalize date to pd.Timestamp for comparison."""
    if isinstance(d, pd.Timestamp):
        # Remove timezone for date comparison if present
        if d.tzinfo is not None:
            return d.tz_localize(None)
        return d
    return pd.Timestamp(d)


def validate_no_lookahead(
    vintage_date: date | datetime | pd.Timestamp,
    asof_date: date | datetime | pd.Timestamp
) -> bool:
    """
    Validate that vintage_date does not create lookahead bias.
    
    Note: Timezone-aware timestamps are normalized to timezone-naive before
    comparison (timezone info is removed). This ensures consistent comparison
    across different timestamp types.
    
    Args:
        vintage_date: The vintage date used
        asof_date: The as-of evaluation date
        
    Returns:
        True if valid (vintage_date <= asof_date), False otherwise
    """
    norm_vintage = _normalize_date(vintage_date)
    norm_asof = _normalize_date(asof_date)
    return norm_vintage <= norm_asof

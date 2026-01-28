from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import re
from typing import Literal

import pandas as pd

from nowcast_data.pit.adapters.base import PITAdapter


@dataclass
class TargetPolicy:
    mode: Literal["latest_available", "nth_release", "next_release"]
    nth: int | None = None
    max_release_rank: int | None = None


def quarter_end_date(ref_quarter: str | pd.Period) -> date:
    """Return the quarter-end date for a reference quarter.

    Args:
        ref_quarter: Quarterly reference in ``YYYYQn`` format or a pandas Period
            (e.g., ``"2025Q1"``, ``pd.Period("2025Q1", freq="Q")``).

    Returns:
        The calendar quarter-end date.

    Raises:
        ValueError: If the reference quarter does not match ``YYYYQn``.
    """
    ref_str = str(ref_quarter)
    match = re.match(r"^(\d{4})Q([1-4])$", ref_str)
    if not match:
        raise ValueError(f"Expected ref quarter in format YYYYQn, got {ref_str}")
    year = int(match.group(1))
    quarter = int(match.group(2))
    if quarter == 1:
        return date(year, 3, 31)
    if quarter == 2:
        return date(year, 6, 30)
    if quarter == 3:
        return date(year, 9, 30)
    return date(year, 12, 31)


def list_quarterly_target_releases_asof(
    adapter: PITAdapter,
    *,
    series_key: str,
    ref_quarter: str | pd.Period,
    asof_date: date,
) -> pd.DataFrame:
    """List available quarterly releases up to an as-of date.

    Args:
        adapter: PIT adapter that supports ``list_pit_observations_asof``.
        series_key: PIT series key for the target series.
        ref_quarter: Quarterly reference in ``YYYYQn`` format or a pandas Period.
        asof_date: Vintage cut-off date (treated as end-of-day UTC).

    Returns:
        DataFrame sorted by ``asof_utc`` (ascending) with columns:
        ``obs_date`` (quarter end), ``asof_utc`` (release timestamp), and ``value``.
    """
    obs_date = quarter_end_date(ref_quarter)
    releases = adapter.list_pit_observations_asof(
        series_key=series_key,
        obs_date=obs_date,
        asof_date=asof_date,
    )
    releases = releases.loc[:, ["obs_date", "asof_utc", "value"]].copy()
    return releases.sort_values("asof_utc", kind="mergesort").reset_index(drop=True)


def resolve_target_from_releases(
    releases: pd.DataFrame,
    policy: TargetPolicy,
) -> tuple[float | None, dict]:
    """Resolve a target value from available releases.

    Args:
        releases: DataFrame containing ``obs_date``, ``asof_utc``, and ``value`` columns.
        policy: Target policy configuration.

    Returns:
        Tuple of (value, metadata). ``value`` is ``None`` if the policy cannot select
        a release. Metadata includes:
        ``policy_mode``, ``selected_release_rank``, ``selected_release_asof_utc``,
        ``available_release_ranks``, ``n_releases_available``, and ``obs_date``.

    Raises:
        ValueError: If the policy configuration is invalid.
    """
    releases_sorted = releases.sort_values("asof_utc", kind="mergesort").reset_index(drop=True)
    if policy.max_release_rank is not None:
        if policy.max_release_rank < 1:
            raise ValueError("max_release_rank must be >= 1")
        releases_sorted = releases_sorted.head(policy.max_release_rank).reset_index(drop=True)

    k = len(releases_sorted)
    available_release_ranks = list(range(1, k + 1))
    obs_date = releases_sorted["obs_date"].iloc[0] if k > 0 else None

    value: float | None = None
    selected_rank: int | None = None
    selected_asof_utc = None

    if policy.mode == "latest_available":
        if k > 0:
            selected_rank = k
            row = releases_sorted.iloc[selected_rank - 1]
            row_value = row["value"]
            value = float(row_value) if pd.notna(row_value) else None
            if value is not None:
                selected_asof_utc = row["asof_utc"]
    elif policy.mode == "nth_release":
        if policy.nth is None:
            raise ValueError("TargetPolicy.nth must be set for nth_release mode")
        if policy.nth < 1:
            raise ValueError("TargetPolicy.nth must be >= 1")
        selected_rank = policy.nth
        if policy.nth <= k:
            row = releases_sorted.iloc[selected_rank - 1]
            row_value = row["value"]
            value = float(row_value) if pd.notna(row_value) else None
            if value is not None:
                selected_asof_utc = row["asof_utc"]
    elif policy.mode == "next_release":
        selected_rank = k + 1
        if policy.max_release_rank is not None:
            selected_rank = min(selected_rank, policy.max_release_rank)
    else:
        raise ValueError(f"Unknown target policy mode: {policy.mode}")

    meta = {
        "policy_mode": policy.mode,
        "selected_release_rank": selected_rank,
        "selected_release_asof_utc": selected_asof_utc if value is not None else None,
        "available_release_ranks": available_release_ranks,
        "n_releases_available": k,
        "obs_date": obs_date,
    }
    return value, meta

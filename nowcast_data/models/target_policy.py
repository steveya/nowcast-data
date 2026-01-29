from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Literal

import pandas as pd

from nowcast_data.time.nowcast_calendar import refperiod_to_quarter_end
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
    return refperiod_to_quarter_end(ref_quarter)


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
        ``target_release_rank``, ``available_release_ranks``, ``n_releases_available``,
        and ``obs_date``. ``selected_*`` fields only describe an available release,
        while ``target_release_rank`` is used to signal a future (next) release.

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
    target_release_rank: int | None = None

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
        target_release_rank = k + 1
        if policy.max_release_rank is not None:
            target_release_rank = min(target_release_rank, policy.max_release_rank)
    else:
        raise ValueError(f"Unknown target policy mode: {policy.mode}")

    meta = {
        "policy_mode": policy.mode,
        "selected_release_rank": selected_rank,
        "selected_release_asof_utc": selected_asof_utc if value is not None else None,
        "target_release_rank": target_release_rank,
        "available_release_ranks": available_release_ranks,
        "n_releases_available": k,
        "obs_date": obs_date,
    }
    return value, meta


def resolve_quarterly_final_target(
    adapter: PITAdapter,
    *,
    series_key: str,
    ref_quarter: str | pd.Period,
    evaluation_asof_date: date,
    policy: TargetPolicy = TargetPolicy(mode="latest_available", max_release_rank=3),
) -> tuple[float | None, dict]:
    """Resolve a final target value for a quarterly reference.

    Args:
        adapter: PIT adapter that supports ``list_pit_observations_asof``.
        series_key: PIT series key for the target series.
        ref_quarter: Quarterly reference in ``YYYYQn`` format or a pandas Period.
        evaluation_asof_date: As-of date used to resolve the final target proxy.
        policy: Target selection policy (defaults to latest_available capped to 3 releases).

    Returns:
        Tuple of (value, metadata) from ``resolve_target_from_releases``.
    """
    releases = list_quarterly_target_releases_asof(
        adapter,
        series_key=series_key,
        ref_quarter=ref_quarter,
        asof_date=evaluation_asof_date,
    )
    return resolve_target_from_releases(releases, policy)


def get_quarterly_release_observation_stream(
    adapter: PITAdapter,
    *,
    series_key: str,
    ref_quarter: str | pd.Period,
    asof_date: date,
    max_release_rank: int | None = None,
) -> pd.DataFrame:
    """Return a time-ordered observation stream of quarterly releases."""
    releases = list_quarterly_target_releases_asof(
        adapter,
        series_key=series_key,
        ref_quarter=ref_quarter,
        asof_date=asof_date,
    )
    if max_release_rank is not None:
        if max_release_rank < 1:
            raise ValueError("max_release_rank must be >= 1")
        releases = releases.head(max_release_rank).reset_index(drop=True)

    if releases.empty:
        return pd.DataFrame(
            {
                "series_key": pd.Series(dtype="object"),
                "ref_quarter": pd.Series(dtype="object"),
                "obs_date": pd.Series(dtype="datetime64[ns, UTC]"),
                "release_rank": pd.Series(dtype="int64"),
                "asof_utc": pd.Series(dtype="datetime64[ns, UTC]"),
                "value": pd.Series(dtype="float64"),
            }
        )

    release_rank = pd.Series(range(1, len(releases) + 1), name="release_rank")
    stream = releases.copy()
    stream.insert(0, "series_key", series_key)
    stream.insert(1, "ref_quarter", str(ref_quarter))
    stream.insert(3, "release_rank", release_rank)
    return stream.loc[
        :, ["series_key", "ref_quarter", "obs_date", "release_rank", "asof_utc", "value"]
    ]

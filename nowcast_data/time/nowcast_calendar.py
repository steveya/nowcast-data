"""Nowcast calendar utilities for ref-period semantics."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import re
from typing import Optional

try:  # pragma: no cover - optional dependency
    from alphaforge.time.ref_period import RefPeriod, RefFreq
except ImportError:  # pragma: no cover - fallback for type hints
    RefPeriod = None  # type: ignore[assignment]
    RefFreq = None  # type: ignore[assignment]

from nowcast_data.pit.adapters.base import PITAdapter


@dataclass(frozen=True)
class _FallbackRefPeriod:
    year: int
    quarter: int

    def __str__(self) -> str:
        return f"{self.year}Q{self.quarter}"


def _make_ref_period(year: int, quarter: int) -> "RefPeriod | _FallbackRefPeriod":
    ref_str = f"{year}Q{quarter}"
    if RefPeriod is None:
        return _FallbackRefPeriod(year=year, quarter=quarter)
    ref = RefPeriod.parse(ref_str)
    if str(ref) != ref_str:
        return _FallbackRefPeriod(year=year, quarter=quarter)
    return ref


def _refperiod_to_key(ref: "RefPeriod | _FallbackRefPeriod | str") -> str:
    if hasattr(ref, "to_key"):
        return ref.to_key()  # type: ignore[no-any-return]
    return str(ref)


def _refperiod_to_obs_date(ref: "RefPeriod | _FallbackRefPeriod | str") -> date:
    ref_key = _refperiod_to_key(ref)
    match = re.match(r"^(\d{4})Q(\d+)$", ref_key)
    if not match:
        raise ValueError(f"Expected quarterly RefPeriod in format YYYYQN, got {ref_key}")
    year = int(match.group(1))
    quarter = int(match.group(2))
    if quarter not in {1, 2, 3, 4}:
        raise ValueError(f"Invalid quarter: {quarter}. Must be 1, 2, 3, or 4")
    if quarter == 1:
        return date(year, 3, 31)
    if quarter == 2:
        return date(year, 6, 30)
    if quarter == 3:
        return date(year, 9, 30)
    return date(year, 12, 31)


def infer_current_quarter(asof_date: date) -> "RefPeriod | _FallbackRefPeriod":
    """Infer the current reference quarter from an asof date."""
    quarter = ((asof_date.month - 1) // 3) + 1
    return _make_ref_period(asof_date.year, quarter)


def infer_previous_quarter(asof_date: date) -> "RefPeriod | _FallbackRefPeriod":
    """Infer the previous reference quarter from an asof date."""
    quarter = ((asof_date.month - 1) // 3) + 1
    year = asof_date.year
    if quarter == 1:
        quarter = 4
        year -= 1
    else:
        quarter -= 1
    return _make_ref_period(year, quarter)


def refperiod_to_quarter_end(ref: "RefPeriod | _FallbackRefPeriod | str") -> date:
    """Convert a ref period to its quarter-end date."""
    return _refperiod_to_obs_date(ref)


def get_target_asof_ref(
    adapter: PITAdapter,
    series_id_or_key: str,
    asof_date: date,
    ref: "RefPeriod | _FallbackRefPeriod | str",
    freq: Optional["RefFreq"] = None,
    *,
    metadata=None,
) -> float | None:
    """
    Retrieve the target value for a ref period as-of a vintage.

    Args:
        adapter: PIT adapter instance used for ref-period snapshots.
        series_id_or_key: Series identifier or canonical series key.
        asof_date: Point-in-time evaluation date.
        ref: Reference period to fetch (quarterly).
        freq: Optional RefFreq override (defaults to quarterly when available).
        metadata: Optional series metadata passed through to the adapter.

    Returns:
        The target value for the ref period at the given vintage, or None if missing.

    Raises:
        NotImplementedError: If the adapter does not support ref-period snapshots.
        ValueError: If multiple observations are returned for a single ref period.
    """
    if freq is None and RefFreq is not None:
        freq = RefFreq.Q
    if type(adapter).fetch_asof_ref is PITAdapter.fetch_asof_ref:
        adapter_name = getattr(adapter, "name", adapter.__class__.__name__)
        raise NotImplementedError(
            f"Adapter '{adapter_name}' does not support ref-period snapshots"
        )
    ref_key = _refperiod_to_key(ref)
    observations = adapter.fetch_asof_ref(
        series_id_or_key,
        asof_date,
        start_ref=ref_key,
        end_ref=ref_key,
        freq=freq,
        metadata=metadata,
    )
    if observations and len(observations) > 1:
        raise ValueError(
            f"Expected single observation for ref period, got {len(observations)}"
        )
    if observations:
        return observations[0].value
    obs_date = _refperiod_to_obs_date(ref)
    series_key = metadata.series_key if metadata is not None else series_id_or_key
    fallback = adapter.list_pit_observations_asof(
        series_key=series_key,
        obs_date=obs_date,
        asof_date=asof_date,
    )
    if fallback.empty:
        return None
    return float(fallback.iloc[-1]["value"])

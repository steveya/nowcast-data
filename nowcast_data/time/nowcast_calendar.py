"""Nowcast calendar utilities for ref-period semantics."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
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
    if RefPeriod is None:
        return _FallbackRefPeriod(year=year, quarter=quarter)
    return RefPeriod.parse(f"{year}Q{quarter}")


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


def refperiod_to_quarter_end(ref: "RefPeriod | _FallbackRefPeriod") -> date:
    """Convert a ref period to its quarter-end date."""
    ref_str = str(ref)
    if "Q" not in ref_str:
        raise ValueError(f"Expected quarterly RefPeriod, got {ref_str}")
    year_str, quarter_str = ref_str.split("Q", 1)
    year = int(year_str)
    quarter = int(quarter_str)
    if quarter == 1:
        return date(year, 3, 31)
    if quarter == 2:
        return date(year, 6, 30)
    if quarter == 3:
        return date(year, 9, 30)
    if quarter == 4:
        return date(year, 12, 31)
    raise ValueError(f"Invalid quarter in RefPeriod: {ref_str}")


def get_target_asof_ref(
    adapter: PITAdapter,
    series_id_or_key: str,
    asof_date: date,
    ref: "RefPeriod | _FallbackRefPeriod",
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
    if adapter.fetch_asof_ref is PITAdapter.fetch_asof_ref:
        adapter_name = getattr(adapter, "name", adapter.__class__.__name__)
        raise NotImplementedError(
            f"Adapter '{adapter_name}' does not support ref-period snapshots"
        )
    observations = adapter.fetch_asof_ref(
        series_id_or_key,
        asof_date,
        start_ref=ref,
        end_ref=ref,
        freq=freq,
        metadata=metadata,
    )
    if not observations:
        return None
    if len(observations) > 1:
        raise ValueError(
            f"Expected single observation for ref period, got {len(observations)}"
        )
    return observations[0].value

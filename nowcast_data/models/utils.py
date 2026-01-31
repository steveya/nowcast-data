from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from nowcast_data.models.target_policy import quarter_end_date


def to_utc_naive(
    values: pd.Series | pd.DatetimeIndex | object,
) -> pd.Series | pd.DatetimeIndex | pd.Timestamp:
    """Convert datetime-like values to UTC-naive timestamps."""
    if isinstance(values, pd.Series) and pd.api.types.is_datetime64_any_dtype(values):
        parsed = values
    elif isinstance(values, pd.DatetimeIndex):
        parsed = values
    else:
        parsed = pd.to_datetime(values, utc=True, errors="coerce")
    if isinstance(parsed, pd.Series):
        if parsed.dt.tz is None:
            parsed = parsed.dt.tz_localize("UTC")
        else:
            parsed = parsed.dt.tz_convert("UTC")
        return parsed.dt.tz_localize(None)
    if isinstance(parsed, pd.DatetimeIndex):
        if parsed.tz is None:
            parsed = parsed.tz_localize("UTC")
        else:
            parsed = parsed.tz_convert("UTC")
        return parsed.tz_localize(None)
    if pd.isna(parsed):
        return parsed
    return parsed.tz_convert("UTC").tz_localize(None)


def to_quarter_period(ts: pd.Timestamp) -> pd.Period:
    """Converts a timestamp to a quarter period."""
    ts = to_utc_naive(ts)
    if pd.isna(ts):
        raise ValueError("Invalid or missing observation date")
    return pd.Timestamp(ts).to_period("Q")


def agg_series(series: pd.Series, method: str) -> float:
    """Aggregate a series by a given method."""
    if series.empty:
        return np.nan
    if method == "sum":
        return float(series.sum())
    if method == "last":
        return float(series.iloc[-1])
    if method == "mean":
        return float(series.mean())
    return float(series.mean())


def validate_monthly_obs_dates(
    obs_dates_naive: pd.Series | pd.DatetimeIndex,
    *,
    series_key: str,
    obs_date_anchor: str | None,
) -> None:
    """Validate monthly obs_date alignment based on anchor convention.

    Args:
        obs_dates_naive: UTC-naive observation dates.
        series_key: Series key used for error messages.
        obs_date_anchor: "start", "end", or None.

    Raises:
        ValueError: If any obs_date values are not aligned to the allowed anchors.
    """
    dates_index = pd.DatetimeIndex(obs_dates_naive)
    is_month_start = dates_index.is_month_start
    is_month_end = dates_index.is_month_end

    if obs_date_anchor == "start":
        allowed = is_month_start
        rule = "month-start"
    elif obs_date_anchor == "end":
        allowed = is_month_end
        rule = "month-end"
    else:
        allowed = is_month_start | is_month_end
        rule = "month-start or month-end"

    invalid = ~allowed
    if invalid.any():
        sample = pd.Index(dates_index[invalid]).strftime("%Y-%m-%d").unique().tolist()[:3]
        raise ValueError(
            f"Monthly predictor series '{series_key}' has obs_date values not aligned "
            f"to {rule}: {sample}"
        )


def apply_quarter_cutoff(
    series: pd.Series,
    *,
    asof_date: date,
    include_partial_quarters: bool,
    current_quarter: pd.Period | None = None,
) -> pd.Series:
    """Filter a series to avoid leakage across quarter boundaries.

    Rules:
    - For quarters < current_quarter: allow obs_date <= quarter_end_date(quarter).
    - For quarter == current_quarter:
        - if include_partial_quarters: allow obs_date <= asof_date
        - else: drop current-quarter observations
    - For quarters > current_quarter: drop observations
    """
    if series.empty:
        return series

    if current_quarter is None:
        current_quarter = pd.Period(pd.Timestamp(asof_date), freq="Q")

    quarters = series.index.map(to_quarter_period)
    keep_mask: list[bool] = []

    for obs_ts, quarter in zip(series.index, quarters):
        if quarter < current_quarter:
            cutoff = pd.Timestamp(quarter_end_date(str(quarter)))
            keep_mask.append(obs_ts <= cutoff)
            continue
        if quarter == current_quarter:
            if include_partial_quarters:
                keep_mask.append(obs_ts <= pd.Timestamp(asof_date))
            else:
                keep_mask.append(False)
            continue
        keep_mask.append(False)

    return series.loc[keep_mask]


def daily_feature_stats(series: pd.Series) -> dict[str, float]:
    """Compute minimal daily feature stats for a filtered daily series.

    Features:
    - last
    - mean_5d (last 5 observations)
    - mean_20d (last 20 observations)
    - std_20d (ddof=0; NaN if <2 observations)
    - n_obs
    """
    if series.empty:
        return {
            "last": np.nan,
            "mean_5d": np.nan,
            "mean_20d": np.nan,
            "std_20d": np.nan,
            "n_obs": 0,
        }

    clean = series.dropna()
    if clean.empty:
        return {
            "last": np.nan,
            "mean_5d": np.nan,
            "mean_20d": np.nan,
            "std_20d": np.nan,
            "n_obs": 0,
        }

    tail_5 = clean.iloc[-5:]
    tail_20 = clean.iloc[-20:]
    std_20 = float(tail_20.std(ddof=0)) if len(tail_20) >= 2 else np.nan

    return {
        "last": float(clean.iloc[-1]),
        "mean_5d": float(tail_5.mean()) if not tail_5.empty else np.nan,
        "mean_20d": float(tail_20.mean()) if not tail_20.empty else np.nan,
        "std_20d": std_20,
        "n_obs": int(clean.notna().sum()),
    }


def expand_daily_series_to_frame(
    series: pd.Series,
    *,
    series_key: str,
    quarter_index: pd.Index,
) -> pd.DataFrame:
    """Expand a daily series into quarterly feature columns.

    The series should already be filtered by apply_quarter_cutoff.
    """
    feature_rows: dict[str, dict[pd.Period, float]] = {}
    for quarter in quarter_index:
        quarter_series = series.loc[series.index.map(to_quarter_period) == quarter]
        stats = daily_feature_stats(quarter_series)
        for feat_name, value in stats.items():
            col_name = f"{series_key}.{feat_name}"
            feature_rows.setdefault(col_name, {})[quarter] = value

    frame = pd.DataFrame(feature_rows).reindex(quarter_index)
    return frame

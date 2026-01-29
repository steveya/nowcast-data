from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Iterable

import numpy as np
import pandas as pd

from nowcast_data.models.bridge import _agg_series, _to_quarter_period, _to_utc_naive
from nowcast_data.models.target_features import (
    QuarterlyTargetFeatureSpec,
    get_quarterly_target_release_features,
)
from nowcast_data.models.target_policy import (
    TargetPolicy,
    list_quarterly_target_releases_asof,
    resolve_quarterly_final_target,
    resolve_target_from_releases,
)
from nowcast_data.pit.adapters.base import PITAdapter
from nowcast_data.pit.core.catalog import SeriesCatalog
from nowcast_data.pit.core.models import SeriesMetadata


@dataclass
class VintageTrainingDatasetConfig:
    target_series_key: str
    predictor_series_keys: list[str]
    agg_spec: dict[str, str]
    include_partial_quarters: bool = True
    ref_offsets: list[int] = field(default_factory=lambda: [-1, 0, 1])
    evaluation_asof_date: date | None = None
    final_target_policy: TargetPolicy = field(
        default_factory=lambda: TargetPolicy(mode="latest_available", max_release_rank=3)
    )
    target_feature_spec: QuarterlyTargetFeatureSpec | None = None


def _fetch_asof_series(
    adapter: PITAdapter,
    series_key: str,
    asof_date: date,
    metadata: SeriesMetadata | None,
    ingest_from_ctx_source: bool,
):
    try:
        return adapter.fetch_asof(
            series_key,
            asof_date,
            metadata=metadata,
            ingest_from_ctx_source=ingest_from_ctx_source,
        )
    except TypeError:
        return adapter.fetch_asof(series_key, asof_date, metadata=metadata)


def _build_predictor_frame(
    adapter: PITAdapter,
    catalog: SeriesCatalog | None,
    *,
    predictor_series_keys: Iterable[str],
    agg_spec: dict[str, str],
    asof_date: date,
    include_partial_quarters: bool,
    ingest_from_ctx_source: bool,
) -> tuple[pd.DataFrame, pd.Period, dict[str, int], dict[str, date | None]]:
    predictor_series_keys = list(predictor_series_keys)
    extra_agg_keys = set(agg_spec) - set(predictor_series_keys)
    if extra_agg_keys:
        raise ValueError(f"agg_spec contains non-predictor keys: {sorted(extra_agg_keys)}")
    allowed_methods = {"mean", "sum", "last"}
    invalid_methods = {
        key: method for key, method in agg_spec.items() if method.lower() not in allowed_methods
    }
    if invalid_methods:
        formatted = ", ".join(f"{key}={method}" for key, method in invalid_methods.items())
        raise ValueError(f"agg_spec contains invalid methods: {formatted}")

    metadata_by_key: dict[str, SeriesMetadata] = {}
    if catalog is not None:
        for key in predictor_series_keys:
            meta = catalog.get(key)
            if meta is not None:
                metadata_by_key[key] = meta

    raw_by_key: dict[str, pd.Series] = {}
    for series_key in predictor_series_keys:
        meta = metadata_by_key.get(series_key)
        observations = _fetch_asof_series(
            adapter,
            series_key,
            asof_date,
            meta,
            ingest_from_ctx_source,
        )
        if not observations:
            raw_by_key[series_key] = pd.Series(dtype="float64")
            continue
        data = pd.DataFrame(
            {
                "obs_date": [obs.obs_date for obs in observations],
                "value": [obs.value for obs in observations],
            }
        )
        obs_dates_utc = pd.to_datetime(data["obs_date"], utc=True, errors="coerce")
        if obs_dates_utc.isna().any():
            raise ValueError(f"Series '{series_key}' has unparseable obs_date values.")
        obs_dates_naive = _to_utc_naive(obs_dates_utc)
        if meta is not None and str(meta.frequency).lower() == "m":
            non_month_end = obs_dates_naive.loc[~obs_dates_naive.dt.is_month_end]
            if not non_month_end.empty:
                sample = non_month_end.dt.strftime("%Y-%m-%d").unique()[:3].tolist()
                raise ValueError(
                    "Monthly predictor series "
                    f"'{series_key}' has non-month-end obs_date values: {sample}"
                )
        series = pd.Series(data["value"].to_numpy(), index=pd.DatetimeIndex(obs_dates_naive))
        raw_by_key[series_key] = series.sort_index()

    quarter_index = pd.Index(
        sorted(
            {
                _to_quarter_period(ts)
                for series in raw_by_key.values()
                for ts in series.index
            }
        ),
        name="ref_quarter",
    )

    current_quarter = pd.Period(pd.Timestamp(asof_date), freq="Q")
    if include_partial_quarters:
        quarter_index = (
            pd.Index(quarter_index, name="ref_quarter")
            .union(pd.Index([current_quarter], name="ref_quarter"))
            .sort_values()
        )
        keep_quarters = quarter_index <= current_quarter
    else:
        keep_quarters = quarter_index < current_quarter
    quarter_index = quarter_index[keep_quarters]

    predictor_frame = pd.DataFrame(index=quarter_index)
    nobs_current: dict[str, int] = {}
    last_obs_date_current_quarter: dict[str, date | None] = {}
    for series_key in predictor_series_keys:
        series = raw_by_key.get(series_key, pd.Series(dtype="float64"))
        if series.empty:
            predictor_frame[series_key] = np.nan
            nobs_current[series_key] = 0
            last_obs_date_current_quarter[series_key] = None
            continue
        series = series.sort_index()
        grouped = series.groupby(series.index.map(_to_quarter_period))
        agg_method = agg_spec.get(series_key, "mean").lower()
        predictor_frame[series_key] = grouped.apply(lambda values: _agg_series(values, agg_method))
        predictor_frame[series_key] = predictor_frame[series_key].reindex(quarter_index)
        current_series = series.loc[series.index.map(_to_quarter_period) == current_quarter]
        nobs_current[series_key] = int(current_series.notna().sum())
        last_obs_date_current_quarter[series_key] = (
            pd.Timestamp(current_series.index.max()).date() if not current_series.empty else None
        )

    return predictor_frame, current_quarter, nobs_current, last_obs_date_current_quarter


def build_vintage_training_dataset(
    adapter: PITAdapter,
    catalog: SeriesCatalog | None,
    *,
    config: VintageTrainingDatasetConfig,
    asof_date: date,
    ingest_from_ctx_source: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """Build a vintage training dataset with optional target-derived features."""
    if config.target_series_key in config.predictor_series_keys:
        raise ValueError("target_series_key must not be in predictor_series_keys")

    predictor_frame, current_quarter, nobs_current, last_obs_date_current_quarter = (
        _build_predictor_frame(
            adapter,
            catalog,
            predictor_series_keys=config.predictor_series_keys,
            agg_spec=config.agg_spec,
            asof_date=asof_date,
            include_partial_quarters=config.include_partial_quarters,
            ingest_from_ctx_source=ingest_from_ctx_source,
        )
    )

    desired_quarters = sorted({current_quarter + offset for offset in config.ref_offsets})
    desired_index = pd.PeriodIndex(desired_quarters, freq="Q", name="ref_quarter")
    predictor_frame = predictor_frame.reindex(desired_index)

    if config.evaluation_asof_date is None:
        raise ValueError("evaluation_asof_date must be provided in VintageTrainingDatasetConfig")
    evaluation_asof_date = config.evaluation_asof_date
    y_asof_values: list[float] = []
    y_final_values: list[float] = []
    feature_rows: list[dict[str, float]] = []
    target_meta_by_ref: dict[str, dict] = {}

    latest_policy = TargetPolicy(mode="latest_available")

    for ref_quarter in desired_index:
        releases = list_quarterly_target_releases_asof(
            adapter,
            series_key=config.target_series_key,
            ref_quarter=str(ref_quarter),
            asof_date=asof_date,
        )
        y_asof_value, y_asof_meta = resolve_target_from_releases(releases, latest_policy)
        y_asof_values.append(y_asof_value if y_asof_value is not None else np.nan)

        y_final_value, y_final_meta = resolve_quarterly_final_target(
            adapter,
            series_key=config.target_series_key,
            ref_quarter=str(ref_quarter),
            evaluation_asof_date=evaluation_asof_date,
            policy=config.final_target_policy,
        )
        y_final_values.append(y_final_value if y_final_value is not None else np.nan)

        feature_meta = None
        features: dict[str, float] = {}
        if config.target_feature_spec is not None:
            features, feature_meta = get_quarterly_target_release_features(
                adapter,
                series_key=config.target_series_key,
                ref_quarter=str(ref_quarter),
                asof_date=asof_date,
                spec=config.target_feature_spec,
            )
        feature_rows.append(features)

        target_meta_by_ref[str(ref_quarter)] = {
            "y_asof_latest": y_asof_meta,
            "y_final": y_final_meta,
            "features": feature_meta,
        }

    dataset = predictor_frame.copy()
    if feature_rows:
        feature_frame = pd.DataFrame(feature_rows, index=desired_index)
        dataset = dataset.join(feature_frame)

    dataset["y_asof_latest"] = pd.Series(y_asof_values, index=desired_index, dtype="float64")
    dataset["y_final"] = pd.Series(y_final_values, index=desired_index, dtype="float64")
    dataset.index.name = "ref_quarter"

    meta = {
        "current_ref_quarter": str(current_quarter),
        "nobs_current": nobs_current,
        "last_obs_date_current_quarter": last_obs_date_current_quarter,
        "target_release_meta": target_meta_by_ref,
    }
    return dataset, meta

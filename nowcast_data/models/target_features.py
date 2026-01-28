from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from nowcast_data.pit.adapters.base import PITAdapter
from nowcast_data.models.target_policy import (
    TargetPolicy,
    list_quarterly_target_releases_asof,
    resolve_target_from_releases,
)


@dataclass
class QuarterlyTargetFeatureSpec:
    """Specification for quarterly target-derived features."""

    include_release_values: bool = True
    release_ranks: list[int] = field(default_factory=lambda: [1, 2, 3])
    include_latest_available: bool = True
    include_revision_deltas: bool = True
    include_release_metadata: bool = True
    max_release_rank: int | None = 3


def _initialize_feature_key(series_key: str, suffix: str) -> str:
    return f"{series_key}.{suffix}"


def get_quarterly_target_release_features(
    adapter: PITAdapter,
    *,
    series_key: str,
    ref_quarter: str | pd.Period,
    asof_date: date,
    spec: QuarterlyTargetFeatureSpec,
) -> tuple[dict[str, float], dict]:
    """Construct release-derived target features for a quarterly reference."""
    releases = list_quarterly_target_releases_asof(
        adapter,
        series_key=series_key,
        ref_quarter=ref_quarter,
        asof_date=asof_date,
    )
    if spec.max_release_rank is not None:
        if spec.max_release_rank < 1:
            raise ValueError("max_release_rank must be >= 1")
        releases = releases.head(spec.max_release_rank).reset_index(drop=True)

    k = len(releases)
    features: dict[str, float] = {}
    meta: dict[str, Any] = {
        "series_key": series_key,
        "ref_quarter": str(ref_quarter),
        "n_releases_available": k,
    }

    release_values: dict[int, float] = {}
    release_asof: dict[int, pd.Timestamp] = {}
    for rank, row in enumerate(releases.itertuples(index=False), start=1):
        release_values[rank] = float(row.value) if pd.notna(row.value) else np.nan
        release_asof[rank] = row.asof_utc

    if spec.include_release_values:
        for rank in spec.release_ranks:
            features[_initialize_feature_key(series_key, f"rel{rank}")] = release_values.get(
                rank, np.nan
            )

    if spec.include_latest_available:
        latest_value, latest_meta = resolve_target_from_releases(
            releases,
            TargetPolicy(mode="latest_available", max_release_rank=spec.max_release_rank),
        )
        features[_initialize_feature_key(series_key, "latest")] = (
            float(latest_value) if latest_value is not None else np.nan
        )
        meta["latest_release_meta"] = latest_meta

    if spec.include_release_metadata:
        features[_initialize_feature_key(series_key, "n_releases")] = float(k)
        if k > 0:
            last_asof = releases["asof_utc"].iloc[-1]
            asof_ts = pd.Timestamp(asof_date, tz="UTC").normalize() + pd.Timedelta(days=1)
            days_since_last = (asof_ts - last_asof).days
            features[_initialize_feature_key(series_key, "days_since_last")] = float(
                max(days_since_last, 0)
            )
        else:
            features[_initialize_feature_key(series_key, "days_since_last")] = np.nan

    if spec.include_revision_deltas:
        rel1 = release_values.get(1, np.nan)
        rel2 = release_values.get(2, np.nan)
        rel3 = release_values.get(3, np.nan)
        latest_value = features.get(_initialize_feature_key(series_key, "latest"), np.nan)
        features[_initialize_feature_key(series_key, "rel2_minus_rel1")] = (
            rel2 - rel1 if not np.isnan(rel2) and not np.isnan(rel1) else np.nan
        )
        features[_initialize_feature_key(series_key, "rel3_minus_rel2")] = (
            rel3 - rel2 if not np.isnan(rel3) and not np.isnan(rel2) else np.nan
        )
        features[_initialize_feature_key(series_key, "latest_minus_rel1")] = (
            latest_value - rel1
            if not np.isnan(latest_value) and not np.isnan(rel1)
            else np.nan
        )

    meta["release_asof_utc"] = release_asof
    meta["releases"] = releases
    return features, meta

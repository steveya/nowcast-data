from datetime import date

import pandas as pd
import pytest

from nowcast_data.models.target_policy import TargetPolicy, resolve_target_from_releases


def test_empty_releases_returns_none() -> None:
    releases = pd.DataFrame(columns=["obs_date", "asof_utc", "value"])
    value, meta = resolve_target_from_releases(releases, TargetPolicy(mode="latest_available"))
    assert value is None
    assert meta["selected_release_rank"] is None
    assert meta["n_releases_available"] == 0


def test_nan_value_latest_available_returns_none() -> None:
    releases = pd.DataFrame(
        [
            {
                "obs_date": pd.Timestamp("2025-03-31", tz="UTC"),
                "asof_utc": pd.Timestamp("2025-04-25", tz="UTC"),
                "value": float("nan"),
            }
        ]
    )
    value, meta = resolve_target_from_releases(releases, TargetPolicy(mode="latest_available"))
    assert value is None
    assert meta["selected_release_rank"] == 1
    assert meta["selected_release_asof_utc"] is None


def test_invalid_max_release_rank_raises() -> None:
    releases = pd.DataFrame(
        [
            {
                "obs_date": pd.Timestamp("2025-03-31", tz="UTC"),
                "asof_utc": pd.Timestamp("2025-04-25", tz="UTC"),
                "value": 1.0,
            }
        ]
    )
    with pytest.raises(ValueError, match="max_release_rank must be >= 1"):
        resolve_target_from_releases(releases, TargetPolicy(mode="latest_available", max_release_rank=0))


def test_max_release_rank_caps_next_release() -> None:
    releases = pd.DataFrame(
        [
            {
                "obs_date": pd.Timestamp("2025-03-31", tz="UTC"),
                "asof_utc": pd.Timestamp("2025-04-25", tz="UTC"),
                "value": 1.0,
            }
        ]
    )
    value, meta = resolve_target_from_releases(
        releases, TargetPolicy(mode="next_release", max_release_rank=1)
    )
    assert value is None
    assert meta["selected_release_rank"] is None
    assert meta["target_release_rank"] == 1

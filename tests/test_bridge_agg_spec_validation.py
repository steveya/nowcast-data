from datetime import date

import pytest

pytest.importorskip("alphaforge")
from nowcast_data.models.bridge import build_rt_quarterly_dataset  # noqa: E402
from nowcast_data.pit.adapters.alphaforge import AlphaForgePITAdapter  # noqa: E402


def test_agg_spec_extra_keys_raises(pit_context) -> None:
    adapter = AlphaForgePITAdapter(ctx=pit_context)
    with pytest.raises(ValueError, match="agg_spec contains non-predictor keys"):
        build_rt_quarterly_dataset(
            adapter,
            None,
            target_series_key="BASE_GDP",
            predictor_series_keys=["P1"],
            agg_spec={"P1": "mean", "P2": "mean"},
            asof_date=date(2025, 5, 15),
        )


def test_agg_spec_invalid_method_raises(pit_context) -> None:
    adapter = AlphaForgePITAdapter(ctx=pit_context)
    with pytest.raises(ValueError, match="agg_spec contains invalid methods"):
        build_rt_quarterly_dataset(
            adapter,
            None,
            target_series_key="BASE_GDP",
            predictor_series_keys=["P1"],
            agg_spec={"P1": "median"},
            asof_date=date(2025, 5, 15),
        )


def test_target_in_predictors_raises(pit_context) -> None:
    adapter = AlphaForgePITAdapter(ctx=pit_context)
    with pytest.raises(ValueError, match="target_series_key must not be in predictor_series_keys"):
        build_rt_quarterly_dataset(
            adapter,
            None,
            target_series_key="BASE_GDP",
            predictor_series_keys=["BASE_GDP"],
            agg_spec={"BASE_GDP": "mean"},
            asof_date=date(2025, 5, 15),
        )

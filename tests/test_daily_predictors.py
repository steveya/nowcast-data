"""Tests for daily predictor support and cutoff logic."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from nowcast_data.models.bridge import build_rt_quarterly_dataset
from nowcast_data.models.datasets import VintageTrainingDatasetConfig
from nowcast_data.models.panel import build_vintage_panel_dataset
from nowcast_data.pit.adapters.alphaforge import AlphaForgePITAdapter
from nowcast_data.pit.core.catalog import SeriesCatalog
from nowcast_data.pit.core.models import SeriesMetadata


def _daily_catalog() -> SeriesCatalog:
    catalog = SeriesCatalog()
    catalog.add(
        SeriesMetadata(
            series_key="DAILY_FCI",
            country="US",
            source="TEST",
            source_series_id="DAILY_FCI",
            frequency="d",
            pit_mode="NO_PIT",
        )
    )
    return catalog


class TestDailyPredictors:
    def test_daily_cutoff_no_leakage(self, pit_context) -> None:
        """Current-quarter daily features must exclude obs_date after asof_date."""
        adapter = AlphaForgePITAdapter(ctx=pit_context)
        catalog = _daily_catalog()

        asof_date = date(2025, 2, 10)
        dataset, _, _ = build_rt_quarterly_dataset(
            adapter,
            catalog,
            target_series_key="BASE_GDP",
            predictor_series_keys=["DAILY_FCI"],
            agg_spec={"DAILY_FCI": "mean"},
            asof_date=asof_date,
            include_partial_quarters=True,
        )

        current_quarter = pd.Period("2025Q1", freq="Q")
        assert current_quarter in dataset.index

        row = dataset.loc[current_quarter]
        assert "DAILY_FCI.last" in dataset.columns
        assert "DAILY_FCI.mean_5d" in dataset.columns
        assert "DAILY_FCI.mean_20d" in dataset.columns
        assert "DAILY_FCI.std_20d" in dataset.columns
        assert "DAILY_FCI.n_obs" in dataset.columns

        # Expected values from obs_date <= 2025-02-10 only
        assert np.isclose(row["DAILY_FCI.last"], 14.0)
        assert np.isclose(row["DAILY_FCI.mean_5d"], 12.0)
        assert np.isclose(row["DAILY_FCI.mean_20d"], 12.0)
        assert np.isclose(row["DAILY_FCI.std_20d"], np.sqrt(2.0))
        assert np.isclose(row["DAILY_FCI.n_obs"], 5.0)

    def test_include_partial_quarters_flag_behavior(self, pit_context) -> None:
        """When include_partial_quarters=False, current quarter is excluded."""
        adapter = AlphaForgePITAdapter(ctx=pit_context)
        catalog = _daily_catalog()

        dataset, _, _ = build_rt_quarterly_dataset(
            adapter,
            catalog,
            target_series_key="BASE_GDP",
            predictor_series_keys=["DAILY_FCI"],
            agg_spec={"DAILY_FCI": "mean"},
            asof_date=date(2025, 2, 10),
            include_partial_quarters=False,
        )

        current_quarter = pd.Period("2025Q1", freq="Q")
        assert current_quarter not in dataset.index

    def test_daily_features_exist_in_panel(self, pit_context) -> None:
        """Panel dataset should include expanded daily feature columns."""
        adapter = AlphaForgePITAdapter(ctx=pit_context)
        catalog = _daily_catalog()

        config = VintageTrainingDatasetConfig(
            target_series_key="BASE_GDP",
            predictor_series_keys=["DAILY_FCI"],
            agg_spec={"DAILY_FCI": "mean"},
            include_partial_quarters=True,
            ref_offsets=[0],
            evaluation_asof_date=date(2025, 12, 31),
        )

        xy_panel, _ = build_vintage_panel_dataset(
            adapter,
            catalog,
            config=config,
            vintages=[date(2025, 2, 10)],
        )

        assert not xy_panel.empty
        assert "DAILY_FCI.last" in xy_panel.columns
        assert "DAILY_FCI.mean_5d" in xy_panel.columns
        assert "DAILY_FCI.mean_20d" in xy_panel.columns
        assert "DAILY_FCI.std_20d" in xy_panel.columns
        assert "DAILY_FCI.n_obs" in xy_panel.columns

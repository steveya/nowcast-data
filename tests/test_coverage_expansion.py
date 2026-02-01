from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest
import requests

from nowcast_data.benchmark.manifest import load_manifest
from nowcast_data.benchmark.pit_builder import (
    SeriesSpec,
    apply_benchmark_transforms,
    build_monthly_panel_asof,
    compute_vintage_grid,
    load_benchmark_metadata,
)
from nowcast_data.features.recipes import FeatureRecipe, RECIPE_REGISTRY, build_agg_spec_from_recipes
from nowcast_data.features.transformers import MissingnessFilter, QuarterlyFeatureBuilder
from nowcast_data.pit.adapters.alphaforge import AlphaForgePITAdapter
from nowcast_data.pit.adapters.base import PITAdapter
from nowcast_data.pit.adapters.fred import FREDALFREDAdapter
from nowcast_data.pit.api import PITDataManager
from nowcast_data.pit.core.catalog import SeriesCatalog
from nowcast_data.pit.core.models import PITObservation, SeriesMetadata
from nowcast_data.pit.core.vintage_logic import select_vintage_for_asof, validate_no_lookahead
from nowcast_data.pit.exceptions import PITNotSupportedError, SourceFetchError, VintageNotFoundError


class _DummyResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def json(self) -> dict:
        return self._payload

    def raise_for_status(self) -> None:
        return None


class _DummyAdapter(PITAdapter):
    def __init__(self, observations: list[PITObservation]) -> None:
        self._observations = observations

    @property
    def name(self) -> str:
        return "dummy"

    def supports_pit(self, series_id: str) -> bool:
        return True

    def list_vintages(self, series_id: str) -> list[date]:
        return [date(2020, 1, 1)]

    def fetch_asof(
        self,
        series_id: str,
        asof_date: date,
        start: date | None = None,
        end: date | None = None,
        *,
        metadata: SeriesMetadata | None = None,
    ) -> list[PITObservation]:
        return self._observations


def test_exceptions_default_messages() -> None:
    assert "ABC" in str(PITNotSupportedError("ABC"))
    assert "XYZ" in str(VintageNotFoundError("XYZ", date(2020, 1, 1)))
    err = SourceFetchError("FRED_ALFRED", original_error=ValueError("boom"))
    assert "FRED_ALFRED" in str(err)


def test_manifest_invalid_json(tmp_path: Path) -> None:
    bad_file = tmp_path / "manifest.json"
    bad_file.write_text("{not-json")
    with pytest.raises(ValueError, match="Failed to load benchmark manifest"):
        load_manifest(bad_file)


def test_recipe_flow_override(monkeypatch) -> None:
    monkeypatch.setitem(
        RECIPE_REGISTRY,
        "flow_mean",
        FeatureRecipe(kind="flow", agg="mean", change="pct"),
    )
    agg = build_agg_spec_from_recipes(["FLOW_MEAN"], {"FLOW_MEAN": "flow_mean"})
    assert agg["FLOW_MEAN"] == "sum"


def test_missingness_filter_requires_fit() -> None:
    filt = MissingnessFilter()
    with pytest.raises(ValueError, match="not fitted"):
        filt.transform(pd.DataFrame({"x": [1.0]}))


def test_quarterly_feature_builder_requires_meta_columns() -> None:
    builder = QuarterlyFeatureBuilder(predictor_keys=["x"])
    df = pd.DataFrame({"x": [1.0]})
    with pytest.raises(ValueError, match="missing"):
        builder.transform(df)


def test_benchmark_transforms_and_panel(pit_context) -> None:
    specs = [
        SeriesSpec(series_id="BASE_GDP", series_key="base_gdp", frequency="q", transform="level"),
        SeriesSpec(series_id="DAILY_FCI", series_key="daily_fci", frequency="m", transform="diff"),
        SeriesSpec(series_id="GDP", series_key="gdp", frequency="q", transform="logdiff"),
        SeriesSpec(series_id="US_GDP_SAAR", series_key="us_gdp_saar", frequency="q", transform="pctchange"),
        SeriesSpec(series_id="BAD", series_key="bad", frequency="m", transform="unknown"),
    ]
    vintages = compute_vintage_grid(pit_context, specs)
    assert vintages == []
    panel = pd.DataFrame(
        {
            spec.series_key: pd.Series([1.0, 2.0], index=pd.to_datetime(["2025-01-31", "2025-02-28"]))
            for spec in specs
        }
    )
    transformed = apply_benchmark_transforms(panel, specs)
    assert set(transformed.columns) == {spec.series_key for spec in specs}


def test_load_benchmark_metadata(tmp_path: Path) -> None:
    csv_path = tmp_path / "meta.csv"
    csv_path.write_text("series,name,freq,transform\nGDP,GDP,q,level\n")
    specs = load_benchmark_metadata(csv_path)
    assert specs[0].series_key == "gdp"


def test_vintage_logic_helpers() -> None:
    vintages = [date(2020, 1, 1), date(2020, 2, 1)]
    assert select_vintage_for_asof(vintages, date(2020, 1, 15)) == date(2020, 1, 1)
    assert validate_no_lookahead(date(2020, 1, 1), date(2020, 1, 1))
    assert not validate_no_lookahead(date(2020, 2, 1), date(2020, 1, 1))


def test_pit_api_alpha_paths(pit_context) -> None:
    catalog = SeriesCatalog()
    catalog.add(
        SeriesMetadata(
            series_key="GDP",
            country="US",
            source="alphaforge",
            source_series_id="GDP",
            frequency="Q",
            pit_mode="ALFRED_REALTIME",
            adapter="alphaforge",
        )
    )
    adapter = AlphaForgePITAdapter(ctx=pit_context)
    manager = PITDataManager(catalog, adapters={"alphaforge": adapter})
    df = manager.get_series_asof("GDP", date(2025, 2, 15))
    assert not df.empty
    assert manager.get_series_vintages("GDP")

    panel = manager.get_panel_asof(["GDP", "UNKNOWN"], date(2025, 2, 15), wide=False)
    assert "series_key" in panel.columns


def test_pit_api_no_pit_raises() -> None:
    catalog = SeriesCatalog()
    catalog.add(
        SeriesMetadata(
            series_key="NOPIT",
            country="US",
            source="dummy",
            source_series_id="NOPIT",
            frequency="Q",
            pit_mode="NO_PIT",
        )
    )
    manager = PITDataManager(catalog, adapters={"dummy": _DummyAdapter([])})
    with pytest.raises(PITNotSupportedError):
        manager.get_series_asof("NOPIT", date(2025, 1, 1))


def test_pit_api_build_cube(pit_context) -> None:
    catalog = SeriesCatalog()
    catalog.add(
        SeriesMetadata(
            series_key="GDP",
            country="US",
            source="alphaforge",
            source_series_id="GDP",
            frequency="Q",
            pit_mode="ALFRED_REALTIME",
            adapter="alphaforge",
        )
    )
    adapter = AlphaForgePITAdapter(ctx=pit_context)
    manager = PITDataManager(catalog, adapters={"alphaforge": adapter})
    cube = manager.build_pit_cube(
        ["GDP"],
        [date(2025, 1, 10), date(2025, 2, 10)],
        start=date(2024, 1, 1),
        end=date(2025, 12, 31),
    )
    assert not cube.empty


def test_fred_list_vintages_cache(monkeypatch) -> None:
    adapter = FREDALFREDAdapter(api_key="test")
    calls = {"count": 0}

    def fake_request(url: str, params: dict) -> _DummyResponse:
        calls["count"] += 1
        return _DummyResponse({"vintage_dates": ["2020-01-01", "2020-02-01"]})

    monkeypatch.setattr(adapter, "_request_with_retry", fake_request)
    vintages = adapter.list_vintages("GDP")
    assert vintages == [date(2020, 1, 1), date(2020, 2, 1)]
    assert adapter.list_vintages("GDP") == vintages
    assert calls["count"] == 1


def test_fred_fetch_asof_parses(monkeypatch) -> None:
    adapter = FREDALFREDAdapter(api_key="test")
    monkeypatch.setattr(adapter, "list_vintages", lambda series_id: [date(2020, 1, 1)])

    def fake_request(url: str, params: dict) -> _DummyResponse:
        return _DummyResponse(
            {
                "observations": [
                    {"date": "2020-01-01", "value": ".", "realtime_start": "2020-01-01"},
                    {"date": "2020-02-01", "value": "2.5", "realtime_start": "2020-02-01"},
                ]
            }
        )

    monkeypatch.setattr(adapter, "_request_with_retry", fake_request)
    obs = adapter.fetch_asof("GDP", date(2020, 2, 1), start=date(2019, 1, 1))
    assert len(obs) == 1
    assert obs[0].value == 2.5
    assert adapter._infer_frequency("2020-01-01") == "M"


def test_fred_request_with_retry(monkeypatch) -> None:
    adapter = FREDALFREDAdapter(api_key="test")
    attempts = {"count": 0}

    def fake_get(url: str, params: dict, timeout: int):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise requests.exceptions.RequestException("boom")
        return _DummyResponse({})

    monkeypatch.setattr(adapter._session, "get", fake_get)
    response = adapter._request_with_retry("http://example.com", {}, max_retries=2)
    assert isinstance(response, _DummyResponse)
    assert attempts["count"] == 2


def test_fred_supports_pit_false(monkeypatch) -> None:
    adapter = FREDALFREDAdapter(api_key="test")

    def fail_list_vintages(series_id: str) -> list[date]:
        raise SourceFetchError("FRED_ALFRED")

    monkeypatch.setattr(adapter, "list_vintages", fail_list_vintages)
    assert adapter.supports_pit("GDP") is False

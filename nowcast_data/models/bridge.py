from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV

from nowcast_data.pit.adapters.alphaforge import AlphaForgePITAdapter
from nowcast_data.pit.api import PITDataManager
from nowcast_data.pit.core.catalog import SeriesCatalog
from nowcast_data.pit.core.models import SeriesMetadata
from nowcast_data.time.nowcast_calendar import (
    get_target_asof_ref,
    infer_current_quarter,
    refperiod_to_quarter_end,
)


@dataclass
class BridgeConfig:
    target_series_key: str
    predictor_series_keys: list[str]
    agg_spec: dict[str, str]
    standardize: bool = True
    model: str = "ridge"
    alphas: list[float] = field(default_factory=lambda: list(np.logspace(-4, 4, 20)))
    min_train_quarters: int = 20
    include_partial_quarters: bool = True


def _to_quarter_period(ts: pd.Timestamp) -> pd.Period:
    ts = pd.Timestamp(ts)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts.to_period("Q")


def _agg_series(series: pd.Series, method: str) -> float:
    if series.empty:
        return np.nan
    if method == "sum":
        return float(series.sum())
    if method == "last":
        return float(series.iloc[-1])
    return float(series.mean())


def build_rt_quarterly_dataset(
    adapter: AlphaForgePITAdapter,
    catalog: SeriesCatalog | None,
    *,
    target_series_key: str,
    predictor_series_keys: Iterable[str],
    agg_spec: dict[str, str],
    asof_date: date,
    include_partial_quarters: bool = True,
) -> tuple[pd.DataFrame, dict[str, int], dict[str, int]]:
    predictor_series_keys = list(predictor_series_keys)
    series_keys = [target_series_key, *predictor_series_keys]
    metadata_by_key: dict[str, SeriesMetadata] = {}
    if catalog is not None:
        for key in series_keys:
            meta = catalog.get(key)
            if meta is not None:
                metadata_by_key[key] = meta

    raw_by_key: dict[str, pd.Series] = {}
    for series_key in series_keys:
        meta = metadata_by_key.get(series_key)
        query_key = (
            meta.source_series_id if meta is not None and meta.source_series_id else series_key
        )
        observations = adapter.fetch_asof(query_key, asof_date, metadata=meta)
        if not observations:
            raw_by_key[series_key] = pd.Series(dtype="float64")
            continue
        data = pd.DataFrame(
            {
                "obs_date": [obs.obs_date for obs in observations],
                "value": [obs.value for obs in observations],
            }
        )
        data["obs_date"] = pd.to_datetime(data["obs_date"])
        series = pd.Series(data["value"].to_numpy(), index=pd.DatetimeIndex(data["obs_date"]))
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

    current_quarter = _to_quarter_period(refperiod_to_quarter_end(infer_current_quarter(asof_date)))
    if include_partial_quarters:
        keep_quarters = quarter_index <= current_quarter
    else:
        keep_quarters = quarter_index < current_quarter
    quarter_index = quarter_index[keep_quarters]
    if include_partial_quarters and current_quarter not in quarter_index:
        quarter_index = quarter_index.append(pd.Index([current_quarter], name="ref_quarter"))
        quarter_index = quarter_index.sort_values()

    if quarter_index.empty:
        dataset = pd.DataFrame(columns=["y"], index=quarter_index)
        return dataset, {}, {}

    predictor_frame = pd.DataFrame(index=quarter_index)
    nobs_current: dict[str, int] = {}
    for series_key in predictor_series_keys:
        series = raw_by_key.get(series_key, pd.Series(dtype="float64"))
        if series.empty:
            predictor_frame[series_key] = np.nan
            nobs_current[series_key] = 0
            continue
        series = series.sort_index()
        grouped = series.groupby(series.index.map(_to_quarter_period))
        agg_method = agg_spec.get(series_key, "mean").lower()
        predictor_frame[series_key] = grouped.apply(lambda values: _agg_series(values, agg_method))
        predictor_frame[series_key] = predictor_frame[series_key].reindex(quarter_index)
        current_series = series.loc[series.index.map(_to_quarter_period) == current_quarter]
        nobs_current[series_key] = int(current_series.notna().sum())

    target_series = raw_by_key.get(target_series_key, pd.Series(dtype="float64"))
    if not target_series.empty:
        target_series = target_series.sort_index()
        target_quarters = target_series.groupby(target_series.index.map(_to_quarter_period)).last()
        target_quarters = target_quarters.reindex(quarter_index)
    else:
        target_quarters = pd.Series(index=quarter_index, dtype="float64")

    dataset = predictor_frame.copy()
    dataset["y"] = target_quarters
    dataset.index.name = "ref_quarter"
    return dataset, nobs_current, {key: len(series) for key, series in raw_by_key.items()}


class BridgeNowcaster:
    def __init__(
        self,
        config: BridgeConfig,
        pit_manager_or_adapter: PITDataManager | AlphaForgePITAdapter,
        catalog: SeriesCatalog | None = None,
    ) -> None:
        self.config = config
        if isinstance(pit_manager_or_adapter, PITDataManager):
            adapter = pit_manager_or_adapter.adapters.get("alphaforge")
            if adapter is None:
                raise ValueError("PITDataManager missing alphaforge adapter")
            self.adapter = adapter
            self.catalog = pit_manager_or_adapter.catalog
        else:
            self.adapter = pit_manager_or_adapter
            self.catalog = catalog

    def _prepare_features(self, dataset: pd.DataFrame, current_quarter: pd.Period) -> tuple[
        pd.DataFrame,
        pd.Series,
        pd.Series,
        dict[str, pd.Series],
    ]:
        predictors = dataset.drop(columns=["y"])
        target = dataset["y"]

        train_mask = predictors.index < current_quarter
        train_X = predictors.loc[train_mask].copy()
        train_y = target.loc[train_mask].copy()

        nan_frac = train_X.isna().mean()
        train_X = train_X.loc[:, nan_frac <= 0.5]

        means = train_X.mean()
        train_X = train_X.fillna(means)
        current_X = predictors.loc[[current_quarter]].copy()
        current_X = current_X.reindex(columns=train_X.columns).fillna(means)

        if self.config.standardize and not train_X.empty:
            stds = train_X.std(ddof=0).replace(0.0, 1.0)
            train_X = (train_X - means) / stds
            current_X = (current_X - means) / stds
        else:
            stds = pd.Series(1.0, index=train_X.columns)

        stats = {"mean": means, "std": stds}
        return train_X, train_y, current_X.iloc[0], stats

    def fit_predict_one(self, asof_date: date) -> dict:
        current_ref = infer_current_quarter(asof_date)
        current_quarter = _to_quarter_period(refperiod_to_quarter_end(current_ref))

        dataset, nobs_current, _ = build_rt_quarterly_dataset(
            self.adapter,
            self.catalog,
            target_series_key=self.config.target_series_key,
            predictor_series_keys=self.config.predictor_series_keys,
            agg_spec=self.config.agg_spec,
            asof_date=asof_date,
            include_partial_quarters=self.config.include_partial_quarters,
        )

        if dataset.empty or current_quarter not in dataset.index:
            return {
                "asof_date": asof_date,
                "ref_quarter": str(current_ref),
                "y_true_asof": np.nan,
                "y_pred": np.nan,
                "n_train": 0,
                "n_features": 0,
                "alpha_selected": np.nan,
                "mean_months_observed": np.nan,
            }

        train_X, train_y, current_X, _ = self._prepare_features(dataset, current_quarter)
        train_y = train_y.dropna()
        train_X = train_X.loc[train_y.index]

        if len(train_y) < self.config.min_train_quarters or train_X.empty:
            return {
                "asof_date": asof_date,
                "ref_quarter": str(current_ref),
                "y_true_asof": np.nan,
                "y_pred": np.nan,
                "n_train": len(train_y),
                "n_features": train_X.shape[1],
                "alpha_selected": np.nan,
                "mean_months_observed": float(np.mean(list(nobs_current.values())))
                if nobs_current
                else np.nan,
            }

        model = RidgeCV(alphas=self.config.alphas)
        model.fit(train_X.to_numpy(), train_y.to_numpy())
        y_pred = float(model.predict(current_X.to_numpy().reshape(1, -1))[0])

        y_true = get_target_asof_ref(
            self.adapter,
            self.config.target_series_key,
            asof_date,
            ref=current_ref,
        )

        return {
            "asof_date": asof_date,
            "ref_quarter": str(current_ref),
            "y_true_asof": y_true if y_true is not None else np.nan,
            "y_pred": y_pred,
            "n_train": len(train_y),
            "n_features": train_X.shape[1],
            "alpha_selected": float(model.alpha_),
            "mean_months_observed": float(np.mean(list(nobs_current.values())))
            if nobs_current
            else np.nan,
        }

    def predict_many(self, vintages: Iterable[date]) -> pd.DataFrame:
        rows = [self.fit_predict_one(vintage) for vintage in vintages]
        df = pd.DataFrame(rows)
        if not df.empty:
            df["asof_date"] = pd.to_datetime(df["asof_date"]).dt.date
        return df

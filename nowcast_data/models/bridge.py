from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, RidgeCV

from nowcast_data.pit.adapters.base import PITAdapter
from nowcast_data.pit.api import PITDataManager
from nowcast_data.pit.core.catalog import SeriesCatalog
from nowcast_data.pit.core.models import SeriesMetadata
from nowcast_data.time.nowcast_calendar import get_target_asof_ref, infer_current_quarter


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
    max_nan_fraction: float = 0.5
    ingest_from_ctx_source: bool = False


def _to_utc_naive(values: pd.Series | pd.DatetimeIndex | object) -> pd.Series | pd.DatetimeIndex | pd.Timestamp:
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


def _to_quarter_period(ts: pd.Timestamp) -> pd.Period:
    ts = _to_utc_naive(ts)
    if pd.isna(ts):
        raise ValueError("Invalid or missing observation date")
    return pd.Timestamp(ts).to_period("Q")


def _agg_series(series: pd.Series, method: str) -> float:
    if series.empty:
        return np.nan
    if method == "sum":
        return float(series.sum())
    if method == "last":
        return float(series.iloc[-1])
    if method == "mean":
        return float(series.mean())
    return float(series.mean())


def build_rt_quarterly_dataset(
    adapter: PITAdapter,
    catalog: SeriesCatalog | None,
    *,
    target_series_key: str,
    predictor_series_keys: Iterable[str],
    agg_spec: dict[str, str],
    asof_date: date,
    include_partial_quarters: bool = True,
    ingest_from_ctx_source: bool = False,
) -> tuple[pd.DataFrame, dict[str, int], dict[str, date | None]]:
    """Build quarterly aggregates for a single vintage date.

    Args:
        adapter: PIT adapter used to fetch series snapshots.
        catalog: Optional series catalog for metadata lookup.
        target_series_key: Series key for the quarterly target.
        predictor_series_keys: Series keys for monthly predictors.
        agg_spec: Mapping from series key to aggregation method.
        asof_date: Vintage date for point-in-time data.
        include_partial_quarters: Include the current quarter in the dataset.
        ingest_from_ctx_source: Allow ingest from ctx sources (default False).

    Returns:
        Tuple of (dataset, nobs_current, last_obs_date_current_quarter).
    """
    predictor_series_keys = list(predictor_series_keys)
    predictor_key_set = set(predictor_series_keys)
    series_keys = [target_series_key, *predictor_series_keys]
    metadata_by_key: dict[str, SeriesMetadata] = {}
    if catalog is not None:
        for key in series_keys:
            meta = catalog.get(key)
            if meta is not None:
                metadata_by_key[key] = meta

    raw_by_key: dict[str, pd.Series] = {}
    nobs_current: dict[str, int] = {}
    last_obs_date_current_quarter: dict[str, date | None] = {}
    for series_key in series_keys:
        meta = metadata_by_key.get(series_key)
        # Always query PIT snapshots by canonical series_key; metadata controls source ingest.
        try:
            observations = adapter.fetch_asof(
                series_key,
                asof_date,
                metadata=meta,
                ingest_from_ctx_source=ingest_from_ctx_source,
            )
        except TypeError:
            observations = adapter.fetch_asof(series_key, asof_date, metadata=meta)
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
        if series_key in predictor_key_set and (
            (meta is not None and str(meta.frequency).lower() == "m")
            or (meta is None and series_key in agg_spec)
        ):
            # Heuristic: predictors in agg_spec are treated as monthly unless metadata says otherwise.
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

    current_quarter = pd.Period(str(infer_current_quarter(asof_date)), freq="Q")
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

    if quarter_index.empty:
        dataset = pd.DataFrame(columns=["y"], index=quarter_index)
        return dataset, nobs_current, last_obs_date_current_quarter

    predictor_frame = pd.DataFrame(index=quarter_index)
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

    target_meta = metadata_by_key.get(target_series_key)
    target_values = []
    for ref_quarter in quarter_index:
        # Use ref-period snapshots for all quarters to align with PIT target semantics.
        ref_value = get_target_asof_ref(
            adapter,
            target_series_key,
            asof_date,
            ref=str(ref_quarter),
            metadata=target_meta,
        )
        target_values.append(ref_value if ref_value is not None else np.nan)
    target_quarters = pd.Series(target_values, index=quarter_index, dtype="float64")

    dataset = predictor_frame.copy()
    dataset["y"] = target_quarters
    dataset.index.name = "ref_quarter"
    return dataset, nobs_current, last_obs_date_current_quarter


class BridgeNowcaster:
    """Baseline bridge nowcasting model using quarterly aggregates.

    Fits a regression on historical quarters and predicts GDP for the current
    quarter inferred from the as-of date.
    """
    def __init__(
        self,
        config: BridgeConfig,
        pit_manager_or_adapter: PITDataManager | PITAdapter,
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
        """Prepare training/current-quarter features with basic preprocessing.

        Drops features with too many missing values, imputes remaining NaNs,
        and optionally standardizes the predictors based on the training window.
        """
        predictors = dataset.drop(columns=["y"])
        target = dataset["y"]

        train_mask = predictors.index < current_quarter
        train_X = predictors.loc[train_mask].copy()
        train_y = target.loc[train_mask].copy()

        nan_frac = train_X.isna().mean()
        train_X = train_X.loc[:, nan_frac <= self.config.max_nan_fraction]

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
        """Fit the model for a single vintage date and return diagnostics."""
        current_ref = infer_current_quarter(asof_date)
        current_quarter = pd.Period(str(current_ref), freq="Q")

        dataset, nobs_current, last_obs_date_current_quarter = build_rt_quarterly_dataset(
            self.adapter,
            self.catalog,
            target_series_key=self.config.target_series_key,
            predictor_series_keys=self.config.predictor_series_keys,
            agg_spec=self.config.agg_spec,
            asof_date=asof_date,
            include_partial_quarters=self.config.include_partial_quarters,
            ingest_from_ctx_source=self.config.ingest_from_ctx_source,
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
                "last_obs_date_current_quarter": {},
                "nobs_current": {},
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
                "last_obs_date_current_quarter": last_obs_date_current_quarter,
                "nobs_current": nobs_current,
            }

        if self.config.model == "ridge":
            model = RidgeCV(alphas=self.config.alphas)
            model.fit(train_X.to_numpy(), train_y.to_numpy())
            y_pred = float(model.predict(current_X.to_numpy().reshape(1, -1))[0])
            alpha_selected = float(model.alpha_)
        elif self.config.model == "ols":
            model = LinearRegression(fit_intercept=True)
            model.fit(train_X.to_numpy(), train_y.to_numpy())
            y_pred = float(model.predict(current_X.to_numpy().reshape(1, -1))[0])
            alpha_selected = np.nan
        else:
            raise ValueError("model must be 'ridge' or 'ols'")

        target_meta = self.catalog.get(self.config.target_series_key) if self.catalog else None
        y_true = get_target_asof_ref(
            self.adapter,
            self.config.target_series_key,
            asof_date,
            ref=current_ref,
            metadata=target_meta,
        )

        return {
            "asof_date": asof_date,
            "ref_quarter": str(current_ref),
            "y_true_asof": y_true if y_true is not None else np.nan,
            "y_pred": y_pred,
            "n_train": len(train_y),
            "n_features": train_X.shape[1],
            "alpha_selected": alpha_selected,
            "mean_months_observed": float(np.mean(list(nobs_current.values())))
            if nobs_current
            else np.nan,
            "last_obs_date_current_quarter": last_obs_date_current_quarter,
            "nobs_current": nobs_current,
        }

    def predict_many(self, vintages: Iterable[date]) -> pd.DataFrame:
        """Predict GDP for multiple vintages and return a tidy DataFrame."""
        rows = [self.fit_predict_one(vintage) for vintage in vintages]
        df = pd.DataFrame(rows)
        if not df.empty:
            df["asof_date"] = pd.to_datetime(df["asof_date"]).dt.date
        return df

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Iterable, Literal

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, RidgeCV

from nowcast_data.features import QuarterlyFeatureBuilder, compute_gdp_qoq_saar
from nowcast_data.pit.adapters.base import PITAdapter
from nowcast_data.pit.api import PITDataManager
from nowcast_data.pit.core.catalog import SeriesCatalog
from nowcast_data.pit.core.models import SeriesMetadata
from nowcast_data.time.nowcast_calendar import get_target_asof_ref, infer_current_quarter
from nowcast_data.models.utils import (
    agg_series,
    apply_quarter_cutoff,
    expand_daily_series_to_frame,
    to_quarter_period,
    to_utc_naive,
    validate_monthly_obs_dates,
)
from nowcast_data.models.target_features import QuarterlyTargetFeatureSpec
from nowcast_data.models.target_policy import TargetPolicy


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
    label: Literal["y_asof_latest", "y_final"] = "y_asof_latest"
    evaluation_asof_date: date | None = None
    include_target_release_features: bool = False
    target_feature_spec: QuarterlyTargetFeatureSpec | None = None
    final_target_policy: TargetPolicy = field(
        default_factory=lambda: TargetPolicy(mode="latest_available", max_release_rank=3)
    )
    # Deprecated: use use_real_time_target_as_feature + real_time_feature_cols instead.
    include_y_asof_latest_as_feature: bool = False
    use_real_time_target_as_feature: bool = True
    real_time_feature_cols: list[str] = field(
        default_factory=lambda: ["y_asof_latest_growth", "y_asof_latest_level"]
    )
    # training_label_mode="revision" uses y_revision = y_stable_growth - y_real_time_growth.
    # Stable predictions are reconstructed as y_pred_stable = y_true_real_time + y_pred_revision.
    training_label_mode: Literal["stable", "revision"] = "stable"
    stable_label_col: str = "y_stable_growth"
    real_time_label_col: str = "y_asof_latest_growth"


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
        predictor_series_keys: Series keys for predictors (monthly or daily).
        agg_spec: Mapping from series key to aggregation method.
        asof_date: Vintage date for point-in-time data.
        include_partial_quarters: Include the current quarter in the dataset.
        ingest_from_ctx_source: Allow ingest from ctx sources (default False).

    Returns:
        Tuple of (dataset, nobs_current, last_obs_date_current_quarter).

    Notes:
        Daily predictors (frequency "d" or "b") are expanded into multiple features:
        last, mean_5d, mean_20d, std_20d, n_obs. All predictors are filtered to
        prevent leakage beyond the as-of date within the current quarter.
    """
    predictor_series_keys = list(predictor_series_keys)
    extra_agg_keys = set(agg_spec) - set(predictor_series_keys)
    if extra_agg_keys:
        raise ValueError(f"agg_spec contains non-predictor keys: {sorted(extra_agg_keys)}")
    if target_series_key in predictor_series_keys:
        raise ValueError("target_series_key must not be in predictor_series_keys")
    allowed_methods = {"mean", "sum", "last"}
    invalid_methods = {
        key: method for key, method in agg_spec.items() if method.lower() not in allowed_methods
    }
    if invalid_methods:
        formatted = ", ".join(f"{key}={method}" for key, method in invalid_methods.items())
        raise ValueError(f"agg_spec contains invalid methods: {formatted}")
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
    current_quarter = pd.Period(str(infer_current_quarter(asof_date)), freq="Q")
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
        obs_dates_naive = to_utc_naive(obs_dates_utc)
        # Validate month-end for monthly predictors (when metadata explicitly specifies 'm').
        # Note: Predictors without metadata are not validated since we cannot determine frequency.
        if (
            series_key in predictor_key_set
            and meta is not None
            and str(meta.frequency).lower() == "m"
        ):
            validate_monthly_obs_dates(
                obs_dates_naive,
                series_key=series_key,
                obs_date_anchor=meta.obs_date_anchor,
            )
        series = pd.Series(data["value"].to_numpy(), index=pd.DatetimeIndex(obs_dates_naive))
        series = series.sort_index()
        series = apply_quarter_cutoff(
            series,
            asof_date=asof_date,
            include_partial_quarters=include_partial_quarters,
            current_quarter=current_quarter,
        )
        raw_by_key[series_key] = series

    quarter_index = pd.Index(
        sorted({to_quarter_period(ts) for series in raw_by_key.values() for ts in series.index}),
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
        meta = metadata_by_key.get(series_key)
        frequency = str(meta.frequency).lower() if meta is not None else ""

        if series.empty:
            if frequency in {"d", "b"}:
                empty_daily = expand_daily_series_to_frame(
                    series,
                    series_key=series_key,
                    quarter_index=quarter_index,
                )
                predictor_frame = predictor_frame.join(empty_daily)
            else:
                predictor_frame[series_key] = np.nan
            nobs_current[series_key] = 0
            last_obs_date_current_quarter[series_key] = None
            continue

        series = series.sort_index()

        if frequency in {"d", "b"}:
            daily_frame = expand_daily_series_to_frame(
                series,
                series_key=series_key,
                quarter_index=quarter_index,
            )
            predictor_frame = predictor_frame.join(daily_frame)
        else:
            grouped = series.groupby(series.index.map(to_quarter_period))
            agg_method = agg_spec.get(series_key, "mean").lower()
            predictor_frame[series_key] = grouped.apply(
                lambda values: agg_series(values, agg_method)
            )
            predictor_frame[series_key] = predictor_frame[series_key].reindex(quarter_index)

        current_series = series.loc[series.index.map(to_quarter_period) == current_quarter]
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
    dataset["y_asof_latest_level"] = target_quarters
    dataset["y_asof_latest_growth"] = compute_gdp_qoq_saar(dataset["y_asof_latest_level"])
    dataset["ref_quarter_end"] = pd.PeriodIndex(dataset.index, freq="Q").end_time
    dataset["asof_date"] = pd.to_datetime(asof_date)
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

    def fit_predict_one(self, asof_date: date) -> dict:
        """Fit the model for a single vintage date and return diagnostics.

        Note: For offline learning with label="y_final", this method trains only
        on historical quarters within a single vintage. For proper walk-forward
        backtesting that trains on multiple historical vintages, use
        `run_backtest()` from `nowcast_data.models.backtest` instead.

        Supports two label modes:
        - "y_asof_latest": Uses latest available target value (online learning).
        - "y_final": Uses final target value from evaluation_asof_date (offline learning).
          Only works if evaluation_asof_date is provided in config.
        """
        current_ref = infer_current_quarter(asof_date)
        current_quarter = pd.Period(str(current_ref), freq="Q")

        if self.config.training_label_mode == "revision" and self.config.label != "y_final":
            raise ValueError("training_label_mode='revision' requires config.label='y_final'")

        if self.config.include_y_asof_latest_as_feature:
            import warnings

            warnings.warn(
                "include_y_asof_latest_as_feature is deprecated; "
                "use use_real_time_target_as_feature/real_time_feature_cols instead.",
                DeprecationWarning,
            )
            if not self.config.real_time_feature_cols:
                self.config.real_time_feature_cols = [
                    "y_asof_latest_growth",
                    "y_asof_latest_level",
                ]
            self.config.use_real_time_target_as_feature = True

        if self.config.label == "y_asof_latest":
            # Online label: use build_rt_quarterly_dataset with latest values
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
            label_column = "y"
            label_used = "y_asof_latest"
        elif self.config.label == "y_final":
            # Offline label: use vintage training dataset with final values
            if self.config.evaluation_asof_date is None:
                raise ValueError(
                    "label='y_final' requires evaluation_asof_date to be set in BridgeConfig"
                )
            from nowcast_data.models.datasets import (
                build_vintage_training_dataset,
                VintageTrainingDatasetConfig,
            )

            ref_offsets = list(range(-self.config.min_train_quarters, 1))
            vintage_config = VintageTrainingDatasetConfig(
                target_series_key=self.config.target_series_key,
                predictor_series_keys=self.config.predictor_series_keys,
                agg_spec=self.config.agg_spec,
                include_partial_quarters=self.config.include_partial_quarters,
                ref_offsets=ref_offsets,
                evaluation_asof_date=self.config.evaluation_asof_date,
                final_target_policy=self.config.final_target_policy,
                target_feature_spec=(
                    self.config.target_feature_spec
                    if self.config.include_target_release_features
                    else None
                ),
            )
            dataset, vintage_meta = build_vintage_training_dataset(
                self.adapter,
                self.catalog,
                config=vintage_config,
                asof_date=asof_date,
                ingest_from_ctx_source=self.config.ingest_from_ctx_source,
            )
            nobs_current = vintage_meta.get("nobs_current", {})
            last_obs_date_current_quarter = vintage_meta.get("last_obs_date_current_quarter", {})
            label_column = "y_final"
            label_used = "y_final"
        else:
            raise ValueError(f"label must be 'y_asof_latest' or 'y_final', got {self.config.label}")

        if dataset.empty or current_quarter not in dataset.index:
            return {
                "asof_date": asof_date,
                "ref_quarter": str(current_ref),
                "y_true_asof": np.nan,
                "y_true_final": np.nan,
                "y_true_stable": np.nan,
                "y_true_real_time": np.nan,
                "y_pred": np.nan,
                "y_pred_stable": np.nan,
                "y_pred_revision": np.nan,
                "label_used": label_used,
                "training_label_mode": self.config.training_label_mode,
                "uses_real_time_as_feature": self.config.use_real_time_target_as_feature,
                "real_time_feature_cols": self.config.real_time_feature_cols,
                "stable_label_col": self.config.stable_label_col,
                "real_time_label_col": self.config.real_time_label_col,
                "stable_pred_reconstruction": (
                    "real_time_plus_revision"
                    if self.config.training_label_mode == "revision"
                    else "direct"
                ),
                "feature_cols": [],
                "n_train": 0,
                "n_features": 0,
                "alpha_selected": np.nan,
                "mean_months_observed": np.nan,
                "last_obs_date_current_quarter": {},
                "nobs_current": {},
            }

        if self.config.label == "y_final":
            dataset = dataset.copy()
            if self.config.stable_label_col not in dataset.columns:
                if "y_stable_growth" in dataset.columns:
                    dataset[self.config.stable_label_col] = dataset["y_stable_growth"]
                elif "y_final_3rd_growth" in dataset.columns:
                    dataset[self.config.stable_label_col] = dataset["y_final_3rd_growth"]
                elif "y_final" in dataset.columns:
                    dataset[self.config.stable_label_col] = dataset["y_final"]
            if (
                self.config.real_time_label_col not in dataset.columns
                and "y_asof_latest" in dataset.columns
            ):
                dataset[self.config.real_time_label_col] = dataset["y_asof_latest"]

        time_cols = [col for col in ["asof_date", "ref_quarter_end"] if col in dataset.columns]
        base_cols_to_drop = [
            col
            for col in [
                "y",
                "y_asof_latest",
                "y_final",
                "y_final_3rd_level",
                self.config.stable_label_col,
            ]
            if col in dataset.columns
        ]

        pre_features = dataset.drop(columns=base_cols_to_drop)
        pre_features = pre_features.reset_index(drop=False)
        ref_quarter_col = pre_features["ref_quarter"]
        feature_input = pre_features.drop(columns=["ref_quarter"], errors="ignore")
        feature_builder = QuarterlyFeatureBuilder(
            predictor_keys=self.config.predictor_series_keys,
            time_col="ref_quarter_end",
            group_col="asof_date",
        )
        engineered = feature_builder.transform(feature_input)
        predictors = pd.concat([ref_quarter_col, engineered], axis=1).set_index("ref_quarter")
        predictors = predictors.drop(columns=time_cols, errors="ignore")

        if (
            not self.config.use_real_time_target_as_feature
            and self.config.real_time_feature_cols
        ):
            predictors = predictors.drop(columns=self.config.real_time_feature_cols, errors="ignore")

        if self.config.label == "y_asof_latest":
            target = dataset[label_column]
        else:
            stable_col = (
                self.config.stable_label_col
                if self.config.stable_label_col in dataset.columns
                else label_column
            )
            real_time_col = (
                self.config.real_time_label_col
                if self.config.real_time_label_col in dataset.columns
                else "y_asof_latest"
            )
            if self.config.training_label_mode == "revision":
                target = dataset[stable_col] - dataset[real_time_col]
            else:
                target = dataset[stable_col]

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

        train_y_clean = train_y.dropna()
        train_X_clean = train_X.loc[train_y_clean.index]

        if len(train_y_clean) < self.config.min_train_quarters or train_X_clean.empty:
            return {
                "asof_date": asof_date,
                "ref_quarter": str(current_ref),
                "y_true_asof": np.nan,
                "y_true_final": np.nan,
                "y_pred": np.nan,
                "y_pred_stable": np.nan,
                "y_pred_revision": np.nan,
                "label_used": label_used,
                "training_label_mode": self.config.training_label_mode,
                "uses_real_time_as_feature": self.config.use_real_time_target_as_feature,
                "real_time_feature_cols": self.config.real_time_feature_cols,
                "stable_label_col": self.config.stable_label_col,
                "real_time_label_col": self.config.real_time_label_col,
                "stable_pred_reconstruction": (
                    "real_time_plus_revision"
                    if self.config.training_label_mode == "revision"
                    else "direct"
                ),
                "feature_cols": train_X_clean.columns.tolist(),
                "n_train": len(train_y_clean),
                "n_features": train_X_clean.shape[1],
                "alpha_selected": np.nan,
                "mean_months_observed": (
                    float(np.mean(list(nobs_current.values()))) if nobs_current else np.nan
                ),
                "last_obs_date_current_quarter": last_obs_date_current_quarter,
                "nobs_current": nobs_current,
            }

        if self.config.model == "ridge":
            model = RidgeCV(alphas=self.config.alphas)
            model.fit(train_X_clean.to_numpy(), train_y_clean.to_numpy())
            y_pred_raw = float(model.predict(current_X.to_numpy().reshape(1, -1))[0])
            alpha_selected = float(model.alpha_)
        elif self.config.model == "ols":
            model = LinearRegression(fit_intercept=True)
            model.fit(train_X_clean.to_numpy(), train_y_clean.to_numpy())
            y_pred_raw = float(model.predict(current_X.to_numpy().reshape(1, -1))[0])
            alpha_selected = np.nan
        else:
            raise ValueError("model must be 'ridge' or 'ols'")

        y_pred_stable = y_pred_raw
        y_pred_revision = np.nan
        if self.config.training_label_mode == "revision":
            y_pred_revision = y_pred_raw
            if self.config.real_time_label_col in dataset.columns:
                rt_val = dataset.loc[current_quarter, self.config.real_time_label_col]
                y_pred_stable = float(rt_val) + y_pred_revision if pd.notna(rt_val) else np.nan
            else:
                y_pred_stable = np.nan

        # Get ground truth values
        y_true_asof = np.nan
        y_true_final = np.nan
        y_true_stable = np.nan
        y_true_real_time = np.nan

        if self.config.label == "y_asof_latest":
            y_true_asof = (
                dataset.loc[current_quarter, "y"] if current_quarter in dataset.index else np.nan
            )
        else:
            # For offline label, try to get both values if available
            if "y_asof_latest" in dataset.columns and current_quarter in dataset.index:
                y_true_asof = dataset.loc[current_quarter, "y_asof_latest"]
            if "y_final" in dataset.columns and current_quarter in dataset.index:
                y_true_final = dataset.loc[current_quarter, "y_final"]

        if self.config.stable_label_col in dataset.columns and current_quarter in dataset.index:
            y_true_stable = dataset.loc[current_quarter, self.config.stable_label_col]
        if self.config.real_time_label_col in dataset.columns and current_quarter in dataset.index:
            y_true_real_time = dataset.loc[current_quarter, self.config.real_time_label_col]

        return {
            "asof_date": asof_date,
            "ref_quarter": str(current_ref),
            "y_true_asof": y_true_asof,
            "y_true_final": y_true_final,
            "y_true_stable": y_true_stable,
            "y_true_real_time": y_true_real_time,
            "y_pred": y_pred_raw,
            "y_pred_stable": y_pred_stable,
            "y_pred_revision": y_pred_revision,
            "label_used": label_used,
            "training_label_mode": self.config.training_label_mode,
            "uses_real_time_as_feature": self.config.use_real_time_target_as_feature,
            "real_time_feature_cols": self.config.real_time_feature_cols,
            "stable_label_col": self.config.stable_label_col,
            "real_time_label_col": self.config.real_time_label_col,
            "stable_pred_reconstruction": (
                "real_time_plus_revision"
                if self.config.training_label_mode == "revision"
                else "direct"
            ),
            "feature_cols": train_X_clean.columns.tolist(),
            "n_train": len(train_y_clean),
            "n_features": train_X_clean.shape[1],
            "alpha_selected": alpha_selected,
            "mean_months_observed": (
                float(np.mean(list(nobs_current.values()))) if nobs_current else np.nan
            ),
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

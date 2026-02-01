from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from nowcast_data.features.recipes import get_recipe


class MissingnessFilter(BaseEstimator, TransformerMixin):
    def __init__(self, max_nan_frac: float = 0.5) -> None:
        self.max_nan_frac = max_nan_frac
        # sklearn convention: fitted attributes end with trailing underscore
        self.keep_cols_: list[str] | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray | None = None) -> "MissingnessFilter":
        nan_frac = X.isna().mean()
        self.keep_cols_ = nan_frac[nan_frac <= self.max_nan_frac].index.tolist()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.keep_cols_ is None:
            raise ValueError(
                "This MissingnessFilter instance is not fitted yet. Call fit() before calling "
                "transform()."
            )
        return X[self.keep_cols_]


class QuarterlyFeatureBuilder(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        predictor_keys: list[str],
        *,
        time_col: str = "ref_quarter_end",
        group_col: str = "asof_date",
        add_availability_flags: bool = True,
    ) -> None:
        self.predictor_keys = predictor_keys
        self.time_col = time_col
        self.group_col = group_col
        self.add_availability_flags = add_availability_flags

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray | None = None) -> "QuarterlyFeatureBuilder":
        return self

    def _compute_changes(
        self, series: pd.Series, change: str
    ) -> tuple[pd.Series, pd.Series]:
        if change == "diff":
            return series.diff(1), series.diff(4)
        if change == "pct":
            # Percent change expressed in percentage points.
            return series.pct_change(1) * 100.0, series.pct_change(4) * 100.0
        if change == "logdiff":
            logged = np.log(series)
            # logdiff uses percentage-point log differences (not annualized).
            return logged.diff(1) * 100.0, logged.diff(4) * 100.0
        if change == "logdiff_saar":
            logged = np.log(series)
            # QoQ is annualized (x400); YoY uses 100 for percentage-point change over 4 quarters.
            return logged.diff(1) * 400.0, logged.diff(4) * 100.0
        # Default to simple differences for unknown change spec.
        return series.diff(1), series.diff(4)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        missing = [col for col in [self.time_col, self.group_col] if col not in X.columns]
        if missing:
            raise ValueError(
                "QuarterlyFeatureBuilder requires time and group columns; missing: "
                + ", ".join(missing)
            )

        feature_blocks: list[pd.DataFrame] = []
        for _, group in X.groupby(self.group_col):
            original_index = group.index
            sorted_group = group.sort_values(self.time_col)
            features = pd.DataFrame(index=sorted_group.index)

            for series_key in self.predictor_keys:
                if series_key not in sorted_group.columns:
                    continue
                recipe = get_recipe(series_key)
                series = sorted_group[series_key]

                if recipe.level:
                    features[f"{series_key}__level"] = series

                qoq, yoy = self._compute_changes(series, recipe.change)
                if recipe.qoq:
                    features[f"{series_key}__qoq"] = qoq
                if recipe.yoy:
                    features[f"{series_key}__yoy"] = yoy
                if self.add_availability_flags:
                    features[f"{series_key}__isna"] = series.isna().astype(int)

            # Reindex back to original row order for this group.
            features = features.reindex(original_index)
            feature_blocks.append(features)

        if not feature_blocks:
            return pd.DataFrame(index=X.index)
        return pd.concat(feature_blocks, axis=0).reindex(X.index)


def compute_gdp_qoq_saar(level: pd.Series) -> pd.Series:
    return np.log(level).diff(1) * 400.0

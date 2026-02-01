"""Event-vintage walk-forward GDP backtest (revision-aware, PIT-only by default).

Example:
  python examples/build_vintage_training_dataset.py \
    --meta-csv data/meta_data.csv \
    --series-catalog series_catalog.yaml \
    --target gdpc1 \
    --start-date 2000-01-01 \
    --end-date 2025-12-31 \
    --train-end-date 2012-12-31 \
    --score-start-date 2013-01-01 \
    --score-end-date 2025-12-31 \
    --evaluation-asof-date 2025-12-31 \
    --ref-offsets=-1,0,1 \
    --out-dir outputs/gdp_walkforward_event

Notes:
  level_anchor_policy=real_time_only anchors implied level predictions using the
  prior-quarter real-time GDP level (no stable-truth fallback).
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from nowcast_data.features import (
    MissingnessFilter,
    QuarterlyFeatureBuilder,
    build_agg_spec_from_recipes,
    compute_gdp_qoq_saar,
)
from nowcast_data.models.datasets import (
    VintageTrainingDatasetConfig,
    build_vintage_training_dataset,
)
from nowcast_data.models.target_policy import (
    TargetPolicy,
    quarter_end_date,
    resolve_target_from_releases,
)
from nowcast_data.pit.api import PITDataManager
from nowcast_data.pit.core.catalog import SeriesCatalog
from nowcast_data.pit.core.models import SeriesMetadata


def _compute_metrics(df: pd.DataFrame, *, pred_col: str, truth_col: str) -> dict:
    """Compute RMSE/MAE and ref-offset breakdown for prediction/truth columns."""
    use = df[df[pred_col].notna() & df[truth_col].notna()].copy()
    if use.empty:
        return {"rmse": np.nan, "mae": np.nan, "count": 0, "by_ref_offset": {}}

    err = use[pred_col] - use[truth_col]
    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))

    by_ref_offset: dict[str, dict] = {}
    for ref_offset, g in use.groupby("ref_offset"):
        e = g[pred_col] - g[truth_col]
        by_ref_offset[str(ref_offset)] = {
            "rmse": float(np.sqrt(np.mean(e**2))) if len(g) else np.nan,
            "mae": float(np.mean(np.abs(e))) if len(g) else np.nan,
            "count": int(len(g)),
        }

    return {"rmse": rmse, "mae": mae, "count": int(len(use)), "by_ref_offset": by_ref_offset}


def _parse_date(value: str) -> date:
    return pd.Timestamp(value).date()


def _parse_offsets(value: str) -> list[int]:
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if not parts:
        raise ValueError("--ref-offsets cannot be empty")
    return [int(p) for p in parts]


def _load_series_catalog(series_catalog_path: Path, meta_csv: Path) -> SeriesCatalog:
    catalog = SeriesCatalog(series_catalog_path)

    df = pd.read_csv(meta_csv)
    df.columns = [c.strip().lower() for c in df.columns]
    if "series" not in df.columns:
        raise ValueError("meta_csv must contain a 'series' column")

    for _, row in df.iterrows():
        series = str(row.get("series", "")).strip()
        if not series:
            continue
        series_key = series.upper()
        if catalog.get(series_key) is not None:
            continue
        freq = str(row.get("freq", "m")).strip().lower() or "m"
        catalog.add(
            SeriesMetadata(
                series_key=series_key,
                country="US",
                source="alfred",
                source_series_id=series.upper(),
                frequency=freq,
                pit_mode="ALFRED_REALTIME",
                adapter="alphaforge",
                obs_date_anchor=None,
                description=str(row.get("name", "")) or None,
            )
        )
    return catalog


def resolve_series_key(catalog: SeriesCatalog, key_or_alias: str, meta_series: set[str]) -> str:
    key = key_or_alias.strip().lower()
    if key in meta_series:
        return key

    meta = catalog.get(key)
    if meta is not None:
        candidate = (meta.source_series_id or "").strip().lower()
        if candidate in meta_series:
            return candidate

    fallback_aliases = {
        "us_gdp_saar": "gdpc1",
        "gdp": "gdpc1",
    }
    mapped = fallback_aliases.get(key)
    if mapped and mapped in meta_series:
        return mapped

    raise ValueError(f"Unknown series key or alias: {key_or_alias}")


def collect_event_vintages(
    manager: PITDataManager,
    series_keys: Iterable[str],
    *,
    asof_start: date,
    asof_end: date,
) -> list[date]:
    if "alphaforge" not in manager.adapters:
        raise ValueError("alphaforge adapter is required to list vintages")

    adapter = manager.adapters["alphaforge"]
    event_dates: set[date] = set()

    for series_key in series_keys:
        try:
            vintages = adapter.list_vintages(series_key)
        except Exception as exc:
            print(f"Warning: failed to list vintages for {series_key}: {exc}")
            continue
        for v in vintages:
            ts = pd.Timestamp(v)
            if ts.tzinfo is not None:
                ts = ts.tz_convert("UTC").tz_localize(None)
            v_date = ts.date()
            if asof_start <= v_date <= asof_end:
                event_dates.add(v_date)

    return sorted(event_dates)


def _collect_meta_series(meta_csv: Path) -> list[str]:
    df = pd.read_csv(meta_csv)
    df.columns = [c.strip().lower() for c in df.columns]
    if "series" not in df.columns:
        raise ValueError("meta_csv must contain a 'series' column")
    series = [str(s).strip().lower() for s in df["series"].tolist() if str(s).strip()]
    return series


def _series_key_map(series_keys: list[str]) -> dict[str, str]:
    return {key: key.upper() for key in series_keys}


def _collect_meta_table(meta_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(meta_csv)
    df.columns = [c.strip().lower() for c in df.columns]
    if "series" not in df.columns:
        raise ValueError("meta_csv must contain a 'series' column")
    df["series"] = df["series"].astype(str).str.strip().str.lower()
    return df


def make_pipeline(model: str, predictor_keys: list[str], alphas: list[float]) -> Pipeline:
    steps = [
        ("feat", QuarterlyFeatureBuilder(predictor_keys=predictor_keys)),
        ("miss", MissingnessFilter(max_nan_frac=0.5)),
        ("impute", SimpleImputer(strategy="mean")),
        ("scale", StandardScaler(with_mean=True, with_std=True)),
    ]
    if model == "ridge":
        steps.append(("model", RidgeCV(alphas=alphas)))
    elif model == "ols":
        steps.append(("model", LinearRegression(fit_intercept=True)))
    else:
        raise ValueError(f"Unknown model: {model}")
    return Pipeline(steps)


def describe_pipeline(pipe: Pipeline) -> str:
    return "->".join(name for name, _ in pipe.steps)


def build_base_cols(predictor_keys: list[str]) -> list[str]:
    return [*predictor_keys, "asof_date", "ref_quarter_end"]


def build_model_matrices(
    *,
    history_offset: pd.DataFrame,
    test_row: pd.Series,
    predictor_keys: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Return (X_train, X_test, base_cols) with a stable schema.

    history_offset supplies the training panel for a single ref_offset, test_row is the
    target vintage row, and predictor_keys defines the expected raw predictors. Missing
    columns are added as NaN to keep a consistent schema.
    """
    base_cols = build_base_cols(predictor_keys)
    X_train = history_offset[[c for c in base_cols if c in history_offset.columns]].copy()
    X_test = test_row[[c for c in base_cols if c in test_row.index]].to_frame().T.copy()
    for col in base_cols:
        if col not in X_train.columns:
            X_train[col] = np.nan
        if col not in X_test.columns:
            X_test[col] = np.nan
    X_train = X_train[base_cols]
    X_test = X_test[base_cols]
    return X_train, X_test, base_cols


def _quarter_end_for_ref(ref_quarter: str) -> date:
    period = pd.Period(ref_quarter, freq="Q")
    return quarter_end_date(period)


def _prev_ref_quarter(ref_quarter: str) -> str:
    """Return the previous quarter string for a YYYYQq input."""
    return (pd.Period(ref_quarter, freq="Q") - 1).strftime("%YQ%q")


def implied_level_from_growth(
    panel_vintage: pd.DataFrame,
    ref_quarter: str,
    y_pred_growth: float,
) -> float | None:
    """Convert QoQ SAAR growth prediction to implied level using real-time anchor only."""
    prev_q = _prev_ref_quarter(ref_quarter)
    prev = panel_vintage[panel_vintage["ref_quarter"] == prev_q]
    if prev.empty:
        return None
    prev_row = prev.iloc[0]
    anchor_level = prev_row["y_asof_latest_level"]
    if pd.isna(anchor_level):
        return None
    return float(anchor_level * np.exp(y_pred_growth / 400.0))


def _list_target_releases(
    adapter,
    *,
    series_key: str,
    ref_quarter: str,
    asof_date: date,
) -> pd.DataFrame:
    obs_end = _quarter_end_for_ref(ref_quarter)
    releases = adapter.list_pit_observations_asof(
        series_key=series_key,
        obs_date=obs_end,
        asof_date=asof_date,
    )
    if not releases.empty:
        return releases

    obs_start = pd.Period(ref_quarter, freq="Q").start_time.date()
    releases_start = adapter.list_pit_observations_asof(
        series_key=series_key,
        obs_date=obs_start,
        asof_date=asof_date,
    )
    return releases_start


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ingest-from-ctx-source", action="store_true")
    parser.add_argument("--start-date", default="2000-01-01")
    parser.add_argument("--end-date", default="2025-12-31")
    parser.add_argument("--train-end-date", default="2012-12-31")
    parser.add_argument("--score-start-date", default="2013-01-01")
    parser.add_argument("--score-end-date", default="2025-12-31")
    parser.add_argument("--evaluation-asof-date", default="2025-12-31")
    parser.add_argument("--target", default="gdpc1")
    parser.add_argument("--meta-csv", default="data/meta_data.csv")
    parser.add_argument("--series-catalog", default="series_catalog.yaml")
    parser.add_argument("--out-dir", default="outputs/gdp_walkforward_event")
    parser.add_argument("--ref-offsets", default="-1,0,1")
    parser.add_argument("--drop-predictors", default="")
    parser.add_argument("--vintage-grid", default="event", choices=["event"])
    parser.add_argument("--model", default="ridge", choices=["ridge", "ols"])
    parser.add_argument("--alphas", default="0.01,0.1,1,10,100")
    args = parser.parse_args()

    start_date = _parse_date(args.start_date)
    end_date = _parse_date(args.end_date)
    train_end_date = _parse_date(args.train_end_date)
    score_start_date = _parse_date(args.score_start_date)
    score_end_date = _parse_date(args.score_end_date)
    evaluation_asof_date = _parse_date(args.evaluation_asof_date)
    ref_offsets = _parse_offsets(args.ref_offsets)

    meta_csv = Path(args.meta_csv)
    series_catalog_path = Path(args.series_catalog)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_df = _collect_meta_table(meta_csv)
    series_keys = [str(s).strip().lower() for s in meta_df["series"].tolist() if str(s).strip()]
    series_set = set(series_keys)
    series_key_map = _series_key_map(series_keys)
    pit_to_canonical = {v: k for k, v in series_key_map.items()}

    catalog = _load_series_catalog(series_catalog_path, meta_csv)
    target_canonical = resolve_series_key(catalog, args.target, series_set)
    target_pit = series_key_map[target_canonical]

    drop_predictors = {k.strip().lower() for k in args.drop_predictors.split(",") if k.strip()}
    predictor_keys = [
        key for key in series_keys if key != target_canonical and key not in drop_predictors
    ]
    if not predictor_keys:
        raise ValueError("Predictor list is empty after applying --drop-predictors")

    manager = PITDataManager(catalog)
    if "alphaforge" not in manager.adapters:
        raise ValueError("alphaforge adapter is required for PIT access")

    predictor_pit_keys = [series_key_map[key] for key in predictor_keys]
    agg_spec = build_agg_spec_from_recipes(predictor_pit_keys, pit_to_canonical)

    final_target_policy = TargetPolicy(mode="nth_release", nth=3, max_release_rank=3)
    config = VintageTrainingDatasetConfig(
        target_series_key=target_pit,
        predictor_series_keys=predictor_pit_keys,
        agg_spec=agg_spec,
        include_partial_quarters=True,
        ref_offsets=ref_offsets,
        evaluation_asof_date=evaluation_asof_date,
        final_target_policy=final_target_policy,
        target_feature_spec=None,
    )

    if args.vintage_grid != "event":
        raise ValueError("Only event vintage grid is supported in this script")

    event_vintages = collect_event_vintages(
        manager,
        [target_pit, *predictor_pit_keys],
        asof_start=start_date,
        asof_end=end_date,
    )

    if not event_vintages:
        raise ValueError("No event vintages found in the requested window")

    print("=" * 80)
    print("EVENT WALK-FORWARD BACKTEST")
    print("=" * 80)
    print(f"Target input: {args.target}")
    print(f"Target canonical: {target_canonical}")
    print(f"Target PIT key: {target_pit}")
    print(f"Vintage grid: event ({len(event_vintages)} dates)")
    print(f"As-of window: {start_date} to {end_date}")
    print(f"Train end date: {train_end_date}")
    print(f"Score window: {score_start_date} to {score_end_date}")
    print(f"Evaluation cutoff: {evaluation_asof_date}")
    print(f"Ref offsets: {ref_offsets}")
    print(f"Predictor count: {len(predictor_keys)}")
    print(f"Ingest from ctx source: {bool(args.ingest_from_ctx_source)}")

    adapter = manager.adapters["alphaforge"]
    latest_policy = TargetPolicy(mode="latest_available")
    rows: list[dict] = []

    for asof_date in event_vintages:
        try:
            dataset, meta = build_vintage_training_dataset(
                adapter,
                catalog,
                config=config,
                asof_date=asof_date,
                ingest_from_ctx_source=bool(args.ingest_from_ctx_source),
            )
        except Exception as exc:
            print(f"Warning: failed vintage {asof_date}: {exc}")
            continue

        if dataset.empty:
            continue

        for ref_quarter, row in dataset.iterrows():
            ref_offset = (
                pd.Period(ref_quarter, freq="Q").ordinal
                - pd.Period(meta["current_ref_quarter"], freq="Q").ordinal
            )
            row_dict = {
                pit_to_canonical.get(key, key): value for key, value in row.to_dict().items()
            }
            row_dict.update(
                {
                    "asof_date": asof_date,
                    "ref_quarter": str(ref_quarter),
                    "ref_offset": ref_offset,
                }
            )

            releases_asof = _list_target_releases(
                adapter,
                series_key=target_pit,
                ref_quarter=str(ref_quarter),
                asof_date=asof_date,
            )
            y_asof_value, y_asof_meta = resolve_target_from_releases(releases_asof, latest_policy)
            releases_final = _list_target_releases(
                adapter,
                series_key=target_pit,
                ref_quarter=str(ref_quarter),
                asof_date=evaluation_asof_date,
            )
            y_final_value, y_final_meta = resolve_target_from_releases(
                releases_final,
                final_target_policy,
            )

            row_dict["y_asof_latest_level"] = y_asof_value if y_asof_value is not None else np.nan
            row_dict["y_final_3rd_level"] = y_final_value if y_final_value is not None else np.nan
            row_dict["y_asof_latest_release_rank"] = y_asof_meta.get("selected_release_rank")
            row_dict["y_asof_latest_selected_asof_utc"] = y_asof_meta.get(
                "selected_release_asof_utc"
            )
            row_dict["y_final_release_rank"] = y_final_meta.get("selected_release_rank")
            row_dict["y_final_selected_asof_utc"] = y_final_meta.get("selected_release_asof_utc")
            row_dict["y_final_policy_mode"] = y_final_meta.get("policy_mode")

            rows.append(row_dict)

    if not rows:
        raise RuntimeError("No dataset rows constructed from event vintages")

    panel = pd.DataFrame(rows)
    panel["ref_quarter_end"] = panel["ref_quarter"].map(_quarter_end_for_ref)

    def _add_asof_latest_growth(group: pd.DataFrame) -> pd.DataFrame:
        group = group.sort_values("ref_quarter_end").copy()
        group["y_asof_latest_growth"] = compute_gdp_qoq_saar(group["y_asof_latest_level"])
        return group

    panel = panel.groupby("asof_date", group_keys=False).apply(_add_asof_latest_growth)
    truth_candidates = panel[["ref_quarter", "asof_date", "y_final_3rd_level"]].dropna(
        subset=["y_final_3rd_level"]
    )
    if truth_candidates.empty:
        truth = pd.DataFrame(columns=["ref_quarter", "y_final_3rd_growth"])
    else:
        grouped = truth_candidates.groupby("ref_quarter")["y_final_3rd_level"]
        spread = grouped.max() - grouped.min()
        scale = grouped.mean().abs()
        tolerance = 1e-8 * scale + 1e-10
        bad = spread > tolerance
        if bad.any():
            bad_quarters = bad[bad].index.tolist()[:3]
            summary = (
                truth_candidates[truth_candidates["ref_quarter"].isin(bad_quarters)]
                .groupby("ref_quarter")["y_final_3rd_level"]
                .agg(min="min", max="max", mean="mean")
            )
            summary["spread"] = summary["max"] - summary["min"]
            examples = (
                truth_candidates[truth_candidates["ref_quarter"].isin(bad_quarters)]
                .sort_values(["ref_quarter", "asof_date"])
                .groupby("ref_quarter")
                .head(3)
                .to_string(index=False)
            )
            raise ValueError(
                "Inconsistent y_final_3rd_level across vintages for ref_quarter(s): "
                f"{bad_quarters}. Summary:\n{summary}\nExamples:\n{examples}"
            )

        sorted_candidates = truth_candidates.rename(
            columns={"asof_date": "first_release_asof_date"}
        ).sort_values(["ref_quarter", "first_release_asof_date"])
        truth = sorted_candidates.groupby("ref_quarter", as_index=False).first()
        truth["ref_quarter_end"] = truth["ref_quarter"].map(_quarter_end_for_ref)
        truth = truth.sort_values("ref_quarter_end")
        truth["y_final_3rd_growth"] = compute_gdp_qoq_saar(truth["y_final_3rd_level"])
    panel = panel.merge(
        truth[["ref_quarter", "y_final_3rd_growth"]],
        on="ref_quarter",
        how="left",
        validate="m:1",
    )

    alphas = [float(x.strip()) for x in args.alphas.split(",") if x.strip()]

    pipeline_cache: dict[tuple[Any, ...], Pipeline] = {}
    predictions: list[dict] = []
    for asof_date in sorted(panel["asof_date"].unique()):
        history = panel[panel["asof_date"] < asof_date]
        if history.empty:
            continue
        panel_vintage = panel[panel["asof_date"] == asof_date]

        for ref_offset in ref_offsets:
            history_offset = history[history["ref_offset"] == ref_offset]
            history_offset = history_offset[history_offset["ref_quarter_end"] <= train_end_date]
            history_offset = history_offset[history_offset["y_final_3rd_growth"].notna()]
            available_predictors = [c for c in predictor_keys if c in history_offset.columns]
            if available_predictors:
                n_raw_predictors_present_train = int(
                    history_offset[available_predictors].notna().any(axis=0).sum()
                )
            else:
                n_raw_predictors_present_train = 0

            test_rows = panel[
                (panel["asof_date"] == asof_date) & (panel["ref_offset"] == ref_offset)
            ]
            if len(test_rows) != 1:
                raise ValueError(
                    "Expected exactly one test row for "
                    f"asof_date={asof_date!r} ref_offset={ref_offset!r}; "
                    f"found {len(test_rows)}. Check panel rows for this asof_date/offset."
                )
            test_row = test_rows.iloc[0]

            X_train, X_test, base_cols = build_model_matrices(
                history_offset=history_offset,
                test_row=test_row,
                predictor_keys=predictor_keys,
            )
            if not X_train.index.equals(history_offset.index):
                raise ValueError(
                    "X_train index mismatch with history_offset "
                    f"(train={len(X_train.index)} history={len(history_offset.index)})"
                )
            y_train = history_offset.loc[X_train.index, "y_final_3rd_growth"]
            if not X_train.index.equals(y_train.index):
                train_indices_missing_labels = list(X_train.index.difference(y_train.index)[:5])
                label_indices_missing_features = list(y_train.index.difference(X_train.index)[:5])
                raise ValueError(
                    "X_train index mismatch with y_train "
                    f"(train={len(X_train.index)} y_train={len(y_train.index)}). "
                    "Check for filtering or alignment issues in history_offset. "
                    f"train_indices_missing_labels[:5]={train_indices_missing_labels} "
                    f"label_indices_missing_features[:5]={label_indices_missing_features}"
                )

            # Pipeline structure is fully determined by model choice, alphas, and predictor list.
            pipe_key = (
                "pipeline_v1",
                args.model,
                tuple(alphas),
                tuple(predictor_keys),
            )
            if pipe_key not in pipeline_cache:
                pipeline_cache[pipe_key] = make_pipeline(args.model, predictor_keys, alphas)
            pipe = clone(pipeline_cache[pipe_key])
            pipe.fit(X_train, y_train)
            y_pred_growth = float(pipe.predict(X_test)[0])
            missing_filter = pipe.named_steps.get("miss")
            keep_cols = missing_filter.keep_cols_ if missing_filter is not None else None
            # None means no missing filter step in pipeline; 0 means all features dropped by it.
            n_features_after_missing_filter = int(len(keep_cols)) if keep_cols is not None else None
            if args.model == "ridge":
                alpha_selected = float(pipe.named_steps["model"].alpha_)
            else:
                alpha_selected = np.nan
            y_pred_level = implied_level_from_growth(
                panel_vintage,
                ref_quarter=str(test_row["ref_quarter"]),
                y_pred_growth=y_pred_growth,
            )

            predictions.append(
                {
                    "asof_date": asof_date,
                    "ref_quarter": test_row["ref_quarter"],
                    "ref_offset": ref_offset,
                    "ref_quarter_end": test_row["ref_quarter_end"],
                    "y_pred_growth": y_pred_growth,
                    "y_pred_level": y_pred_level,
                    "y_asof_latest_level": test_row.get("y_asof_latest_level"),
                    "y_asof_latest_growth": test_row.get("y_asof_latest_growth"),
                    "y_final_3rd_level": test_row.get("y_final_3rd_level"),
                    "y_final_3rd_growth": test_row.get("y_final_3rd_growth"),
                    "y_asof_latest_release_rank": test_row.get("y_asof_latest_release_rank"),
                    "y_asof_latest_selected_asof_utc": test_row.get(
                        "y_asof_latest_selected_asof_utc"
                    ),
                    "y_final_release_rank": test_row.get("y_final_release_rank"),
                    "y_final_selected_asof_utc": test_row.get("y_final_selected_asof_utc"),
                    "n_train": int(len(X_train)),
                    "n_raw_predictors_expected": int(len(predictor_keys)),
                    "n_raw_predictors_present_train": n_raw_predictors_present_train,
                    "n_raw_features_total": int(len(base_cols)),
                    # n_features_post_filter retained for backwards compatibility (deprecated,
                    # remove after 2026-12, use n_features_engineered_post_filter instead).
                    "n_features_post_filter": n_features_after_missing_filter,
                    "n_features_engineered_post_filter": n_features_after_missing_filter,
                    "alpha_selected": alpha_selected,
                    "feature_pipeline": describe_pipeline(pipe),
                }
            )

    pred_df = pd.DataFrame(predictions)
    pred_df = pred_df.sort_values(["asof_date", "ref_offset"])

    score_mask = (pred_df["ref_quarter_end"] >= score_start_date) & (
        pred_df["ref_quarter_end"] <= score_end_date
    )
    scored = pred_df.loc[score_mask].copy()

    pipeline_for_metadata = make_pipeline(args.model, predictor_keys, alphas)
    metadata_base_cols = build_base_cols(predictor_keys)
    metrics = {
        "real_time_growth_space": _compute_metrics(
            scored, pred_col="y_pred_growth", truth_col="y_asof_latest_growth"
        ),
        "stable_growth_space_3rd_release": _compute_metrics(
            scored, pred_col="y_pred_growth", truth_col="y_final_3rd_growth"
        ),
        "real_time_level_space": _compute_metrics(
            scored, pred_col="y_pred_level", truth_col="y_asof_latest_level"
        ),
        "stable_level_space_3rd_release": _compute_metrics(
            scored, pred_col="y_pred_level", truth_col="y_final_3rd_level"
        ),
    }

    run_metadata = {
        "target_input": args.target,
        "target_canonical": target_canonical,
        "target_pit_key": target_pit,
        "training_label": "y_final_3rd_growth",
        "feature_pipeline": describe_pipeline(pipeline_for_metadata),
        "recipe_registry_version": "recipes_v1",
        "level_anchor_policy": "real_time_only",
        "stable_truth_policy": "third_release_first_observed",
        "stable_truth_release": "nth_release_3",
        "stable_truth_first_observed_asof_col": "first_release_asof_date",
        "n_raw_predictors_expected": int(len(predictor_keys)),
        "feature_schema": {
            "base_cols": metadata_base_cols,
            "time_cols": ["asof_date", "ref_quarter_end"],
        },
        "start_date": str(start_date),
        "end_date": str(end_date),
        "train_end_date": str(train_end_date),
        "score_start_date": str(score_start_date),
        "score_end_date": str(score_end_date),
        "evaluation_asof_date": str(evaluation_asof_date),
        "ref_offsets": ref_offsets,
        "predictor_count": len(predictor_keys),
        "predictors": predictor_keys,
        "predictor_pit_keys": predictor_pit_keys,
        "drop_predictors": sorted(drop_predictors),
        "event_vintage_count": len(event_vintages),
        "ingest_from_ctx_source": bool(args.ingest_from_ctx_source),
        "final_target_policy": asdict(final_target_policy),
    }

    predictions_path = out_dir / "predictions.parquet"
    metrics_path = out_dir / "metrics.json"
    metadata_path = out_dir / "run_metadata.json"

    try:
        pred_df.to_parquet(predictions_path, index=False)
        print(f"Wrote predictions: {predictions_path}")
    except Exception as exc:
        fallback = out_dir / "predictions.csv"
        pred_df.to_csv(fallback, index=False)
        print(f"Warning: parquet write failed ({exc}); wrote CSV: {fallback}")

    metrics_path.write_text(json.dumps(metrics, indent=2))
    metadata_path.write_text(json.dumps(run_metadata, indent=2))

    print(f"Wrote metrics: {metrics_path}")
    print(f"Wrote metadata: {metadata_path}")


if __name__ == "__main__":
    main()

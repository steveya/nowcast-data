from __future__ import annotations

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
"""

import argparse
import json
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, RidgeCV

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


# -------------------- Transform helpers --------------------

def _collect_meta_table(meta_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(meta_csv)
    df.columns = [c.strip().lower() for c in df.columns]
    if "series" not in df.columns:
        raise ValueError("meta_csv must contain a 'series' column")
    df["series"] = df["series"].astype(str).str.strip().str.lower()
    return df


def _collect_transform_map(meta_csv: Path) -> dict[str, str]:
    """Return per-series transform string from meta_data.csv.

    The repo's meta_data.csv may include a transform column. We support a few common
    column names; if none are present, defaults to 'level' for all series.
    """
    df = _collect_meta_table(meta_csv)
    transform_cols = [
        c
        for c in [
            "transform",
            "transformation",
            "xform",
            "stationarity_transform",
            "stationary_transform",
        ]
        if c in df.columns
    ]
    if not transform_cols:
        return {s: "level" for s in df["series"].tolist() if s}
    col = transform_cols[0]
    out: dict[str, str] = {}
    for _, r in df.iterrows():
        s = str(r.get("series", "")).strip().lower()
        if not s:
            continue
        t = str(r.get(col, "level") or "level").strip().lower()
        out[s] = t if t else "level"
    return out


def _annualization_factor(freq: str) -> float:
    f = (freq or "").lower()
    if f == "q":
        return 400.0
    if f == "m":
        return 1200.0
    if f in {"w", "d", "b"}:
        # not well-defined; treat as monthly-ish default
        return 1200.0
    return 400.0


def _apply_transform_series(s: pd.Series, transform: str, *, freq: str) -> pd.Series:
    """Apply a simple stationarizing transform to a 1D time series.

    This is applied within each vintage (asof_date) across ref_quarter ordering.
    """
    t = (transform or "level").strip().lower()
    if t in {"", "none", "level", "levels"}:
        return s
    if t in {"log"}:
        return np.log(s)
    if t in {"diff", "delta"}:
        return s.diff(1)
    if t in {"log_diff", "dlog", "diff_log"}:
        return np.log(s).diff(1)
    if t in {"pct", "pct_change", "percent_change"}:
        return s.pct_change(1) * 100.0
    if t in {"yoy", "year_over_year"}:
        lag = 4 if (freq or "").lower() == "q" else 12
        return s.pct_change(lag) * 100.0
    if t in {"log_yoy", "dlog_yoy"}:
        lag = 4 if (freq or "").lower() == "q" else 12
        return np.log(s).diff(lag) * 100.0
    if t in {"qoq_saar", "logdiff_saar", "log_diff_saar", "log_diff_ann", "dlog_saar"}:
        # annualized log difference
        a = _annualization_factor(freq)
        return np.log(s).diff(1) * a
    # Unknown transform: fall back to level
    return s


def _transform_panel_inplace(
    panel: pd.DataFrame,
    *,
    meta_df: pd.DataFrame,
    transform_map: dict[str, str],
    catalog: SeriesCatalog,
    series_key_map: dict[str, str],
    predictor_keys: list[str],
    target_key: str,
) -> pd.DataFrame:
    """Create *_level columns and overwrite series columns with transformed values.

    The transform is applied per asof_date across ref_quarter_end ordering.
    Returns a DataFrame with added columns:
      - <series>_level for each predictor/target present
      - y_asof_latest_level, y_final_level
      - y_asof_latest_model, y_final_model
    """
    # Build a quick freq map from catalog (meta uses PIT keys upper).
    freq_map: dict[str, str] = {}
    for s in [*predictor_keys, target_key]:
        pit_key = series_key_map.get(s, s)
        meta = catalog.get(pit_key)
        if meta is not None:
            freq_map[s] = (meta.frequency or "").strip().lower() or "m"

    panel = panel.copy()

    # Preserve target level columns
    if "y_asof_latest" in panel.columns and "y_asof_latest_level" not in panel.columns:
        panel["y_asof_latest_level"] = panel["y_asof_latest"]
    if "y_final" in panel.columns and "y_final_level" not in panel.columns:
        panel["y_final_level"] = panel["y_final"]

    # Preserve predictor level columns
    for s in predictor_keys:
        if s in panel.columns and f"{s}_level" not in panel.columns:
            panel[f"{s}_level"] = panel[s]

    # Apply transforms within each vintage
    def _transform_group(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("ref_quarter_end").copy()

        # predictors
        for s in predictor_keys:
            if s not in g.columns:
                continue
            tr = transform_map.get(s, "level")
            freq = freq_map.get(s, "m")
            g[s] = _apply_transform_series(g[f"{s}_level"], tr, freq=freq)

        # target: build model-space columns from preserved level columns
        t_tr = transform_map.get(target_key, "level")
        t_freq = freq_map.get(target_key, "q")
        if "y_asof_latest_level" in g.columns:
            g["y_asof_latest_model"] = _apply_transform_series(
                g["y_asof_latest_level"], t_tr, freq=t_freq
            )
        if "y_final_level" in g.columns:
            g["y_final_model"] = _apply_transform_series(g["y_final_level"], t_tr, freq=t_freq)

        g["target_transform"] = t_tr
        return g

    panel = panel.groupby("asof_date", group_keys=False).apply(_transform_group)
    return panel


def _inverse_target_transform_one(
    *,
    y_pred_model: float | None,
    target_transform: str,
    ref_quarter: str,
    asof_date: date,
    panel: pd.DataFrame,
) -> float | None:
    """Attempt to invert the target transform to a level prediction for GDP.

    Only implemented for a few simple transforms. For annualized log-diff (qoq_saar),
    we use the previous quarter's y_final_level as the anchor.
    """
    if y_pred_model is None or (isinstance(y_pred_model, float) and np.isnan(y_pred_model)):
        return None
    t = (target_transform or "level").strip().lower()
    if t in {"", "none", "level", "levels"}:
        return float(y_pred_model)
    if t == "log":
        return float(np.exp(y_pred_model))
    if t in {"qoq_saar", "logdiff_saar", "log_diff_saar", "log_diff_ann", "dlog_saar"}:
        # y_pred_model is annualized log-diff: a*log(y_t/y_{t-1}) with a=400 for quarterly.
        a = 400.0
        prev_q = (pd.Period(ref_quarter, freq="Q") - 1).strftime("%YQ%q")
        prev = panel[(panel["asof_date"] == asof_date) & (panel["ref_quarter"] == prev_q)]
        if prev.empty:
            return None
        base = prev.iloc[0].get("y_final_level")
        if base is None or (isinstance(base, float) and np.isnan(base)):
            return None
        return float(base * np.exp(float(y_pred_model) / a))
    # other transforms not invertible without additional state
    return None


def _build_agg_spec(catalog: SeriesCatalog, predictor_keys: list[str]) -> dict[str, str]:
    agg_spec: dict[str, str] = {}
    for key in predictor_keys:
        meta = catalog.get(key)
        freq = (meta.frequency or "").lower() if meta else ""
        if freq in {"d", "b"}:
            agg_spec[key] = "last"
        elif freq in {"m", "q"}:
            agg_spec[key] = "mean"
        else:
            agg_spec[key] = "mean"
    return agg_spec


def _quarter_end_for_ref(ref_quarter: str) -> date:
    period = pd.Period(ref_quarter, freq="Q")
    return quarter_end_date(period)


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


def _compute_metrics(df: pd.DataFrame, truth_col: str) -> dict:
    use = df[df["y_pred"].notna() & df[truth_col].notna()].copy()
    if use.empty:
        return {"rmse": np.nan, "mae": np.nan, "count": 0, "by_ref_offset": {}}

    err = use["y_pred"] - use[truth_col]
    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))

    by_ref_offset: dict[str, dict] = {}
    for ref_offset, g in use.groupby("ref_offset"):
        e = g["y_pred"] - g[truth_col]
        by_ref_offset[str(ref_offset)] = {
            "rmse": float(np.sqrt(np.mean(e**2))) if len(g) else np.nan,
            "mae": float(np.mean(np.abs(e))) if len(g) else np.nan,
            "count": int(len(g)),
        }

    return {"rmse": rmse, "mae": mae, "count": int(len(use)), "by_ref_offset": by_ref_offset}


def _fit_predict_one(
    train_df: pd.DataFrame,
    test_row: pd.Series,
    *,
    feature_cols: list[str],
    label_col: str,
    model: str = "ridge",
    alphas: list[float] | None = None,
) -> tuple[float | None, dict]:
    if train_df.empty:
        return None, {"n_train": 0, "n_features": 0, "alpha_selected": np.nan}

    X = train_df[feature_cols].copy()
    y = train_df[label_col].copy()

    nan_frac = X.isna().mean()
    keep_cols = nan_frac[nan_frac <= 0.5].index.tolist()
    X = X[keep_cols]
    if X.empty:
        return None, {"n_train": int(len(train_df)), "n_features": 0, "alpha_selected": np.nan}

    means = X.mean()
    stds = X.std(ddof=0).replace(0.0, 1.0)
    X = X.fillna(means)
    X = (X - means) / stds

    alpha_selected = np.nan
    if model == "ridge":
        reg = RidgeCV(alphas=alphas or [0.01, 0.1, 1.0, 10.0, 100.0])
        reg.fit(X.to_numpy(), y.to_numpy())
        alpha_selected = float(reg.alpha_)
    elif model == "ols":
        reg = LinearRegression(fit_intercept=True)
        reg.fit(X.to_numpy(), y.to_numpy())
    else:
        raise ValueError(f"Unknown model: {model}")

    x0 = test_row[keep_cols].copy()
    x0 = x0.fillna(means)
    x0 = (x0 - means) / stds
    y_pred = float(reg.predict(x0.to_numpy().reshape(1, -1))[0])

    return y_pred, {
        "n_train": int(len(train_df)),
        "n_features": int(len(keep_cols)),
        "alpha_selected": alpha_selected,
    }


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
    transform_map = _collect_transform_map(meta_csv)
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
    agg_spec = _build_agg_spec(catalog, predictor_pit_keys)

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

            row_dict["y_asof_latest"] = y_asof_value if y_asof_value is not None else np.nan
            row_dict["y_final"] = y_final_value if y_final_value is not None else np.nan
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

    # Apply per-series transforms (PIT-safe) if meta_data.csv provides them.
    panel = _transform_panel_inplace(
        panel,
        meta_df=meta_df,
        transform_map=transform_map,
        catalog=catalog,
        series_key_map=series_key_map,
        predictor_keys=predictor_keys,
        target_key=target_canonical,
    )

    # Model labels are in model-space; preserve levels for audit.
    label_cols = {"y_asof_latest_model", "y_final_model"}
    meta_cols = {
        "asof_date",
        "ref_quarter",
        "ref_offset",
        "ref_quarter_end",
        "target_transform",
        "y_asof_latest_release_rank",
        "y_asof_latest_selected_asof_utc",
        "y_final_release_rank",
        "y_final_selected_asof_utc",
        "y_final_policy_mode",
        "y_asof_latest_level",
        "y_final_level",
    }

    # Exclude preserved level columns from features by default
    feature_cols = [
        c
        for c in panel.columns
        if c not in label_cols
        and c not in meta_cols
        and not c.endswith("_level")
        and not c.startswith("y_")
    ]

    alphas = [float(x.strip()) for x in args.alphas.split(",") if x.strip()]

    predictions: list[dict] = []
    for asof_date in sorted(panel["asof_date"].unique()):
        history = panel[panel["asof_date"] < asof_date]
        if history.empty:
            continue

        for ref_offset in ref_offsets:
            history_offset = history[history["ref_offset"] == ref_offset]
            history_offset = history_offset[history_offset["ref_quarter_end"] <= train_end_date]
            history_offset = history_offset[history_offset["y_final_model"].notna()]

            test_row = panel[
                (panel["asof_date"] == asof_date) & (panel["ref_offset"] == ref_offset)
            ]
            if test_row.empty:
                continue
            test_row = test_row.iloc[0]

            y_pred_model, train_meta = _fit_predict_one(
                history_offset,
                test_row,
                feature_cols=feature_cols,
                label_col="y_final_model",
                model=args.model,
                alphas=alphas,
            )

            target_tr = str(test_row.get("target_transform") or "level")
            y_pred_level = _inverse_target_transform_one(
                y_pred_model=y_pred_model,
                target_transform=target_tr,
                ref_quarter=str(test_row["ref_quarter"]),
                asof_date=asof_date,
                panel=panel,
            )

            predictions.append(
                {
                    "asof_date": asof_date,
                    "ref_quarter": test_row["ref_quarter"],
                    "ref_offset": ref_offset,
                    "ref_quarter_end": test_row["ref_quarter_end"],
                    "target_transform": target_tr,
                    "y_pred_model": y_pred_model,
                    "y_pred_level": y_pred_level,
                    "y_asof_latest_level": test_row.get("y_asof_latest_level"),
                    "y_asof_latest_model": test_row.get("y_asof_latest_model"),
                    "y_final_level": test_row.get("y_final_level"),
                    "y_final_3rd_model": test_row.get("y_final_model"),
                    "y_asof_latest_release_rank": test_row.get("y_asof_latest_release_rank"),
                    "y_asof_latest_selected_asof_utc": test_row.get(
                        "y_asof_latest_selected_asof_utc"
                    ),
                    "y_final_release_rank": test_row.get("y_final_release_rank"),
                    "y_final_selected_asof_utc": test_row.get("y_final_selected_asof_utc"),
                    "n_train": train_meta["n_train"],
                    "n_features": train_meta["n_features"],
                    "alpha_selected": train_meta["alpha_selected"],
                }
            )

pred_df = pd.DataFrame(predictions)
pred_df = pred_df.sort_values(["asof_date", "ref_offset"])

score_mask = (pred_df["ref_quarter_end"] >= score_start_date) & (
    pred_df["ref_quarter_end"] <= score_end_date
)
scored = pred_df.loc[score_mask].copy()

def _compute_metrics(df: pd.DataFrame, *, pred_col: str, truth_col: str) -> dict:
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

metrics = {
    "real_time_model_space": _compute_metrics(scored, pred_col="y_pred_model", truth_col="y_asof_latest_model"),
    "stable_model_space_3rd_release": _compute_metrics(scored, pred_col="y_pred_model", truth_col="y_final_3rd_model"),
    "real_time_level_space": _compute_metrics(scored, pred_col="y_pred_level", truth_col="y_asof_latest_level"),
    "stable_level_space_3rd_release": _compute_metrics(scored, pred_col="y_pred_level", truth_col="y_final_level"),
}

run_metadata = {
    "target_input": args.target,
    "target_canonical": target_canonical,
    "target_pit_key": target_pit,
    "target_transform": str(transform_map.get(target_canonical, "level")),
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

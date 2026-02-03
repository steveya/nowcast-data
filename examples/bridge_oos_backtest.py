"""Revision-aware walk-forward backtest for the bridge nowcast.

This script intentionally lives outside the library package so it can be run as:

    python scripts/bridge_oos_backtest.py --meta-csv data/meta_data.csv

It performs an expanding-window walk-forward backtest using *PIT snapshots* at
each forecast origin ("vintage", aka as-of date).

Key design points
-----------------
1) **No leakage by construction**: every feature comes from a PIT snapshot at the
   forecast origin date.
2) **Revision-aware labels**:
   - `y_asof_latest` is the latest release available *as of the vintage*.
   - `y_final` is a "final-ish" proxy resolved using an `evaluation_asof_date`
     and a configurable `TargetPolicy` (e.g., latest available by 2025-12-31, or
     1st/2nd/3rd print).
3) **Using non-final GDP revisions as features**: enable `--include-target-release-features`
   to add features like `gdpc1.rel1`, `gdpc1.rel2`, `gdpc1.latest`, and revision
   deltas (see `QuarterlyTargetFeatureSpec`). This is the clean way to use
   real-time GDP prints as predictors while forecasting the final GDP target.

Outputs
-------
Writes two artifacts:
* A parquet (or csv) of per-vintage predictions and metadata.
* A json of evaluation metrics (overall and grouped by month-in-quarter).

The script computes **both**:
* "real-time" errors vs `y_asof_latest` (what you would have scored then)
* "final-ish" errors vs `y_final` (offline, stable target proxy)
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import date, datetime
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, RidgeCV

from nowcast_data.models.datasets import (
    VintageTrainingDatasetConfig,
    build_vintage_training_dataset,
)
from nowcast_data.models.target_features import QuarterlyTargetFeatureSpec
from nowcast_data.models.target_policy import TargetPolicy
from nowcast_data.pit.api import PITDataManager
from nowcast_data.pit.core.catalog import SeriesCatalog
from nowcast_data.pit.core.models import SeriesMetadata


def _parse_date(x: str) -> date:
    return pd.Timestamp(x).date()


def _make_calendar_vintages(
    start_date: date,
    end_date: date,
    freq: Literal["D", "W", "B"],
) -> list[date]:
    from datetime import timedelta

    if freq == "D":
        step = timedelta(days=1)
    elif freq == "W":
        step = timedelta(days=7)
    elif freq == "B":
        step = None
    else:
        raise ValueError(f"Unsupported freq={freq}")

    out: list[date] = []
    cur = start_date
    while cur <= end_date:
        out.append(cur)
        if step is not None:
            cur = (pd.Timestamp(cur) + step).date()
        else:
            # next business day
            cur = (pd.Timestamp(cur) + timedelta(days=1)).date()
            while pd.Timestamp(cur).weekday() >= 5:
                cur = (pd.Timestamp(cur) + timedelta(days=1)).date()
    return out


def _load_series_catalog_from_meta_csv(meta_csv: Path) -> SeriesCatalog:
    """Build a minimal SeriesCatalog from the benchmark meta_data.csv.

    The library's dataset builders use the catalog mainly for:
    * frequency (d/b vs m vs q)
    * monthly obs_date anchoring validation (obs_date_anchor)
    * source series mapping for AlphaForge (source_series_id)
    """

    df = pd.read_csv(meta_csv)
    df.columns = [c.strip().lower() for c in df.columns]
    if "series" not in df.columns:
        raise ValueError("meta_csv must contain a 'series' column")

    cat = SeriesCatalog()
    for _, row in df.iterrows():
        series = str(row.get("series", "")).strip()
        if not series:
            continue
        series_key = series.lower()
        series_id = series.upper()
        freq = str(row.get("freq", "m")).strip().lower() or "m"

        # Default assumptions for the benchmark:
        # - data lives in AlphaForge PIT store
        # - obs_date keys are stored as period-end (month-end / quarter-end)
        meta = SeriesMetadata(
            series_key=series_key,
            country="US",
            source="alfred",
            source_series_id=series_id,
            frequency=freq,
            pit_mode="ALFRED_REALTIME",
            adapter="alphaforge",
            obs_date_anchor="end" if freq in {"m", "q"} else None,
            description=str(row.get("name", "")) or None,
        )
        cat.add(meta)
    return cat


def _month_in_quarter(d: date) -> int:
    return ((d.month - 1) % 3) + 1


def _compute_metrics(df: pd.DataFrame, truth_col: str) -> dict:
    use = df[df["y_pred"].notna() & df[truth_col].notna()].copy()
    if use.empty:
        return {"rmse": np.nan, "mae": np.nan, "count": 0, "by_month_in_quarter": {}}

    err = use["y_pred"] - use[truth_col]
    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))
    out = {"rmse": rmse, "mae": mae, "count": int(len(use))}

    by_miq = {}
    for miq, g in use.groupby("month_in_quarter"):
        e = g["y_pred"] - g[truth_col]
        by_miq[str(miq)] = {
            "rmse": float(np.sqrt(np.mean(e**2))) if len(g) else np.nan,
            "mae": float(np.mean(np.abs(e))) if len(g) else np.nan,
            "count": int(len(g)),
        }
    out["by_month_in_quarter"] = by_miq
    return out


def build_vintage_panel_with_release_meta(
    *,
    pit: PITDataManager,
    asof_dates: list[date],
    config: VintageTrainingDatasetConfig,
) -> pd.DataFrame:
    """Build a panel (index=asof_date) and keep target release metadata."""

    rows: list[dict] = []
    for asof in asof_dates:
        try:
            ds, meta = build_vintage_training_dataset(
                pit.adapters["alphaforge"],
                pit.catalog,
                config=config,
                asof_date=asof,
                ingest_from_ctx_source=False,
            )
        except Exception:
            continue
        if ds.empty:
            continue

        current_ref = meta["current_ref_quarter"]
        if current_ref not in ds.index:
            continue

        row = ds.loc[current_ref].to_dict()
        row["asof_date"] = asof
        row["ref_quarter"] = str(current_ref)
        row["month_in_quarter"] = _month_in_quarter(asof)

        # Pull release meta for the current quarter
        rel_meta = meta.get("target_release_meta", {}).get(str(current_ref), {})
        y_asof_meta = rel_meta.get("y_asof_latest", {})
        y_final_meta = rel_meta.get("y_final", {})
        row["y_asof_latest_release_rank"] = y_asof_meta.get("selected_release_rank")
        row["y_asof_latest_selected_asof_utc"] = y_asof_meta.get("selected_release_asof_utc")
        row["y_final_release_rank"] = y_final_meta.get("selected_release_rank")
        row["y_final_selected_asof_utc"] = y_final_meta.get("selected_release_asof_utc")
        row["y_final_policy_mode"] = y_final_meta.get("policy_mode")

        rows.append(row)

    if not rows:
        return pd.DataFrame()

    panel = pd.DataFrame(rows).set_index("asof_date").sort_index()
    return panel


def walk_forward_fit_predict(
    panel: pd.DataFrame,
    *,
    label_col: str,
    model: Literal["ridge", "ols"],
    alphas: list[float],
    min_train_periods: int,
    rolling_window: int | None,
    standardize: bool,
    max_nan_fraction: float,
    feature_exclude: set[str],
) -> pd.DataFrame:
    """Minimal walk-forward engine over a panel indexed by asof_date."""

    panel = panel.copy()
    panel_vintages = list(panel.index)

    # Determine feature columns
    feature_cols = [c for c in panel.columns if c not in feature_exclude]

    rows = []
    for i, test_vintage in enumerate(panel_vintages):
        if rolling_window is None:
            train_vintages = panel_vintages[:i]
        else:
            train_vintages = panel_vintages[max(0, i - rolling_window) : i]

        train = panel.loc[train_vintages]
        # Require labels for training
        train = train[train[label_col].notna()]
        if len(train) < min_train_periods:
            rows.append({"asof_date": test_vintage, "y_pred": np.nan, "n_train": len(train)})
            continue

        X = train[feature_cols]
        y = train[label_col]

        # Drop high-NaN features based on training
        keep = X.columns[X.isna().mean() <= max_nan_fraction]
        X = X[keep]

        # Impute with training means
        means = X.mean()
        X = X.fillna(means)

        # Standardize
        if standardize:
            stds = X.std(ddof=0).replace(0.0, 1.0)
            Xs = (X - means) / stds
        else:
            stds = None
            Xs = X

        # Fit
        alpha_selected = np.nan
        if model == "ridge":
            reg = RidgeCV(alphas=alphas)
            reg.fit(Xs.to_numpy(), y.to_numpy())
            alpha_selected = float(reg.alpha_)
        elif model == "ols":
            reg = LinearRegression(fit_intercept=True)
            reg.fit(Xs.to_numpy(), y.to_numpy())
        else:
            raise ValueError(f"Unknown model: {model}")

        # Test point
        x0 = panel.loc[[test_vintage], keep]
        x0 = x0.fillna(means)
        if standardize:
            x0 = (x0 - means) / stds
        y_pred = float(reg.predict(x0.to_numpy())[0])

        rows.append(
            {
                "asof_date": test_vintage,
                "y_pred": y_pred,
                "n_train": int(len(train)),
                "n_features": int(len(keep)),
                "alpha_selected": alpha_selected,
            }
        )

    out = pd.DataFrame(rows).set_index("asof_date")
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--meta-csv", type=str, required=True, help="Path to meta_data.csv")
    p.add_argument("--target", type=str, default="gdpc1", help="Target series key (default gdpc1)")
    p.add_argument(
        "--start-date",
        type=str,
        default="2000-01-01",
        help="Vintage grid start date (YYYY-MM-DD)",
    )
    p.add_argument(
        "--end-date",
        type=str,
        default="2025-12-31",
        help="Vintage grid end date (YYYY-MM-DD)",
    )
    p.add_argument(
        "--eval-start",
        type=str,
        default="2013-01-01",
        help="Evaluation window start (YYYY-MM-DD)",
    )
    p.add_argument(
        "--eval-end",
        type=str,
        default="2025-12-31",
        help="Evaluation window end (YYYY-MM-DD)",
    )
    p.add_argument(
        "--vintage-grid",
        choices=["calendar", "event"],
        default="calendar",
        help="calendar=regular D/W/B grid; event=union of true PIT vintages",
    )
    p.add_argument("--freq", choices=["D", "W", "B"], default="W", help="Calendar grid freq")
    p.add_argument(
        "--evaluation-asof-date",
        type=str,
        default="2025-12-31",
        help="As-of date used to resolve y_final (YYYY-MM-DD)",
    )
    p.add_argument(
        "--final-target-mode",
        choices=["latest_available", "nth_release"],
        default="latest_available",
        help="How to resolve y_final (final-ish).",
    )
    p.add_argument(
        "--final-target-nth",
        type=int,
        default=3,
        help="If --final-target-mode=nth_release, which release rank to use (1=1st print).",
    )
    p.add_argument(
        "--final-target-max-rank",
        type=int,
        default=3,
        help="Cap number of releases considered when resolving y_final.",
    )
    p.add_argument(
        "--include-target-release-features",
        action="store_true",
        help="Include non-final target releases (e.g., gdpc1.rel1/rel2/rel3) as features.",
    )
    p.add_argument(
        "--include-y-asof-latest-as-feature",
        action="store_true",
        help="Include y_asof_latest as a feature with explicit indicator+zero-imputation.",
    )
    p.add_argument("--model", choices=["ridge", "ols"], default="ridge")
    p.add_argument(
        "--alphas",
        type=str,
        default="0.01,0.1,1,10,100",
        help="Comma-separated ridge alphas.",
    )
    p.add_argument("--min-train-periods", type=int, default=40)
    p.add_argument("--rolling-window", type=int, default=0, help="0 means expanding")
    p.add_argument("--no-standardize", action="store_true")
    p.add_argument("--max-nan-fraction", type=float, default=0.5)
    p.add_argument(
        "--out-dir",
        type=str,
        default="outputs/bridge_backtest",
        help="Output directory.",
    )
    p.add_argument(
        "--out-format",
        choices=["parquet", "csv"],
        default="parquet",
        help="Prediction file format.",
    )

    args = p.parse_args()

    meta_csv = Path(args.meta_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build a minimal catalog from meta_data.csv so frequency-sensitive transforms work.
    catalog = _load_series_catalog_from_meta_csv(meta_csv)

    # PIT manager (expects AlphaForge PIT store configured via env vars).
    pit = PITDataManager(catalog)
    if "alphaforge" not in pit.adapters:
        raise RuntimeError(
            "AlphaForge adapter not available. Ensure FRED_API_KEY is set and "
            "ALPHAFORGE_STORE_ROOT points at your PIT store."
        )

    df_meta = pd.read_csv(meta_csv)
    series_all = [str(s).strip().lower() for s in df_meta["series"].tolist() if str(s).strip()]
    target = args.target.strip().lower()
    predictor_keys = [s for s in series_all if s != target]

    # Agg spec: simple defaults by frequency
    agg_spec = {}
    for s in predictor_keys:
        meta = catalog.get(s)
        freq = (meta.frequency or "").lower() if meta else ""
        if freq in {"d", "b"}:
            # daily features expanded separately; agg_spec unused
            agg_spec[s] = "last"
        elif freq in {"m", "q"}:
            agg_spec[s] = "mean"
        else:
            agg_spec[s] = "mean"

    start_date = _parse_date(args.start_date)
    end_date = _parse_date(args.end_date)
    eval_start = _parse_date(args.eval_start)
    eval_end = _parse_date(args.eval_end)

    evaluation_asof_date = _parse_date(args.evaluation_asof_date)
    if args.final_target_mode == "latest_available":
        final_target_policy = TargetPolicy(
            mode="latest_available", max_release_rank=args.final_target_max_rank
        )
    else:
        final_target_policy = TargetPolicy(
            mode="nth_release",
            nth=int(args.final_target_nth),
            max_release_rank=args.final_target_max_rank,
        )

    # Vintage grid
    if args.vintage_grid == "calendar":
        vintages = _make_calendar_vintages(start_date, end_date, args.freq)
    else:
        # Union of true PIT vintage dates across involved series
        adapter = pit.adapters["alphaforge"]
        all_series = [target] + predictor_keys
        vintages_set: set[date] = set()
        for s in all_series:
            try:
                vintages_set.update(adapter.list_vintages(s))
            except Exception:
                continue
        vintages = sorted([d for d in vintages_set if start_date <= d <= end_date])

    # Dataset config: only current quarter row per vintage (ref_offsets=[0])
    target_feature_spec = (
        QuarterlyTargetFeatureSpec(max_release_rank=args.final_target_max_rank)
        if args.include_target_release_features
        else None
    )
    ds_config = VintageTrainingDatasetConfig(
        target_series_key=target,
        predictor_series_keys=predictor_keys,
        agg_spec=agg_spec,
        include_partial_quarters=True,
        ref_offsets=[0],
        evaluation_asof_date=evaluation_asof_date,
        final_target_policy=final_target_policy,
        target_feature_spec=target_feature_spec,
    )

    panel = build_vintage_panel_with_release_meta(
        pit=pit,
        asof_dates=vintages,
        config=ds_config,
    )
    if panel.empty:
        raise RuntimeError(
            "Built empty panel. Check that PIT data exists and the store root is correct."
        )

    # Walk-forward training label: final GDP (offline). This matches the project goal.
    label_col = "y_final"
    alphas = [float(x.strip()) for x in args.alphas.split(",") if x.strip()]
    rolling_window = None if int(args.rolling_window) <= 0 else int(args.rolling_window)

    feature_exclude = {
        "ref_quarter",
        "month_in_quarter",
        "y_asof_latest",
        "y_final",
        "y_asof_latest_release_rank",
        "y_asof_latest_selected_asof_utc",
        "y_final_release_rank",
        "y_final_selected_asof_utc",
        "y_final_policy_mode",
    }
    preds = walk_forward_fit_predict(
        panel,
        label_col=label_col,
        model=args.model,
        alphas=alphas,
        min_train_periods=int(args.min_train_periods),
        rolling_window=rolling_window,
        standardize=not args.no_standardize,
        max_nan_fraction=float(args.max_nan_fraction),
        feature_exclude=feature_exclude,
    )

    out = panel.join(preds, how="left")
    out = out.reset_index().rename(columns={"index": "asof_date"})

    # Filter evaluation window
    out["asof_date"] = pd.to_datetime(out["asof_date"]).dt.date
    eval_mask = (out["asof_date"] >= eval_start) & (out["asof_date"] <= eval_end)
    out_eval = out.loc[eval_mask].copy()

    # Metrics: real-time and final-ish
    metrics = {
        "run_at_utc": datetime.utcnow().isoformat() + "Z",
        "config": {
            "meta_csv": str(meta_csv),
            "target": target,
            "start_date": str(start_date),
            "end_date": str(end_date),
            "eval_start": str(eval_start),
            "eval_end": str(eval_end),
            "vintage_grid": args.vintage_grid,
            "freq": args.freq,
            "evaluation_asof_date": str(evaluation_asof_date),
            "final_target_policy": asdict(final_target_policy),
            "include_target_release_features": bool(args.include_target_release_features),
            "model": args.model,
            "alphas": alphas,
            "min_train_periods": int(args.min_train_periods),
            "rolling_window": rolling_window,
            "standardize": not args.no_standardize,
            "max_nan_fraction": float(args.max_nan_fraction),
        },
        "metrics": {
            "real_time_vs_y_asof_latest": _compute_metrics(out_eval, "y_asof_latest"),
            "finalish_vs_y_final": _compute_metrics(out_eval, "y_final"),
        },
    }

    # Persist
    pred_path = out_dir / (
        f"predictions_{target}_{args.vintage_grid}."
        f"{'parquet' if args.out_format == 'parquet' else 'csv'}"
    )
    metrics_path = out_dir / f"metrics_{target}_{args.vintage_grid}.json"

    if args.out_format == "parquet":
        out.to_parquet(pred_path, index=False)
    else:
        out.to_csv(pred_path, index=False)

    metrics_path.write_text(json.dumps(metrics, indent=2))

    print(f"Wrote predictions: {pred_path}")
    print(f"Wrote metrics:     {metrics_path}")


if __name__ == "__main__":
    main()

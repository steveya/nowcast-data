#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
from pathlib import Path

from alphaforge.data.context import DataContext
from alphaforge.data.fred_source import FREDDataSource
from alphaforge.store.duckdb_parquet import DuckDBParquetStore

from nowcast_data.benchmark.pit_builder import (
    build_and_export_benchmark_pit,
    ingest_alfred_full_history,
    load_benchmark_metadata,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build benchmark PIT dataset")
    parser.add_argument("--meta-path", default="data/meta_data.csv")
    parser.add_argument("--store-root", default="./pit_store")
    parser.add_argument("--out-dir", default="./benchmark_pit")
    parser.add_argument("--skip-ingest", action="store_true")
    args = parser.parse_args()

    store = DuckDBParquetStore(root=args.store_root)
    sources = {}
    fred_api_key = os.environ.get("FRED_API_KEY")
    if fred_api_key:
        sources["fred"] = FREDDataSource(api_key=fred_api_key)

    ctx = DataContext(sources=sources, calendars={}, store=store)
    specs = load_benchmark_metadata(Path(args.meta_path))

    if not args.skip_ingest and "fred" in sources:
        for spec in specs:
            ingest_alfred_full_history(ctx, spec, fred_source_name="fred")

    build_and_export_benchmark_pit(ctx, specs, out_dir=args.out_dir)


if __name__ == "__main__":
    main()

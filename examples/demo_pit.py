"""Example usage of the PIT macro data library.

This script can also build a PIT store for all series listed in meta_data.csv
over a specified as-of date range.
"""

from datetime import date
import json
from pathlib import Path
import time
import pandas as pd

from nowcast_data.pit.core.catalog import SeriesCatalog
from nowcast_data.pit.api import PITDataManager


def _collect_fred_vintages(
    *,
    manager: PITDataManager,
    series_ids: list[str],
    start_date: date,
    end_date: date,
    min_delay_seconds: float = 0.5,
    max_delay_seconds: float = 120.0,
    max_retries: int = 5,
) -> dict[str, list[date]]:
    if "alphaforge" not in manager.adapters:
        raise ValueError("alphaforge adapter is required to list FRED vintages")
    adapter = manager.adapters["alphaforge"]
    ctx = adapter._ctx
    source = ctx.sources.get("fred")
    fred_client = getattr(source, "_fred", None)
    if fred_client is None:
        raise ValueError("FRED client not available in alphaforge context")

    vintages_by_series: dict[str, list[date]] = {}
    retryable_substrings = (
        "rate limit",
        "too many requests",
        "remote end closed connection",
        "timed out",
        "timeout",
        "connection reset",
        "connection aborted",
        "temporarily unavailable",
        "service unavailable",
        "502",
        "503",
        "504",
    )

    def _fetch_vintage_dates(series_id: str) -> list[str] | None:
        delay = min_delay_seconds
        for attempt in range(1, max_retries + 1):
            try:
                dates = fred_client.get_series_vintage_dates(series_id)
                time.sleep(min_delay_seconds)
                return dates
            except Exception as exc:
                message = str(exc).lower()
                if attempt >= max_retries:
                    print(f"Warning: failed to list vintages for {series_id}: {exc}")
                    return None
                is_retryable = any(token in message for token in retryable_substrings)
                if is_retryable:
                    print(
                        f"Retry {attempt}/{max_retries} vintages for {series_id} "
                        f"after {delay:.1f}s: {exc}"
                    )
                    time.sleep(delay)
                    delay = min(delay * 2, max_delay_seconds)
                    continue
                print(f"Warning: failed to list vintages for {series_id}: {exc}")
                return None

    for series_id in series_ids:
        vintage_dates = _fetch_vintage_dates(series_id)
        if vintage_dates is None:
            vintages_by_series[series_id] = []
            continue

        vintages = pd.to_datetime(vintage_dates, utc=True).date
        vintages = [v for v in vintages if start_date <= v <= end_date]
        vintages_by_series[series_id] = sorted(set(vintages))
    return vintages_by_series


def main():
    """Demonstrate PIT data retrieval for G6 countries."""

    # Load series catalog
    catalog_path = Path(__file__).parent.parent / "series_catalog.yaml"
    catalog = SeriesCatalog(catalog_path)

    print("=== PIT Macro Data Library Demo ===\n")

    # List available series
    print("Available series:")
    for series_key in catalog.list_series():
        meta = catalog.get(series_key)
        pit_status = "✓ PIT" if meta.pit_mode != "NO_PIT" else "✗ No PIT"
        print(f"  {series_key:20s} [{meta.country}] {pit_status}")

    print("\n" + "=" * 50 + "\n")

    # Initialize PIT data manager
    # Note: Requires FRED_API_KEY environment variable for US data
    try:
        manager = PITDataManager(catalog)
        print("PIT Data Manager initialized successfully\n")
    except ValueError as e:
        print(f"Warning: {e}")
        print("Set FRED_API_KEY environment variable to access US data\n")
        return

    # Example 1: Get single series as-of a specific date
    print("Example 1: US GDP as-of 2020-01-15")
    print("-" * 50)

    try:
        df = manager.get_series_asof(
            series_key="US_GDP_SAAR",
            asof_date=date(2020, 1, 15),
            start=date(2019, 1, 1),
            end=date(2019, 12, 31),
        )
        print(f"Retrieved {len(df)} observations")
        print(f"Columns: {list(df.columns)}")
        if len(df) > 0:
            print(f"\nFirst observation:")
            print(df.iloc[0])
    except Exception as e:
        print(f"Error: {e}")

    print("\n" + "=" * 50 + "\n")

    # Example 2: Get multiple series (panel) as-of a date
    print("Example 2: US Panel (GDP, CPI, Unemployment) as-of 2020-06-01")
    print("-" * 50)

    try:
        panel = manager.get_panel_asof(
            series_keys=["US_GDP_SAAR", "US_CPI", "US_UNRATE"],
            asof_date=date(2020, 6, 1),
            start=date(2019, 1, 1),
            end=date(2020, 5, 31),
            wide=True,
        )
        print(f"Panel shape: {panel.shape}")
        print(f"Series: {list(panel.columns)}")
        if not panel.empty:
            print(f"\nLast 5 observations:")
            print(panel.tail())
    except Exception as e:
        print(f"Error: {e}")

    print("\n" + "=" * 50 + "\n")

    # Example 3: List vintages for a series
    print("Example 3: List vintages for US GDP")
    print("-" * 50)

    try:
        vintages = manager.get_series_vintages("US_GDP_SAAR")
        print(f"Total vintages: {len(vintages)}")
        if len(vintages) > 0:
            print(f"First vintage: {vintages[0]}")
            print(f"Last vintage: {vintages[-1]}")
            if len(vintages) > 5:
                print(f"Recent vintages: {vintages[-5:]}")
    except Exception as e:
        print(f"Error: {e}")

    print("\n" + "=" * 50 + "\n")

    # Example 4: Build PIT cube (multiple series x multiple asof dates)
    print("Example 4: Build small PIT cube")
    print("-" * 50)
    print("Building cube: US_GDP_SAAR at 3 asof dates...")

    try:
        cube = manager.build_pit_cube(
            series_keys=["US_GDP_SAAR"],
            asof_dates=[
                date(2020, 1, 15),
                date(2020, 3, 15),
                date(2020, 6, 15),
            ],
            start=date(2019, 10, 1),
            end=date(2019, 12, 31),
        )
        print(f"Cube shape: {cube.shape}")
        print(f"Unique asof dates: {cube['asof_date'].nunique()}")
        print(f"Unique obs dates: {cube['obs_date'].nunique()}")

        # Show how values may differ across vintages
        if not cube.empty:
            print("\nSample: Same observation at different vintages:")
            sample_obs_date = cube["obs_date"].iloc[0]
            subset = cube[cube["obs_date"] == sample_obs_date][["asof_date", "value"]]
            print(subset)
    except Exception as e:
        print(f"Error: {e}")

    print("\n" + "=" * 50 + "\n")
    print("Demo complete!")


def build_pit_store_from_metadata_csv(
    *,
    manager: PITDataManager,
    metadata_csv: Path,
    start_date: date,
    end_date: date,
    grid_mode: str = "event",
    asof_freq: str = "B",
    min_delay_seconds: float = 0.5,
    max_delay_seconds: float = 120.0,
    max_retries: int = 5,
    series_filter: list[str] | None = None,
    progress_path: Path | None = None,
    checkpoint_every: int = 100,
) -> None:
    """Build a PIT store for all series listed in meta_data.csv.

    Args:
        manager: PITDataManager with alphaforge adapter configured.
        metadata_csv: Path to meta_data.csv (expects a "series" column).
        start_date: Start of as-of date range.
        end_date: End of as-of date range.
        grid_mode: "event" for vintage-only as-of dates or "daily" for business-day grid.
        asof_freq: Frequency for as-of dates (used when grid_mode="daily").
        min_delay_seconds: Minimum delay between requests.
        max_delay_seconds: Maximum backoff delay on repeated rate-limit retries.
        max_retries: Max retries on rate-limit or transient errors.
        series_filter: Optional list of series IDs to ingest (upper-case FRED ids).
        progress_path: Optional path to JSON file to persist progress.
        checkpoint_every: Flush progress/DB every N as-of fetches (0 to disable).
    """
    if "alphaforge" not in manager.adapters:
        raise ValueError("alphaforge adapter is required to build PIT store")

    meta_df = pd.read_csv(metadata_csv)
    if "series" not in meta_df.columns:
        raise ValueError("meta_data.csv must contain a 'series' column")

    series_ids = meta_df["series"].dropna().astype(str).str.strip().str.upper().unique().tolist()
    if series_filter:
        allowed = {s.strip().upper() for s in series_filter}
        series_ids = [s for s in series_ids if s in allowed]

    progress: dict[str, list[str]] = {}
    if progress_path is not None and progress_path.exists():
        try:
            progress = json.loads(progress_path.read_text())
        except Exception:
            progress = {}

    print("\n" + "=" * 80)
    print("BUILD PIT STORE")
    print("=" * 80)
    print(f"Series count: {len(series_ids)}")
    print(f"Grid mode: {grid_mode}")

    adapter = manager.adapters["alphaforge"]

    def _fetch_with_retry(series_id: str, asof_date: date) -> int:
        retryable_substrings = (
            "rate limit",
            "too many requests",
            "remote end closed connection",
            "timed out",
            "timeout",
            "connection reset",
            "connection aborted",
            "temporarily unavailable",
            "service unavailable",
            "502",
            "503",
            "504",
        )
        delay = min_delay_seconds
        for attempt in range(1, max_retries + 1):
            try:
                observations = adapter.fetch_asof(
                    series_id,
                    asof_date,
                    start=start_date,
                    end=end_date,
                    metadata=None,
                    ingest_from_ctx_source=True,
                )
                time.sleep(min_delay_seconds)
                return len(observations) if observations is not None else 0
            except Exception as exc:
                message = str(exc).lower()
                if attempt >= max_retries:
                    print(f"Warning: failed {series_id} @ {asof_date}: {exc}")
                    return 0
                is_retryable = any(token in message for token in retryable_substrings)
                if is_retryable:
                    print(
                        f"Retry {attempt}/{max_retries} for {series_id} @ {asof_date} "
                        f"after {delay:.1f}s: {exc}"
                    )
                    time.sleep(delay)
                    delay = min(delay * 2, max_delay_seconds)
                    continue
                print(f"Warning: failed {series_id} @ {asof_date}: {exc}")
                return 0

    def _checkpoint_store(fetches_since_checkpoint: int) -> int:
        if checkpoint_every <= 0:
            return fetches_since_checkpoint
        if fetches_since_checkpoint < checkpoint_every:
            return fetches_since_checkpoint
        try:
            adapter._ctx.pit.conn.execute("CHECKPOINT")
        except Exception as exc:
            print(f"Warning: checkpoint failed: {exc}")
            return fetches_since_checkpoint
        if progress_path is not None:
            progress_path.write_text(json.dumps(progress, indent=2))
        print(f"Checkpoint: saved progress after {fetches_since_checkpoint} fetches")
        return 0

    if grid_mode == "daily":
        asof_dates = pd.date_range(start=start_date, end=end_date, freq=asof_freq).date.tolist()
        print(f"As-of dates: {len(asof_dates)} ({start_date} to {end_date}, freq={asof_freq})")
        total = len(series_ids) * len(asof_dates)
        processed = 0
        fetches_since_checkpoint = 0
        for series_id in series_ids:
            print(f"Loading series: {series_id}")
            completed = {d for d in progress.get(series_id, [])}
            for asof_date in asof_dates:
                if str(asof_date) in completed:
                    processed += 1
                    continue
                _fetch_with_retry(series_id, asof_date)
                fetches_since_checkpoint += 1
                completed.add(str(asof_date))
                progress[series_id] = sorted(completed)
                if progress_path is not None:
                    progress_path.write_text(json.dumps(progress, indent=2))

                fetches_since_checkpoint = _checkpoint_store(fetches_since_checkpoint)

                processed += 1
                if processed % 100 == 0 or processed == total:
                    print(f"Progress: {processed}/{total}")
        return

    if grid_mode != "event":
        raise ValueError("grid_mode must be 'event' or 'daily'")

    vintages_by_series = _collect_fred_vintages(
        manager=manager,
        series_ids=series_ids,
        start_date=start_date,
        end_date=end_date,
        min_delay_seconds=min_delay_seconds,
        max_delay_seconds=max_delay_seconds,
        max_retries=max_retries,
    )
    total = sum(len(vs) for vs in vintages_by_series.values())
    processed = 0
    fetches_since_checkpoint = 0

    for series_id, vintages in vintages_by_series.items():
        print(f"Loading series: {series_id}")
        completed = {d for d in progress.get(series_id, [])}
        if vintages and len(completed) >= len(vintages):
            print(f"Skipping {series_id}: already loaded {len(completed)} vintages")
            processed += len(vintages)
            continue
        for asof_date in vintages:
            if str(asof_date) in completed:
                processed += 1
                continue
            _fetch_with_retry(series_id, asof_date)
            fetches_since_checkpoint += 1
            completed.add(str(asof_date))
            progress[series_id] = sorted(completed)
            if progress_path is not None:
                progress_path.write_text(json.dumps(progress, indent=2))

            fetches_since_checkpoint = _checkpoint_store(fetches_since_checkpoint)

            processed += 1
            if processed % 100 == 0 or processed == total:
                print(f"Progress: {processed}/{total}")


def save_gdp_pit_demo(manager: PITDataManager):
    """
    This example demonstrates how to get point-in-time data for US GDP (US_GDP_SAAR)
    and save it to a CSV file.
    """
    print("\n" + "=" * 50 + "\n")
    print("Example 5: Save US GDP Point-in-Time data to CSV")
    print("-" * 50)

    # Create output directory
    output_dir = Path(__file__).parent.parent / "outputs" / "gdp_pit_demo"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving output to {output_dir}")

    series_key = "US_GDP_SAAR"

    # 1. Get all vintage dates for the series
    print(f"Fetching vintage dates for {series_key}...")
    vintage_dates = manager.get_series_vintages(series_key)
    print(f"Found {len(vintage_dates)} vintage dates.")

    # For this example, we will use all vintages
    print(f"Fetching data for all {len(vintage_dates)} vintages...")

    all_series = []
    import time

    for vintage in vintage_dates:
        df = manager.get_series_asof(series_key, vintage)
        # The 'asof_date' column represents the vintage date
        # The 'obs_date' column represents the reference period
        df = df.set_index(["asof_date", "obs_date"])["value"]
        df.index = df.index.set_levels(pd.to_datetime(df.index.levels[1]).to_period("Q"), level=1)
        all_series.append(df)
        time.sleep(0.1)

    # Concatenate all series into a single multi-indexed series
    result_series = pd.concat(all_series)

    # Reshape the data to have vintages as columns
    result_df = result_series.unstack(level=0)

    # Save to CSV
    output_path = output_dir / "gdp_pit_demo.csv"
    result_df.to_csv(output_path)
    print(f"Successfully saved data to {output_path}")


if __name__ == "__main__":
    main()
    # Also run the new demo
    catalog_path = Path(__file__).parent.parent / "series_catalog.yaml"
    catalog = SeriesCatalog(catalog_path)
    try:
        manager = PITDataManager(catalog)
        save_gdp_pit_demo(manager)
        # Build PIT store for all series in meta_data.csv
        metadata_csv = Path(__file__).parent.parent / "data" / "meta_data.csv"
        build_pit_store_from_metadata_csv(
            manager=manager,
            metadata_csv=metadata_csv,
            start_date=date(2000, 1, 1),
            end_date=date(2025, 12, 31),
            grid_mode="event",
            series_filter=["GDPC1"],
            progress_path=Path(__file__).parent.parent / "outputs" / "pit_progress.json",
        )
    except ValueError as e:
        print(f"Warning: {e}")

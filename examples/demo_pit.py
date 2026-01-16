"""Example usage of the PIT macro data library."""

from datetime import date
from pathlib import Path
import pandas as pd

from nowcast_data.pit.core.catalog import SeriesCatalog
from nowcast_data.pit.api import PITDataManager


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
    
    print("\n" + "="*50 + "\n")
    
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
    
    print("\n" + "="*50 + "\n")
    
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
    
    print("\n" + "="*50 + "\n")
    
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
    
    print("\n" + "="*50 + "\n")
    
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
            sample_obs_date = cube['obs_date'].iloc[0]
            subset = cube[cube['obs_date'] == sample_obs_date][['asof_date', 'value']]
            print(subset)
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n" + "="*50 + "\n")
    print("Demo complete!")


def save_gdp_pit_demo(manager: PITDataManager):
    """
    This example demonstrates how to get point-in-time data for US GDP (US_GDP_SAAR)
    and save it to a CSV file.
    """
    print("\n" + "="*50 + "\n")
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
        df.index = df.index.set_levels(pd.to_datetime(df.index.levels[1]).to_period('Q'), level=1)
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
    except ValueError as e:
        print(f"Warning: {e}")

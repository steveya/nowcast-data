"""Example usage of the PIT macro data library."""

from datetime import date
from pathlib import Path

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


if __name__ == "__main__":
    main()

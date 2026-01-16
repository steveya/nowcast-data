import os
from datetime import date
from dotenv import load_dotenv
import pandas as pd

from nowcast_data.pit.api import PITDataManager
from nowcast_data.pit.core.catalog import SeriesCatalog

# Load environment variables from .env file
load_dotenv()


def get_usgdp_point_in_time():
    """
    This example demonstrates how to get point-in-time data for US GDP (US_GDP_SAAR).
    The output is a pandas Series with a multi-index, where the first level is the
    vintage date (the date the data was available) and the second level is the
    observation date (the period the data refers to).
    """
    if "FRED_API_KEY" not in os.environ:
        print("FRED_API_KEY environment variable not set. Please set it in your .env file.")
        return

    # Load the series catalog
    catalog_path = os.path.join(os.path.dirname(__file__), "..", "series_catalog.yaml")
    catalog = SeriesCatalog(catalog_path)
    data_manager = PITDataManager(catalog=catalog)

    series_key = "US_GDP_SAAR"

    # 1. Get all vintage dates for the series
    print(f"Fetching vintage dates for {series_key}...")
    vintage_dates = data_manager.get_series_vintages(series_key)
    print(f"Found {len(vintage_dates)} vintage dates.")

    # For this example, we will use the last 5 vintages to keep the output small
    sample_vintages = vintage_dates[-5:]
    print(f"Fetching data for the last {len(sample_vintages)} vintages...")

    all_series = []
    for vintage in sample_vintages:
        print(f"  Fetching data for vintage: {vintage}")
        df = data_manager.get_series_asof(series_key, vintage)
        # The 'asof_date' column represents the vintage date
        # The 'obs_date' column represents the reference period
        df = df.set_index(["asof_date", "obs_date"])["value"]
        df.index = df.index.set_levels(pd.to_datetime(df.index.levels[1]).to_period('Q'), level=1)
        all_series.append(df)

    # Concatenate all series into a single multi-indexed series
    result_series = pd.concat(all_series)

    print("\n--- US GDP Point-in-Time Data (Sample) ---")
    print(result_series)

    # You can also reshape the data to have vintages as columns
    print("\n--- US GDP Point-in-Time Data (Reshaped) ---")
    print(result_series.unstack(level=0))

    outpath = Path("outputs/get_usgdp")
    outpath.mkdir(exist_ok=True, parents=True
    result_series.to_csv(outpath / "us_gdp_point_in_time.csv")

if __name__ == "__main__":
    get_usgdp_point_in_time()

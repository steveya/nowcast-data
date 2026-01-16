import os
from datetime import date
from unittest import TestCase
from dotenv import load_dotenv
import pandas as pd

from nowcast_data.pit.api import PITDataManager
from nowcast_data.pit.core.catalog import SeriesCatalog

load_dotenv()


class TestUSGDP(TestCase):
    def setUp(self):
        if "FRED_API_KEY" not in os.environ:
            self.skipTest("FRED_API_KEY environment variable not set")

        catalog_path = os.path.join(os.path.dirname(__file__), "..", "series_catalog.yaml")
        self.catalog = SeriesCatalog(catalog_path)
        self.data_manager = PITDataManager(catalog=self.catalog)

    def test_get_usgdp_point_in_time(self):
        series_key = "US_GDP_SAAR"

        # 1. Get all vintage dates
        vintage_dates = self.data_manager.get_series_vintages(series_key)
        self.assertTrue(len(vintage_dates) > 0)

        # For demonstration, let's use a subset of vintages
        sample_vintages = vintage_dates[-5:]

        all_series = []
        for vintage in sample_vintages:
            df = self.data_manager.get_series_asof(series_key, vintage)
            # The 'asof_date' column represents the vintage date
            # The 'obs_date' column represents the reference period
            df = df.set_index(["asof_date", "obs_date"])["value"]
            all_series.append(df)

        # Concatenate all series into a single multi-indexed series
        result_series = pd.concat(all_series)

        # Check the structure of the result
        self.assertIsInstance(result_series, pd.Series)
        self.assertIsInstance(result_series.index, pd.MultiIndex)
        self.assertEqual(result_series.index.nlevels, 2)
        self.assertEqual(result_series.index.names, ["asof_date", "obs_date"])

        # Check that the number of unique vintage dates is what we expect
        self.assertEqual(len(result_series.index.get_level_values(0).unique()), len(sample_vintages))
        
        print("\n--- US GDP Point-in-Time Data ---")
        print(result_series)

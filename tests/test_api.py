import os
from datetime import date
from unittest import TestCase
from dotenv import load_dotenv
from nowcast_data.pit.api import PITDataManager
from nowcast_data.pit.core.catalog import SeriesCatalog

load_dotenv()

class TestApi(TestCase):
    def setUp(self):
        # This test requires a FRED_API_KEY environment variable to be set
        if "FRED_API_KEY" not in os.environ:
            self.skipTest("FRED_API_KEY environment variable not set")
        
        # Load the catalog
        catalog_path = os.path.join(os.path.dirname(__file__), '..', 'series_catalog.yaml')
        self.catalog = SeriesCatalog(catalog_path)
        
        # Create the data manager
        self.data_manager = PITDataManager(catalog=self.catalog)

    def test_get_series_asof_with_alphaforge_adapter(self):
        # Get the US_GDP_SAAR series, which is configured to use the alphaforge adapter
        df = self.data_manager.get_series_asof(
            series_key="US_GDP_SAAR",
            asof_date=date(2023, 1, 1),
            start=date(2022, 1, 1),
            end=date(2022, 12, 31),
        )

        # Assert that the DataFrame is not empty
        self.assertFalse(df.empty)

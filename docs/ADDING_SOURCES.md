# How to Add a New Source or Series

This guide explains how to extend the nowcast-data library with new data sources or add new series to existing sources.

## Adding a New Series to an Existing Source

### Example: Adding US Industrial Production to FRED

1. **Find the series identifier** from the data source (FRED series ID: INDPRO)

2. **Add entry to `series_catalog.yaml`**:

```yaml
US_INDPRO:
  country: US
  source: FRED_ALFRED
  source_series_id: INDPRO
  frequency: M
  seasonal_adjustment: SA
  units: Index 2017=100
  description: Industrial Production Index
  pit_mode: ALFRED_REALTIME
```

3. **That's it!** The series is now available:

```python
from nowcast_data.pit.api import PITDataManager
from nowcast_data.pit.core.catalog import SeriesCatalog

catalog = SeriesCatalog("series_catalog.yaml")
manager = PITDataManager(catalog)

df = manager.get_series_asof("US_INDPRO", date(2020, 1, 15))
```

### Series Configuration Fields

Required fields:
- `country`: ISO country code (US, EA, UK, CA, CH, JP)
- `source`: Adapter name (FRED_ALFRED, ECB_RTDB, etc.)
- `source_series_id`: Provider's series identifier
- `frequency`: D/W/M/Q/A
- `pit_mode`: ALFRED_REALTIME, DISCRETE_VINTAGES_SNAP, or NO_PIT

Optional fields:
- `seasonal_adjustment`: SA, NSA, SAAR
- `units`: Description of units
- `description`: Series description
- `transforms`: List of transformations to apply

## Adding a New Data Source Adapter

### Example: Implementing the ECB RTDB Adapter

Let's implement the ECB Real-Time Database adapter as an example.

#### Step 1: Understand the Data Source

Research the ECB RTDB:
- How are vintages organized?
- What API or file format is used?
- How frequently are vintages released?
- What authentication is needed?

For ECB RTDB:
- Vintages are released semi-annually
- Data available as CSV/XML downloads
- Each vintage is a complete snapshot
- No authentication required

#### Step 2: Implement the Adapter

Edit `nowcast_data/pit/adapters/ecb.py`:

```python
"""ECB Real-Time Database adapter for Euro Area data."""

from datetime import date
from typing import Optional, List
import requests
import pandas as pd

from nowcast_data.pit.adapters.base import PITAdapter
from nowcast_data.pit.core.models import PITObservation
from nowcast_data.pit.exceptions import (
    PITNotSupportedError,
    VintageNotFoundError,
    SourceFetchError,
)
from nowcast_data.pit.core.vintage_logic import select_vintage_for_asof


class ECBRTDBAdapter(PITAdapter):
    """Adapter for ECB Real-Time Database (Euro Area)."""
    
    BASE_URL = "https://www.ecb.europa.eu/stats/rtdb"
    
    def __init__(self):
        """Initialize ECB adapter."""
        self._session = requests.Session()
        self._vintage_cache = {}
    
    @property
    def name(self) -> str:
        return "ECB_RTDB"
    
    def supports_pit(self, series_id: str) -> bool:
        """Check if series has vintages."""
        try:
            vintages = self.list_vintages(series_id)
            return len(vintages) > 0
        except Exception:
            return False
    
    def list_vintages(self, series_id: str) -> List[date]:
        """
        List available vintage dates from ECB RTDB.
        
        Implementation: 
        - ECB releases vintages twice per year
        - Parse vintage list from ECB catalog
        """
        # TODO: Implement actual vintage list retrieval
        # For now, return example vintages
        return [
            date(2020, 1, 15),
            date(2020, 7, 15),
            date(2021, 1, 15),
            date(2021, 7, 15),
        ]
    
    def fetch_asof(
        self,
        series_id: str,
        asof_date: date,
        start: Optional[date] = None,
        end: Optional[date] = None
    ) -> List[PITObservation]:
        """
        Fetch observations as of asof_date.
        
        Implementation:
        1. Get available vintages
        2. Select latest vintage <= asof_date
        3. Download that vintage's CSV
        4. Parse and return observations
        """
        vintages = self.list_vintages(series_id)
        if not vintages:
            raise PITNotSupportedError(series_id)
        
        # Select appropriate vintage
        selected_vintage = select_vintage_for_asof(vintages, asof_date)
        if selected_vintage is None:
            raise VintageNotFoundError(series_id, asof_date)
        
        # Fetch data for selected vintage
        # TODO: Implement actual data fetching
        observations = []
        
        return observations
```

#### Step 3: Register the Adapter

The adapter is automatically available once implemented. Update `PITDataManager` if needed:

```python
# In nowcast_data/pit/api.py
self.adapters["ECB_RTDB"] = ECBRTDBAdapter()
```

#### Step 4: Add Series to Catalog

Add Euro Area series to `series_catalog.yaml`:

```yaml
EA_GDP:
  country: EA
  source: ECB_RTDB
  source_series_id: MNA.Q.Y.I8.W2.S1.S1.B.B1GQ._Z._Z._Z.EUR.V.N
  frequency: Q
  seasonal_adjustment: SA
  units: Million EUR
  description: Gross Domestic Product at market prices
  pit_mode: DISCRETE_VINTAGES_SNAP
```

#### Step 5: Write Tests

Create `tests/test_ecb_adapter.py`:

```python
import pytest
from datetime import date
from nowcast_data.pit.adapters.ecb import ECBRTDBAdapter

def test_adapter_name():
    adapter = ECBRTDBAdapter()
    assert adapter.name == "ECB_RTDB"

def test_list_vintages():
    adapter = ECBRTDBAdapter()
    vintages = adapter.list_vintages("test_series")
    assert isinstance(vintages, list)
    
# TODO: Add more tests with mocked responses
```

## Adapter Implementation Checklist

When implementing a new adapter, ensure:

- [ ] Implement all methods from `PITAdapter` base class:
  - [ ] `name` property
  - [ ] `supports_pit(series_id)`
  - [ ] `list_vintages(series_id)`
  - [ ] `fetch_asof(series_id, asof_date, start, end)`

- [ ] Handle errors properly:
  - [ ] Raise `PITNotSupportedError` for non-PIT series
  - [ ] Raise `VintageNotFoundError` when appropriate
  - [ ] Raise `SourceFetchError` for fetch failures

- [ ] Implement proper vintage selection:
  - [ ] Use `select_vintage_for_asof` for discrete vintages
  - [ ] Validate no lookahead bias

- [ ] Add retry logic with exponential backoff

- [ ] Implement caching where appropriate

- [ ] Parse provider-specific formats correctly

- [ ] Handle missing values consistently

- [ ] Add comprehensive tests

## Data Source Patterns

### Continuous Vintage (FRED-style)

For sources where you can query any date:

```python
def fetch_asof(self, series_id, asof_date, start, end):
    # Query API with asof_date parameter
    response = self._api_call(
        series_id=series_id,
        realtime_date=asof_date,
        obs_start=start,
        obs_end=end,
    )
    return self._parse_response(response, asof_date)
```

### Discrete Vintages (ECB-style)

For sources with periodic vintage snapshots:

```python
def fetch_asof(self, series_id, asof_date, start, end):
    # Get vintage list
    vintages = self.list_vintages(series_id)
    
    # Select vintage
    vintage_date = select_vintage_for_asof(vintages, asof_date)
    if vintage_date is None:
        raise VintageNotFoundError(series_id, asof_date)
    
    # Download vintage file
    data = self._download_vintage(series_id, vintage_date)
    
    # Filter to date range
    observations = self._parse_and_filter(data, start, end)
    
    return observations
```

### File-based Vintages

For sources that publish vintage files:

```python
def _get_vintage_file_url(self, series_id, vintage_date):
    """Construct URL for vintage file."""
    return f"{self.BASE_URL}/{series_id}/{vintage_date:%Y%m%d}.csv"

def _download_and_parse_vintage(self, url):
    """Download and parse CSV vintage file."""
    response = requests.get(url)
    df = pd.read_csv(io.StringIO(response.text))
    return df
```

## Testing New Adapters

### Unit Tests

Test adapter logic in isolation:

```python
def test_vintage_selection():
    adapter = MyAdapter()
    vintages = [date(2020, 1, 1), date(2020, 7, 1)]
    
    # Test selecting between vintages
    selected = adapter._select_vintage(vintages, date(2020, 3, 15))
    assert selected == date(2020, 1, 1)
```

### Integration Tests with Mocks

Test with mocked HTTP responses:

```python
@patch('requests.Session.get')
def test_fetch_asof_with_mock(mock_get):
    # Mock API response
    mock_get.return_value.json.return_value = {
        "observations": [
            {"date": "2019-12-31", "value": "21000.0"}
        ]
    }
    
    adapter = MyAdapter()
    obs = adapter.fetch_asof("TEST_SERIES", date(2020, 1, 15))
    
    assert len(obs) == 1
    assert obs[0].value == 21000.0
```

### Golden Tests

Test with real data (if permitted):

```python
@pytest.mark.integration
def test_fetch_real_data():
    """Test with actual API (requires credentials)."""
    adapter = MyAdapter(api_key=os.getenv("API_KEY"))
    
    obs = adapter.fetch_asof(
        "KNOWN_SERIES",
        date(2020, 1, 15),
        start=date(2019, 1, 1),
        end=date(2019, 12, 31),
    )
    
    # Verify against known values
    assert len(obs) == 4  # Quarterly data
```

## Best Practices

1. **Use existing patterns**: Follow the FRED adapter implementation

2. **Cache aggressively**: Vintage lists rarely change

3. **Handle errors gracefully**: Return meaningful error messages

4. **Validate inputs**: Check date ranges, series IDs

5. **Document formats**: Explain provider-specific quirks

6. **Test thoroughly**: Cover edge cases

7. **Rate limit**: Respect API limits

8. **Version metadata**: Store API version in provenance

## Getting Help

- Review existing adapters in `nowcast_data/pit/adapters/`
- Check the FRED adapter for a complete example
- See `PITAdapter` base class for required methods
- Read `docs/IMPLEMENTATION.md` for architecture details

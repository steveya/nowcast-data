"""Switzerland macro data adapter (SECO/SNB)."""

from datetime import date
from typing import Optional, List

from nowcast_data.pit.adapters.base import PITAdapter
from nowcast_data.pit.core.models import PITObservation, SeriesMetadata
from nowcast_data.pit.exceptions import PITNotSupportedError


class SwissAdapter(PITAdapter):
    """
    Adapter for Swiss macro data (SECO/SNB).
    
    Best-effort implementation - vintages may be limited.
    Stub implementation for now.
    """
    
    def __init__(self):
        """Initialize Swiss adapter."""
        # TODO: Add configuration for Swiss data sources
        pass
    
    @property
    def name(self) -> str:
        return "SWISS_SECO"
    
    def supports_pit(self, series_id: str) -> bool:
        """Check if series supports PIT - stub implementation."""
        # TODO: Implement actual check
        return False
    
    def list_vintages(self, series_id: str) -> List[date]:
        """
        List available vintage dates.
        
        Stub implementation.
        """
        # TODO: Implement vintage list if available
        raise PITNotSupportedError(
            series_id,
            "Swiss adapter not yet implemented"
        )
    
    def fetch_asof(
        self,
        series_id: str,
        asof_date: date,
        start: Optional[date] = None,
        end: Optional[date] = None,
        *,
        metadata: Optional[SeriesMetadata] = None,
    ) -> List[PITObservation]:
        """
        Fetch observations as of asof_date.
        
        Stub implementation.
        """
        # TODO: Implement actual fetch
        raise PITNotSupportedError(
            series_id,
            "Swiss adapter not yet implemented"
        )

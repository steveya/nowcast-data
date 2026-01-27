"""ECB Real-Time Database adapter for Euro Area data."""

from datetime import date
from typing import Optional, List

from nowcast_data.pit.adapters.base import PITAdapter
from nowcast_data.pit.core.models import PITObservation
from nowcast_data.pit.exceptions import PITNotSupportedError


class ECBRTDBAdapter(PITAdapter):
    """
    Adapter for ECB Real-Time Database (Euro Area).
    
    Stub implementation - to be completed with actual ECB RTDB integration.
    """
    
    def __init__(self):
        """Initialize ECB adapter."""
        # TODO: Add configuration for ECB RTDB access
        pass
    
    @property
    def name(self) -> str:
        return "ECB_RTDB"
    
    def supports_pit(self, series_id: str) -> bool:
        """Check if series supports PIT - stub implementation."""
        # TODO: Implement actual check against ECB RTDB catalog
        return False
    
    def list_vintages(self, series_id: str) -> List[date]:
        """
        List available vintage dates.
        
        Stub implementation - returns empty list.
        """
        # TODO: Implement actual vintage list retrieval from ECB RTDB
        raise PITNotSupportedError(
            series_id,
            "ECB RTDB adapter not yet implemented"
        )
    
    def fetch_asof(
        self,
        series_id: str,
        asof_date: date,
        start: Optional[date] = None,
        end: Optional[date] = None,
        *,
        metadata: Optional[object] = None,
    ) -> List[PITObservation]:
        """
        Fetch observations as of asof_date.
        
        Stub implementation.
        """
        # TODO: Implement actual fetch with discrete vintage selection
        raise PITNotSupportedError(
            series_id,
            "ECB RTDB adapter not yet implemented"
        )

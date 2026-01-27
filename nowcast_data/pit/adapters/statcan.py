"""Statistics Canada Real-Time Database adapter for Canadian data."""

from datetime import date
from typing import Optional, List

from nowcast_data.pit.adapters.base import PITAdapter
from nowcast_data.pit.core.models import PITObservation, SeriesMetadata
from nowcast_data.pit.exceptions import PITNotSupportedError


class StatCanRealTimeAdapter(PITAdapter):
    """
    Adapter for Statistics Canada Real-Time Tables.
    
    Stub implementation - to be completed with actual StatCan integration.
    """
    
    def __init__(self):
        """Initialize StatCan adapter."""
        # TODO: Add configuration for StatCan real-time tables
        pass
    
    @property
    def name(self) -> str:
        return "STATCAN_REALTIME"
    
    def supports_pit(self, series_id: str) -> bool:
        """Check if series supports PIT - stub implementation."""
        # TODO: Implement actual check against StatCan catalog
        return False
    
    def list_vintages(self, series_id: str) -> List[date]:
        """
        List available vintage dates.
        
        Stub implementation.
        """
        # TODO: Implement actual vintage list from StatCan
        raise PITNotSupportedError(
            series_id,
            "StatCan adapter not yet implemented"
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
        # TODO: Implement actual fetch with discrete vintage selection
        raise PITNotSupportedError(
            series_id,
            "StatCan adapter not yet implemented"
        )

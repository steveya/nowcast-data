"""Bank of England Real-Time Database adapter for UK data."""

from datetime import date
from typing import Optional, List

from nowcast_data.pit.adapters.base import PITAdapter
from nowcast_data.pit.core.models import PITObservation, SeriesMetadata
from nowcast_data.pit.exceptions import PITNotSupportedError


class BOERTDBAdapter(PITAdapter):
    """
    Adapter for Bank of England Real-Time Database (UK GDP).
    
    Stub implementation - to be completed with actual BoE RTDB integration.
    """
    
    def __init__(self):
        """Initialize BoE adapter."""
        # TODO: Add configuration for BoE RTDB access
        pass
    
    @property
    def name(self) -> str:
        return "BOE_RTDB"
    
    def supports_pit(self, series_id: str) -> bool:
        """Check if series supports PIT - stub implementation."""
        # TODO: Implement actual check against BoE RTDB catalog
        return False
    
    def list_vintages(self, series_id: str) -> List[date]:
        """
        List available vintage dates.
        
        Stub implementation.
        """
        # TODO: Implement actual vintage list retrieval from BoE RTDB
        raise PITNotSupportedError(
            series_id,
            "BoE RTDB adapter not yet implemented"
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
            "BoE RTDB adapter not yet implemented"
        )

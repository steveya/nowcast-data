"""Base adapter interface for PIT data sources."""

from abc import ABC, abstractmethod
from datetime import date
from typing import Optional, List
import pandas as pd

from nowcast_data.pit.core.models import PITObservation


class PITAdapter(ABC):
    """Base class for point-in-time data adapters."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Adapter name/identifier."""
        pass
    
    @abstractmethod
    def supports_pit(self, series_id: str) -> bool:
        """
        Check if a series supports point-in-time retrieval.
        
        Args:
            series_id: Source-specific series identifier
            
        Returns:
            True if PIT is supported, False otherwise
        """
        pass
    
    @abstractmethod
    def list_vintages(self, series_id: str) -> List[date]:
        """
        List available vintage dates for a series.
        
        Args:
            series_id: Source-specific series identifier
            
        Returns:
            List of vintage dates (sorted)
            
        Raises:
            PITNotSupportedError: If series doesn't support vintages
            SourceFetchError: If fetching fails
        """
        pass
    
    @abstractmethod
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
        Fetch observations as they were known on asof_date.
        
        Args:
            series_id: Source-specific series identifier
            asof_date: Point-in-time evaluation date
            start: Optional start date for observation period
            end: Optional end date for observation period
            
        Returns:
            List of PIT observations
            
        Raises:
            PITNotSupportedError: If series doesn't support PIT
            VintageNotFoundError: If no vintage available at asof_date
            SourceFetchError: If fetching fails
        """
        pass
    
    def fetch_vintage(
        self,
        series_id: str,
        vintage_date: date,
        start: Optional[date] = None,
        end: Optional[date] = None
    ) -> List[PITObservation]:
        """
        Fetch a specific vintage of observations.
        
        Optional method - not all adapters may support explicit vintage fetching.
        Default implementation uses fetch_asof with vintage_date as asof_date.
        
        Args:
            series_id: Source-specific series identifier
            vintage_date: Specific vintage date to fetch
            start: Optional start date for observation period
            end: Optional end date for observation period
            
        Returns:
            List of PIT observations
        """
        return self.fetch_asof(series_id, vintage_date, start, end)

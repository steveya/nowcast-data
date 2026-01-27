"""Base adapter interface for PIT data sources."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date
from typing import Optional, List, TYPE_CHECKING
import pandas as pd

from nowcast_data.pit.core.models import PITObservation, SeriesMetadata

if TYPE_CHECKING:
    from alphaforge.time.ref_period import RefPeriod, RefFreq


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
        metadata: Optional[SeriesMetadata] = None,
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

    def fetch_asof_ref(
        self,
        series_id: str,
        asof_date: date,
        start_ref: str | "RefPeriod" | None = None,
        end_ref: str | "RefPeriod" | None = None,
        *,
        freq: Optional["RefFreq"] = None,
        metadata: Optional[SeriesMetadata] = None,
    ) -> List[PITObservation]:
        """Optional ref-period snapshot query."""
        raise NotImplementedError("Ref-period snapshot queries not supported.")

    def fetch_revisions_ref(
        self,
        series_id: str,
        ref: str | "RefPeriod",
        start_asof: Optional[date] = None,
        end_asof: Optional[date] = None,
        *,
        freq: Optional["RefFreq"] = None,
        metadata: Optional[SeriesMetadata] = None,
    ) -> pd.Series:
        """Optional ref-period revision timeline query."""
        raise NotImplementedError("Ref-period revision queries not supported.")
    
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

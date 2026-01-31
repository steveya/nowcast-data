"""Series catalog configuration management."""

import yaml
from pathlib import Path
from typing import Dict, Optional, List
from nowcast_data.pit.core.models import SeriesMetadata, PITMode


class SeriesCatalog:
    """Manages series metadata from configuration."""

    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize catalog from configuration file.

        Args:
            config_path: Path to YAML configuration file. If None, uses default.
        """
        self._metadata: Dict[str, SeriesMetadata] = {}
        if config_path:
            self.load(config_path)

    def load(self, config_path: Path) -> None:
        """Load series metadata from YAML configuration."""
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        if not config:
            return

        for series_key, series_config in config.items():
            metadata = SeriesMetadata(
                series_key=series_key,
                country=series_config.get("country", ""),
                source=series_config.get("source", ""),
                source_series_id=series_config.get("source_series_id", ""),
                frequency=series_config.get("frequency", ""),
                pit_mode=series_config.get("pit_mode", "NO_PIT"),
                seasonal_adjustment=series_config.get("seasonal_adjustment"),
                units=series_config.get("units"),
                description=series_config.get("description"),
                transforms=series_config.get("transforms"),
                adapter=series_config.get("adapter"),
                obs_date_anchor=series_config.get("obs_date_anchor"),
            )
            self._metadata[series_key] = metadata

    def get(self, series_key: str) -> Optional[SeriesMetadata]:
        """Get metadata for a series."""
        return self._metadata.get(series_key)

    def get_all(self) -> Dict[str, SeriesMetadata]:
        """Get all series metadata."""
        return self._metadata.copy()

    def add(self, metadata: SeriesMetadata) -> None:
        """Add or update series metadata."""
        self._metadata[metadata.series_key] = metadata

    def list_series(self, country: Optional[str] = None, source: Optional[str] = None) -> List[str]:
        """
        List series keys, optionally filtered.

        Args:
            country: Filter by country code
            source: Filter by source name

        Returns:
            List of series keys
        """
        series = []
        for key, meta in self._metadata.items():
            if country and meta.country != country:
                continue
            if source and meta.source != source:
                continue
            series.append(key)
        return sorted(series)

    def supports_pit(self, series_key: str) -> bool:
        """Check if a series supports PIT retrieval."""
        meta = self.get(series_key)
        if not meta:
            return False
        return meta.pit_mode != "NO_PIT"

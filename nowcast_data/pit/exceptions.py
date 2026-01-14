"""Custom exceptions for PIT data operations."""


class PITError(Exception):
    """Base exception for PIT-related errors."""

    pass


class PITNotSupportedError(PITError):
    """Raised when a series does not support point-in-time retrieval."""

    def __init__(self, series_key: str, message: str = None):
        self.series_key = series_key
        if message is None:
            message = f"Series '{series_key}' does not support point-in-time retrieval"
        super().__init__(message)


class VintageNotFoundError(PITError):
    """Raised when no vintage is available for the requested asof date."""

    def __init__(self, series_key: str, asof_date, message: str = None):
        self.series_key = series_key
        self.asof_date = asof_date
        if message is None:
            message = (
                f"No vintage available for series '{series_key}' "
                f"at or before asof_date={asof_date}"
            )
        super().__init__(message)


class SourceFetchError(PITError):
    """Raised when data fetching from a source fails."""

    def __init__(self, source: str, message: str = None, original_error: Exception = None):
        self.source = source
        self.original_error = original_error
        if message is None:
            message = f"Failed to fetch data from source '{source}'"
            if original_error:
                message += f": {str(original_error)}"
        super().__init__(message)

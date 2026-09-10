"""Exception hierarchy for fail-loud error handling across the package.

Every exception here signals a condition that must stop execution rather
than degrade into a silent NaN or a default value — see CLAUDE.md's
fail-loud principle.
"""

from __future__ import annotations


class ForecastLensError(Exception):
    """Base class for all forecastlens-specific exceptions."""


class InvalidForecastResultError(ForecastLensError):
    """Raised when a ``ForecastResult`` is constructed with inconsistent data."""


class MissingQuantilesError(ForecastLensError):
    """Raised when an operation requires quantiles but only a point forecast is available."""


class UnrecognizedAdapterFormatError(ForecastLensError):
    """Raised when an adapter receives a framework output it does not know how to normalize."""

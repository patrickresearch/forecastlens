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


class InsufficientDataError(ForecastLensError):
    """Raised when there is not enough data to produce a single valid split/window."""


class LeakageError(ForecastLensError):
    """Raised when a computed split/window would let training data see future data.

    This is a hard defensive assertion, not a normal user-facing validation
    error -- it should be geometrically impossible to trigger for valid
    parameters and exists to catch bugs in the splitting logic itself.
    """


class InvalidGeneratorConfigError(ForecastLensError):
    """Raised when a synthetic generator is configured with inconsistent parameters."""

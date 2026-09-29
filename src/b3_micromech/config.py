"""Configuration errors.

Typed solver configuration is added with the strict loader. ``ConfigError`` lives
here so material loading can reject duplicate names before that loader lands.
"""


class ConfigError(ValueError):
    """Raised for unknown keys, bad enum values, or out-of-range numbers."""

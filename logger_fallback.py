"""Logger plumbing for the plugin.

Production: callers pass AstrBot's framework logger (``Star.logger``) down to
every component, which is the only logger the plugin review rules allow.

Local unit tests: AstrBot is not installed, so this module provides a silent
no-op fallback object.  The fallback never prints anything and never imports
Python's standard-library logging component.
"""

from __future__ import annotations

try:  # pragma: no cover - AstrBot 运行时一定存在
    from astrbot.api import logger as _FRAMEWORK_LOGGER
except ImportError:  # 本地单测未安装 AstrBot
    _FRAMEWORK_LOGGER = None


class NullLogger:
    """Silent logger used only when no framework logger is injected."""

    def debug(self, *args, **kwargs):
        return None

    def info(self, *args, **kwargs):
        return None

    def warning(self, *args, **kwargs):
        return None

    def error(self, *args, **kwargs):
        return None

    def exception(self, *args, **kwargs):
        return None

    def critical(self, *args, **kwargs):
        return None


def fallback_logger():
    """Return AstrBot's framework logger, or a silent placeholder in tests."""

    return _FRAMEWORK_LOGGER or NullLogger()

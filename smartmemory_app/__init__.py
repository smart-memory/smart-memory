"""SmartMemory Claude Code plugin — zero-infra persistent memory."""

from importlib.metadata import version as _pkg_version, PackageNotFoundError

try:
    __version__ = _pkg_version("smartmemory")
except PackageNotFoundError:
    __version__ = "dev"

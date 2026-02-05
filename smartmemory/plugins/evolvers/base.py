"""Backwards-compatible alias for the canonical EvolverPlugin base class.

All evolvers should inherit from ``EvolverPlugin`` (defined in
``smartmemory.plugins.base``).  The ``Evolver`` name is kept here so that
existing ``from .base import Evolver`` imports continue to work.
"""

from smartmemory.plugins.base import EvolverPlugin

# Deprecated: use EvolverPlugin directly.
Evolver = EvolverPlugin

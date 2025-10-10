"""
Library-level default store registrations.
Importing this module ensures built-in backends are registered.
"""
# Register JSON backend by importing its module for side-effects
from . import json_store  # noqa: F401  # registers "json"

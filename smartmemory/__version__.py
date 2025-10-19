"""
SmartMemory version information.

Version is read dynamically from package metadata.
Single source of truth: pyproject.toml
"""

try:
    # For installed package, read from metadata
    from importlib.metadata import version
    __version__ = version("smartmemory")
except Exception:
    # For development, read from pyproject.toml
    try:
        import tomllib  # Python 3.11+
    except ImportError:
        try:
            import tomli as tomllib  # Python 3.10
        except ImportError:
            # Fallback if no TOML parser available
            __version__ = "0.1.6"
            __version_info__ = (0, 1, 6)
        else:
            from pathlib import Path
            pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
            with open(pyproject_path, "rb") as f:
                pyproject = tomllib.load(f)
            __version__ = pyproject["project"]["version"]
    else:
        from pathlib import Path
        pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
        with open(pyproject_path, "rb") as f:
            pyproject = tomllib.load(f)
        __version__ = pyproject["project"]["version"]

# Parse version info
try:
    __version_info__ = tuple(int(x) for x in __version__.split("."))
except:
    __version_info__ = (0, 1, 6)

# Version history:
# 0.1.8 - README overhaul: fixed all code snippets to use public API, verified evolvers, added "In Progress" section, removed internal imports
# 0.1.7 - Updated README, removed ChromaDB references, fixed PyPI deployment
# 0.1.6 - Production PyPI deployment setup
# 0.1.5 - Complete bi-temporal implementation: version tracking, temporal search, relationship queries, bi-temporal joins, performance optimizations
# 0.1.4 - Bi-temporal queries: time-travel, audit trails, version history, rollback
# 0.1.3 - Zettelkasten system with wikilink support, documentation, examples, CLI
# 0.1.2 - ChromaDB optional, Python 3.10+ requirement, version externalized
# 0.1.1 - Plugin system with security

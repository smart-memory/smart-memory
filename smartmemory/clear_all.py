"""
Utility to clear SmartMemory state across caches and stores.

Actions:
- Clear SmartGraph (drops all nodes/edges, clears graph caches)
- Clear VectorStore (if supported by current backend)
- Clear Redis-backed extraction caches (best effort)

Usage:
  python -m smartmemory.maintenance.clear_all --all
  python -m smartmemory.maintenance.clear_all --graph
  python -m smartmemory.maintenance.clear_all --vector
  python -m smartmemory.maintenance.clear_all --cache

Notes:
- This is intended for local/dev use. Use with caution in shared environments.
- We do not swallow exceptions silently: failures are logged and surfaced.
"""
from __future__ import annotations

import argparse
import logging
import os
from typing import Any


def _setup_logging(verbose: bool):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="[%(levelname)s] %(message)s")


def clear_graph() -> bool:
    from smartmemory.graph.smartgraph import SmartGraph

    logging.info("Clearing SmartGraph (nodes/edges/caches)...")
    try:
        g = SmartGraph()
        ok = g.clear()
        logging.info("SmartGraph cleared: %s", ok)
        return bool(ok)
    except Exception as e:
        logging.error("Failed to clear SmartGraph: %s", e)
        raise


def clear_vector_store() -> bool:
    try:
        from smartmemory.stores.vector.vector_store import VectorStore
    except Exception as e:
        logging.warning("VectorStore not available: %s", e)
        return False

    logging.info("Clearing VectorStore (if backend supports clear())...")
    try:
        vs = VectorStore()
        ok = vs.clear()
        logging.info("VectorStore cleared via clear(): %s", ok)
        return bool(ok)
    except Exception as e:
        logging.error("Failed to clear VectorStore: %s", e)
        raise


def clear_cache() -> bool:
    try:
        from smartmemory.utils.cache import get_cache
    except Exception as e:
        logging.warning("Cache utils not available: %s", e)
        return False

    logging.info("Clearing Redis-backed caches (best effort)...")
    try:
        cache = get_cache()
        # Try to get redis client from cache object
        client = getattr(cache, 'redis', None)
        if client and hasattr(client, "flushdb"):
            client.flushdb()
            logging.info("Redis FLUSHDB successful")
            return True
        # Extract connection params if available
        url = os.getenv("REDIS_URL") or os.getenv("REDIS_URI")
        if url:
            try:
                import redis  # type: ignore
                r = redis.from_url(url)
                r.flushdb()
                logging.info("Redis FLUSHDB via REDIS_URL successful")
                return True
            except Exception as e:
                logging.warning("Redis module/connection failed: %s", e)
        logging.warning("Could not locate a Redis client to flush; skipped.")
        return False
    except Exception as e:
        logging.error("Failed to clear caches: %s", e)
        raise


def main():
    parser = argparse.ArgumentParser(description="Clear SmartMemory state across caches and stores.")
    parser.add_argument("--all", action="store_true", help="Clear graph, vector store, and caches")
    parser.add_argument("--graph", action="store_true", help="Clear SmartGraph")
    parser.add_argument("--vector", action="store_true", help="Clear VectorStore")
    parser.add_argument("--cache", action="store_true", help="Clear Redis-backed caches")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    _setup_logging(args.verbose)

    if not any([args.all, args.graph, args.vector, args.cache]):
        logging.info("No flags provided; defaulting to --all")
        args.all = True

    results: dict[str, Any] = {}

    try:
        if args.all or args.graph:
            results["graph"] = clear_graph()
        if args.all or args.vector:
            results["vector"] = clear_vector_store()
        if args.all or args.cache:
            results["cache"] = clear_cache()
    except Exception as e:
        logging.error("One or more clear operations failed: %s", e)
        raise

    logging.info("Done. Summary: %s", results)


if __name__ == "__main__":
    main()

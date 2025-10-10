import logging
from typing import Any, List, Dict

logger = logging.getLogger(__name__)


def summarize_episodes(db, summarize_episodes_query, n: int = 5) -> List[Dict[str, Any]]:
    """
    Summarize the top N episodes (or equivalent nodes) using a provided query generator.
    - db: database interface with execute_read(cypher, **params)
    - summarize_episodes_query: function() -> (cypher, params)
    - n: number of episodes to summarize
    """
    cypher, params = summarize_episodes_query()
    result = db.execute_read(cypher, n=n, **params)
    return [{"name": r["name"], "description": r["description"], "related_count": r["rel_count"]} for r in result]


def get_most_connected_episodes(db, get_most_connected_episodes_query, n: int = 5) -> List[Dict[str, Any]]:
    """
    Get the most connected episodes (or equivalent nodes) using a provided query generator.
    - db: database interface with execute_read(cypher, **params)
    - get_most_connected_episodes_query: function() -> (cypher, params)
    - n: number of episodes to retrieve
    """
    cypher, params = get_most_connected_episodes_query()
    result = db.execute_read(cypher, n=n, **params)
    return [{"name": r["name"], "description": r["description"], "related_count": r["rel_count"]} for r in result]

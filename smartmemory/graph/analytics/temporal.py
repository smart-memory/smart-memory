import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


def analyze_temporal_patterns(db, analyze_temporal_bins, analyze_temporal_recency, reference_time_field: str = "reference_time", granularity: str = "day") -> Dict[str, Any]:
    """
    Analyze temporal patterns for nodes with a given reference time field.
    - db: database interface with execute_read(cypher) method
    - analyze_temporal_bins: function(date_expr) -> (cypher, params)
    - analyze_temporal_recency: function() -> (cypher, params)
    - reference_time_field: name of the field storing timestamps
    - granularity: 'day', 'week', or 'month'
    """
    if granularity not in ("day", "week", "month"):
        raise ValueError("granularity must be 'day', 'week', or 'month'")
    if granularity == "day":
        date_expr = f"date(e.{reference_time_field})"
    elif granularity == "week":
        date_expr = f"apoc.date.format(apoc.date.parse(e.{reference_time_field}, 'ms', 'yyyy-MM-dd''T''HH:mm:ssZ'), 'ms', 'YYYY-ww')"
    else:
        date_expr = f"apoc.date.format(apoc.date.parse(e.{reference_time_field}, 'ms', 'yyyy-MM-dd''T''HH:mm:ssZ'), 'ms', 'YYYY-MM')"
    cypher, params = analyze_temporal_bins(date_expr)
    result = db.execute_read(cypher, **params)
    bins = [{"period": r["period"], "count": r["cnt"]} for r in result]
    cypher2, params2 = analyze_temporal_recency()
    recency = db.execute_read(cypher2, **params2)
    recency_stats = recency[0] if recency else {}
    return {"bins": bins, "recency": recency_stats}

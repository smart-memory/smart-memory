"""
Cypher query core for EpisodicMemory operations.
Each function returns a (cypher_string, param_names) tuple.
"""


def match_linked_episodes():
    return (
        "MATCH (a:Episode {name: $src})-[:RELATED]->(b:Episode) RETURN b.name AS linked_name",
        ["src"]
    )


def create_related_edge():
    return (
        "MATCH (a:Episode {name: $src}), (b:Episode {name: $dst}) MERGE (a)-[:RELATED]->(b)",
        ["src", "dst"]
    )


def get_recent_episodes():
    return (
        "MATCH (e:Episode) RETURN e.name AS name, e.source_description AS description, e.reference_time AS time "
        "ORDER BY e.reference_time DESC LIMIT $n",
        ["n"]
    )


def get_most_connected_episodes():
    return (
        "MATCH (e:Episode)-[r:RELATED]->() WITH e, count(r) AS rel_count RETURN e.name AS name, e.source_description AS description, rel_count ORDER BY rel_count DESC LIMIT $n",
        ["n"]
    )


def summarize_episodes():
    return (
        "MATCH (e:Episode) OPTIONAL MATCH (e)-[r:RELATED]->() WITH e, count(r) AS rel_count RETURN e.name AS name, e.source_description AS description, rel_count ORDER BY "
        "rel_count DESC, e.reference_time DESC LIMIT $n",
        ["n"]
    )


def get_episode_facts():
    return (
        "MATCH (e:Episode {name: $name})-[r]->(target) RETURN type(r) AS edge_type, target.name AS target_name",
        ["name"]
    )


def analyze_temporal_bins(date_expr):
    cypher = f"MATCH (e:Episode) WITH {date_expr} AS period, count(*) AS cnt RETURN period, cnt ORDER BY period DESC"
    return cypher, []


def analyze_temporal_recency():
    return (
        "MATCH (e:Episode) RETURN max(e.reference_time) AS newest, min(e.reference_time) AS oldest, count(*) AS total",
        []
    )


def edge_type_breakdown():
    return (
        "MATCH ()-[r]->() RETURN type(r) AS edge_type, count(*) AS cnt ORDER BY cnt DESC",
        []
    )


def match_recent_episodes():
    return (
        "MATCH (e:Episode) RETURN e.name AS name, e.source_description AS description "
        "ORDER BY e.reference_time DESC LIMIT $n",
        ["n"]
    )


def match_most_connected_episodes():
    return (
        "MATCH (e:Episode)-[r:RELATED]->() "
        "RETURN e.name AS name, e.source_description AS description, count(r) AS degree "
        "ORDER BY degree DESC LIMIT $n",
        ["n"]
    )


def match_episode_facts():
    return (
        "MATCH (e:Episode {name: $episode_name})-[r]->(t) "
        "RETURN type(r) AS edge_type, t.name AS target, r AS edge",
        ["episode_name"]
    )


def match_temporal_bins(granularity):
    # granularity: 'day', 'week', or 'month'
    if granularity == "day":
        bin_expr = "date(e.reference_time)"
    elif granularity == "week":
        bin_expr = "date.truncate('week', date(e.reference_time))"
    elif granularity == "month":
        bin_expr = "date.truncate('month', date(e.reference_time))"
    else:
        raise ValueError(f"Invalid granularity: {granularity}")
    cypher = (
        f"MATCH (e:Episode) RETURN {bin_expr} AS bin, count(*) AS count ORDER BY bin DESC"
    )
    return cypher, []


def match_edge_type_breakdown():
    return (
        "MATCH (e:Episode)-[r]->() RETURN type(r) AS edge_type, count(*) AS count ORDER BY count DESC",
        []
    )


def match_export_graph(subset):
    if subset:
        match_clause = "MATCH (e:Episode) WHERE e.name IN $names "
    else:
        match_clause = "MATCH (e:Episode) "
    cypher = (
            match_clause +
            "OPTIONAL MATCH (e)-[r]->(t) "
            "RETURN e.name AS source, t.name AS target, type(r) AS edge_type, e.source_description AS description"
    )
    return cypher, (["names"] if subset else [])


def match_prune_by_age():
    return (
        "MATCH (e:Episode) WHERE datetime(e.reference_time) < datetime($cutoff) RETURN e.name AS name",
        ["cutoff"]
    )


def match_prune_by_degree():
    return (
        "MATCH (e:Episode) OPTIONAL MATCH (e)-[r]->() WITH e, count(r) AS deg WHERE deg < $min_degree RETURN e.name AS name",
        ["min_degree"]
    )


def delete_episodes():
    return (
        "MATCH (e:Episode) WHERE e.name IN $names DETACH DELETE e",
        ["names"]
    )

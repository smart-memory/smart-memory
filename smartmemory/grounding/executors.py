from typing import Dict, Any

from .schemas import (
    OntologyGroundingConfig,
    KnowledgeBaseGroundingConfig,
    CommonsenseGroundingConfig,
    CausalGroundingConfig,
    WikipediaGroundingConfig,
)

SCHEMA_VERSION = "grounding@v1"


def _result_stub(kind: str) -> Dict[str, Any]:
    return {
        "success": True,
        "artifacts": [],
        "metrics": {"stage": kind, "processed": 0},
        "schema_version": SCHEMA_VERSION,
    }


def run_ontology_grounding(run: Dict, cfg: OntologyGroundingConfig) -> Dict[str, Any]:
    return _result_stub("ontology_grounding")


def run_knowledge_base_grounding(run: Dict, cfg: KnowledgeBaseGroundingConfig) -> Dict[str, Any]:
    return _result_stub("knowledge_base")


def run_commonsense_grounding(run: Dict, cfg: CommonsenseGroundingConfig) -> Dict[str, Any]:
    return _result_stub("commonsense_grounding")


def run_causal_grounding(run: Dict, cfg: CausalGroundingConfig) -> Dict[str, Any]:
    return _result_stub("causal_grounding")


def run_wikipedia_grounding(run: Dict, cfg: WikipediaGroundingConfig) -> Dict[str, Any]:
    return _result_stub("wikipedia_grounding")

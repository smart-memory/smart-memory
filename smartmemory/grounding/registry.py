from typing import Any, Callable, Dict, Tuple

from .executors import (
    run_ontology_grounding,
    run_knowledge_base_grounding,
    run_commonsense_grounding,
    run_causal_grounding,
    run_wikipedia_grounding,
)
from .schemas import (
    OntologyGroundingConfig,
    KnowledgeBaseGroundingConfig,
    CommonsenseGroundingConfig,
    CausalGroundingConfig,
    WikipediaGroundingConfig,
)

GROUNDING_REGISTRY: Dict[str, Tuple[Any, Callable]] = {
    "ontology_grounding": (OntologyGroundingConfig, run_ontology_grounding),
    "knowledge_base": (KnowledgeBaseGroundingConfig, run_knowledge_base_grounding),
    "commonsense_grounding": (CommonsenseGroundingConfig, run_commonsense_grounding),
    "causal_grounding": (CausalGroundingConfig, run_causal_grounding),
    "wikipedia_grounding": (WikipediaGroundingConfig, run_wikipedia_grounding),
}

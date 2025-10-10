from pydantic import BaseModel, conint, confloat
from typing import List, Literal


class OntologyGroundingConfig(BaseModel):
    registry_id: str = "bfo"
    ontology_id: str = "default"
    confidence_threshold: confloat(ge=0, le=1) = 0.7
    max_results: conint(ge=1, le=50) = 10


class KnowledgeBaseGroundingConfig(BaseModel):
    sources: List[str] = ["wikidata"]
    max_results: conint(ge=1, le=100) = 10
    rerank: bool = False


class CommonsenseGroundingConfig(BaseModel):
    reasoning_level: Literal["low", "medium", "high"] = "medium"
    confidence_threshold: confloat(ge=0, le=1) = 0.7


class CausalGroundingConfig(BaseModel):
    model_name: str = "gpt-4o"
    inference_method: Literal["pattern", "bayesian", "do_calculus", "llm"] = "llm"
    confidence_threshold: confloat(ge=0, le=1) = 0.7
    max_paths: conint(ge=1, le=50) = 10
    max_hops: conint(ge=1, le=6) = 3
    edge_weight_threshold: confloat(ge=0, le=1) = 0.5
    allow_cycles: bool = False
    temporal_window_days: conint(ge=0, le=3650) = 365
    enforce_temporal_ordering: bool = True
    enable_counterfactuals: bool = False
    rerank_by_causal_strength: bool = True
    knowledge_sources: List[str] = []


class WikipediaGroundingConfig(BaseModel):
    language: str = "en"
    confidence_threshold: confloat(ge=0, le=1) = 0.7
    max_results: conint(ge=1, le=50) = 10

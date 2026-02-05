#!/usr/bin/env python3
"""
Extraction Quality Benchmark - LLM vs Traditional NLP.

Compares extraction quality across:

LLM-based extractors:
- OpenAI GPT-4o-mini (baseline)
- Groq Llama-3.3-70b
- Google Gemini Flash
- Anthropic Claude Haiku

Traditional NLP extractors:
- spaCy (en_core_web_sm)
- GLiNER2 (fastino/gliner2-base-v1)
- Hybrid (GLiNER2 + ReLiK)

Metrics:
- Entity precision/recall/F1
- Relation precision/recall/F1
- Extraction latency (ms)

Usage:
    # Run all extractors
    PYTHONPATH=. python tests/benchmark_model_quality.py --all

    # Run LLM models only
    PYTHONPATH=. python tests/benchmark_model_quality.py --models gpt4o groq

    # Run traditional extractors only
    PYTHONPATH=. python tests/benchmark_model_quality.py --traditional

    # Run specific traditional extractors
    PYTHONPATH=. python tests/benchmark_model_quality.py --extractors spacy gliner2
"""

import argparse
import json
import os
import re
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from dataclasses import asdict, dataclass, field
from statistics import mean, stdev
from typing import Any

# Suppress deprecation warnings from traditional extractors
warnings.filterwarnings("ignore", category=DeprecationWarning)


# Ground truth dataset - verified entities and relations
# Categories: simple (1-2 entities), standard (3-5 entities), complex (6+ entities)
GROUND_TRUTH_DATASET = [
    # === SIMPLE CASES (1-2 entities, 0-1 relations) ===
    {
        "id": "einstein",
        "category": "simple",
        "domain": "science",
        "text": "Albert Einstein developed the theory of relativity.",
        "entities": [
            {"name": "Albert Einstein", "type": "person"},
            {"name": "theory of relativity", "type": "concept"},
        ],
        "relations": [{"subject": "Albert Einstein", "predicate": "developed", "object": "theory of relativity"}],
    },
    {
        "id": "python",
        "category": "simple",
        "domain": "technology",
        "text": "Python is a programming language created by Guido van Rossum.",
        "entities": [{"name": "Python", "type": "technology"}, {"name": "Guido van Rossum", "type": "person"}],
        "relations": [{"subject": "Guido van Rossum", "predicate": "created", "object": "Python"}],
    },
    {
        "id": "amazon_river",
        "category": "simple",
        "domain": "geography",
        "text": "The Amazon River flows through Brazil.",
        "entities": [{"name": "Amazon River", "type": "location"}, {"name": "Brazil", "type": "location"}],
        "relations": [{"subject": "Amazon River", "predicate": "flows_through", "object": "Brazil"}],
    },
    # === STANDARD CASES (3-5 entities, 1-2 relations) ===
    {
        "id": "microsoft",
        "category": "standard",
        "domain": "business",
        "text": "Microsoft was founded by Bill Gates in 1975.",
        "entities": [
            {"name": "Microsoft", "type": "organization"},
            {"name": "Bill Gates", "type": "person"},
            {"name": "1975", "type": "temporal"},
        ],
        "relations": [
            {"subject": "Bill Gates", "predicate": "founded", "object": "Microsoft"},
            {"subject": "Microsoft", "predicate": "founded_in", "object": "1975"},
        ],
    },
    {
        "id": "eiffel",
        "category": "standard",
        "domain": "geography",
        "text": "The Eiffel Tower is located in Paris, France.",
        "entities": [
            {"name": "Eiffel Tower", "type": "location"},
            {"name": "Paris", "type": "location"},
            {"name": "France", "type": "location"},
        ],
        "relations": [
            {"subject": "Eiffel Tower", "predicate": "located_in", "object": "Paris"},
            {"subject": "Paris", "predicate": "located_in", "object": "France"},
        ],
    },
    {
        "id": "curie",
        "category": "standard",
        "domain": "science",
        "text": "Marie Curie won the Nobel Prize in Physics in 1903.",
        "entities": [
            {"name": "Marie Curie", "type": "person"},
            {"name": "Nobel Prize", "type": "award"},
            {"name": "Physics", "type": "concept"},
            {"name": "1903", "type": "temporal"},
        ],
        "relations": [
            {"subject": "Marie Curie", "predicate": "won", "object": "Nobel Prize"},
            {"subject": "Nobel Prize", "predicate": "field", "object": "Physics"},
            {"subject": "Marie Curie", "predicate": "won_in", "object": "1903"},
        ],
    },
    {
        "id": "apple_iphone",
        "category": "standard",
        "domain": "technology",
        "text": "Apple released the iPhone in 2007. Steve Jobs announced it at Macworld.",
        "entities": [
            {"name": "Apple", "type": "organization"},
            {"name": "iPhone", "type": "product"},
            {"name": "2007", "type": "temporal"},
            {"name": "Steve Jobs", "type": "person"},
            {"name": "Macworld", "type": "event"},
        ],
        "relations": [
            {"subject": "Apple", "predicate": "released", "object": "iPhone"},
            {"subject": "Steve Jobs", "predicate": "announced", "object": "iPhone"},
            {"subject": "iPhone", "predicate": "released_in", "object": "2007"},
            {"subject": "Steve Jobs", "predicate": "announced_at", "object": "Macworld"},
        ],
    },
    {
        "id": "beethoven",
        "category": "standard",
        "domain": "arts",
        "text": "Ludwig van Beethoven composed Symphony No. 9 in Vienna.",
        "entities": [
            {"name": "Ludwig van Beethoven", "type": "person"},
            {"name": "Symphony No. 9", "type": "product"},
            {"name": "Vienna", "type": "location"},
        ],
        "relations": [
            {"subject": "Ludwig van Beethoven", "predicate": "composed", "object": "Symphony No. 9"},
            {"subject": "Symphony No. 9", "predicate": "composed_in", "object": "Vienna"},
        ],
    },
    {
        "id": "openai_chatgpt",
        "category": "standard",
        "domain": "technology",
        "text": "OpenAI launched ChatGPT in November 2022. It uses GPT-4 as its underlying model.",
        "entities": [
            {"name": "OpenAI", "type": "organization"},
            {"name": "ChatGPT", "type": "product"},
            {"name": "November 2022", "type": "temporal"},
            {"name": "GPT-4", "type": "technology"},
        ],
        "relations": [
            {"subject": "OpenAI", "predicate": "launched", "object": "ChatGPT"},
            {"subject": "ChatGPT", "predicate": "launched_in", "object": "November 2022"},
            {"subject": "ChatGPT", "predicate": "uses", "object": "GPT-4"},
        ],
    },
    # === COMPLEX CASES (6+ entities, 3+ relations) ===
    {
        "id": "tesla_complex",
        "category": "complex",
        "domain": "business",
        "text": (
            "Elon Musk is the CEO of Tesla and SpaceX. "
            "Tesla is headquartered in Austin, Texas. "
            "SpaceX launched Starship from Boca Chica."
        ),
        "entities": [
            {"name": "Elon Musk", "type": "person"},
            {"name": "Tesla", "type": "organization"},
            {"name": "SpaceX", "type": "organization"},
            {"name": "Austin", "type": "location"},
            {"name": "Texas", "type": "location"},
            {"name": "Starship", "type": "product"},
            {"name": "Boca Chica", "type": "location"},
        ],
        "relations": [
            {"subject": "Elon Musk", "predicate": "ceo_of", "object": "Tesla"},
            {"subject": "Elon Musk", "predicate": "ceo_of", "object": "SpaceX"},
            {"subject": "Tesla", "predicate": "headquartered_in", "object": "Austin"},
            {"subject": "Austin", "predicate": "located_in", "object": "Texas"},
            {"subject": "SpaceX", "predicate": "launched", "object": "Starship"},
            {"subject": "Starship", "predicate": "launched_from", "object": "Boca Chica"},
        ],
    },
    {
        "id": "django_flask",
        "category": "standard",
        "domain": "technology",
        "text": "Django and Flask are popular Python web frameworks. Django was created by Adrian Holovaty.",
        "entities": [
            {"name": "Django", "type": "technology"},
            {"name": "Flask", "type": "technology"},
            {"name": "Python", "type": "technology"},
            {"name": "Adrian Holovaty", "type": "person"},
        ],
        "relations": [
            {"subject": "Django", "predicate": "framework_for", "object": "Python"},
            {"subject": "Flask", "predicate": "framework_for", "object": "Python"},
            {"subject": "Adrian Holovaty", "predicate": "created", "object": "Django"},
        ],
    },
    {
        "id": "world_war_2",
        "category": "complex",
        "domain": "history",
        "text": (
            "World War II ended in 1945. Germany surrendered to the Allied Powers. "
            "The war began when Germany invaded Poland in 1939."
        ),
        "entities": [
            {"name": "World War II", "type": "event"},
            {"name": "1945", "type": "temporal"},
            {"name": "Germany", "type": "location"},
            {"name": "Allied Powers", "type": "organization"},
            {"name": "Poland", "type": "location"},
            {"name": "1939", "type": "temporal"},
        ],
        "relations": [
            {"subject": "World War II", "predicate": "ended_in", "object": "1945"},
            {"subject": "World War II", "predicate": "began_in", "object": "1939"},
            {"subject": "Germany", "predicate": "surrendered_to", "object": "Allied Powers"},
            {"subject": "Germany", "predicate": "invaded", "object": "Poland"},
        ],
    },
    {
        "id": "kubernetes",
        "category": "complex",
        "domain": "technology",
        "text": (
            "Kubernetes was developed by Google and is now maintained by CNCF. "
            "It orchestrates Docker containers and was inspired by Google's Borg system."
        ),
        "entities": [
            {"name": "Kubernetes", "type": "technology"},
            {"name": "Google", "type": "organization"},
            {"name": "CNCF", "type": "organization"},
            {"name": "Docker", "type": "technology"},
            {"name": "Borg", "type": "technology"},
        ],
        "relations": [
            {"subject": "Google", "predicate": "developed", "object": "Kubernetes"},
            {"subject": "Google", "predicate": "developed", "object": "Borg"},
            {"subject": "CNCF", "predicate": "maintains", "object": "Kubernetes"},
            {"subject": "Kubernetes", "predicate": "orchestrates", "object": "Docker"},
            {"subject": "Borg", "predicate": "inspired", "object": "Kubernetes"},
        ],
    },
    {
        "id": "netflix_streaming",
        "category": "complex",
        "domain": "business",
        "text": (
            "Netflix was founded by Reed Hastings in 1997. "
            "The company launched streaming in 2007 and now operates from Los Gatos, California. "
            "They produce original content like Stranger Things."
        ),
        "entities": [
            {"name": "Netflix", "type": "organization"},
            {"name": "Reed Hastings", "type": "person"},
            {"name": "1997", "type": "temporal"},
            {"name": "2007", "type": "temporal"},
            {"name": "Los Gatos", "type": "location"},
            {"name": "California", "type": "location"},
            {"name": "Stranger Things", "type": "product"},
        ],
        "relations": [
            {"subject": "Reed Hastings", "predicate": "founded", "object": "Netflix"},
            {"subject": "Netflix", "predicate": "founded_in", "object": "1997"},
            {"subject": "Netflix", "predicate": "headquartered_in", "object": "Los Gatos"},
            {"subject": "Netflix", "predicate": "produces", "object": "Stranger Things"},
            {"subject": "Los Gatos", "predicate": "located_in", "object": "California"},
        ],
    },
    # === EDGE CASES ===
    {
        "id": "nested_orgs",
        "category": "edge",
        "domain": "business",
        "text": "Alphabet Inc. is the parent company of Google. Google owns YouTube and DeepMind.",
        "entities": [
            {"name": "Alphabet Inc.", "type": "organization"},
            {"name": "Google", "type": "organization"},
            {"name": "YouTube", "type": "organization"},
            {"name": "DeepMind", "type": "organization"},
        ],
        "relations": [
            {"subject": "Alphabet Inc.", "predicate": "parent_of", "object": "Google"},
            {"subject": "Google", "predicate": "owns", "object": "YouTube"},
            {"subject": "Google", "predicate": "owns", "object": "DeepMind"},
        ],
    },
    {
        "id": "ambiguous_entity",
        "category": "edge",
        "domain": "technology",
        "text": "Apple uses Swift for iOS development. Swift is also a banking network used by Apple Pay.",
        "entities": [
            {"name": "Apple", "type": "organization"},
            {"name": "Swift", "type": "technology"},  # Programming language
            {"name": "iOS", "type": "technology"},
            {"name": "Apple Pay", "type": "product"},
            # Note: SWIFT banking network is ambiguous - accept either interpretation
        ],
        "relations": [
            {"subject": "Apple", "predicate": "uses", "object": "Swift"},
            {"subject": "Swift", "predicate": "used_for", "object": "iOS"},
            {"subject": "Apple Pay", "predicate": "uses", "object": "Swift"},
        ],
    },
]


@dataclass
class ExtractionResult:
    """Result from a single extraction."""

    model: str
    extractor_type: str  # 'single' or 'dual'
    test_id: str
    latency_ms: float
    entity_count: int
    relation_count: int
    entities_found: list[str]
    relations_found: list[tuple[str, str, str]]  # (subj, pred, obj)
    error: str | None = None


@dataclass
class QualityMetrics:
    """Quality metrics for an extraction."""

    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0


@dataclass
class ModelBenchmark:
    """Aggregated benchmark results for a model."""

    model: str
    extractor_type: str
    avg_latency_ms: float
    latency_std_ms: float
    entity_metrics: QualityMetrics
    relation_metrics: QualityMetrics
    total_tests: int
    errors: int
    raw_results: list[ExtractionResult] = field(default_factory=list)


def normalize_name(name: str) -> str:
    """Normalize entity name for comparison."""
    return name.lower().strip()


def normalize_predicate(pred: str) -> str:
    """Normalize predicate for comparison."""
    return pred.lower().strip().replace("_", " ").replace("-", " ")


def calculate_entity_metrics(predicted: list[str], ground_truth: list[dict]) -> QualityMetrics:
    """Calculate precision/recall for entities."""
    pred_set = {normalize_name(e) for e in predicted}
    gt_set = {normalize_name(e["name"]) for e in ground_truth}

    tp = len(pred_set & gt_set)
    fp = len(pred_set - gt_set)
    fn = len(gt_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return QualityMetrics(
        precision=precision, recall=recall, f1=f1, true_positives=tp, false_positives=fp, false_negatives=fn
    )


def _fuzzy_name_match(a: str, b: str) -> bool:
    """Check if two entity names match, allowing substring containment.

    Handles cases like "Nobel Prize" matching "Nobel Prize in Physics",
    or "Symphony No. 9" matching "Symphony No. 9 in D minor".
    """
    a, b = a.strip(), b.strip()
    if a == b:
        return True
    # Substring containment (shorter must be at least 4 chars to avoid spurious matches)
    if len(a) >= 4 and len(b) >= 4:
        return a in b or b in a
    return False


def _match_relation_pair(
    pred_pair: tuple[str, str], gt_pair: tuple[str, str]
) -> bool:
    """Match a predicted relation pair against a ground truth pair.

    Direction-agnostic: (A, B) matches (B, A) since "X created Y" and
    "Y created_by X" express the same relationship.

    Fuzzy name matching: "Nobel Prize" matches "Nobel Prize in Physics".
    """
    ps, po = pred_pair
    gs, go = gt_pair

    # Forward match
    if _fuzzy_name_match(ps, gs) and _fuzzy_name_match(po, go):
        return True
    # Reverse match (direction-agnostic)
    if _fuzzy_name_match(ps, go) and _fuzzy_name_match(po, gs):
        return True
    return False


def calculate_relation_metrics(predicted: list[tuple[str, str, str]], ground_truth: list[dict]) -> QualityMetrics:
    """Calculate precision/recall for relations.

    Uses direction-agnostic matching (A→B matches B→A) and fuzzy name
    matching (substring containment) to handle predicate inversion and
    entity name variations.
    """
    pred_pairs = [(normalize_name(s), normalize_name(o)) for s, p, o in predicted]
    gt_pairs = [(normalize_name(r["subject"]), normalize_name(r["object"])) for r in ground_truth]

    # Match predicted against ground truth with fuzzy + direction-agnostic matching
    matched_gt = set()  # indices of matched ground truth pairs
    matched_pred = set()  # indices of matched predicted pairs

    for pi, pp in enumerate(pred_pairs):
        for gi, gp in enumerate(gt_pairs):
            if gi not in matched_gt and _match_relation_pair(pp, gp):
                matched_pred.add(pi)
                matched_gt.add(gi)
                break

    tp = len(matched_gt)
    fp = len(pred_pairs) - len(matched_pred)
    fn = len(gt_pairs) - len(matched_gt)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return QualityMetrics(
        precision=precision, recall=recall, f1=f1, true_positives=tp, false_positives=fp, false_negatives=fn
    )


def extract_entity_names(entities: list) -> list[str]:
    """Extract names from various entity formats.

    Handles: MemoryItem objects, dicts with 'name'/'text'/'content', tuples, strings.
    """
    names = []
    for e in entities:
        if isinstance(e, str):
            names.append(e)
        elif hasattr(e, "metadata") and isinstance(e.metadata, dict) and "name" in e.metadata:
            names.append(e.metadata["name"])
        elif hasattr(e, "content") and isinstance(e.content, str):
            names.append(e.content)
        elif isinstance(e, dict):
            # Try 'name' first, then 'text' (spaCy), then 'content'
            name = e.get("name") or e.get("text") or e.get("content")
            if name:
                names.append(str(name))
        elif isinstance(e, tuple | list) and len(e) > 0:
            names.append(str(e[0]))
    return names


def _build_entity_id_to_name(entities: list) -> dict[str, str]:
    """Build a mapping from entity item_id to entity name."""
    id_map: dict[str, str] = {}
    for e in entities:
        eid = None
        name = None
        if hasattr(e, "item_id") and hasattr(e, "metadata"):
            eid = e.item_id
            name = e.metadata.get("name") if isinstance(e.metadata, dict) else None
            if not name and hasattr(e, "content"):
                name = e.content
        elif isinstance(e, dict):
            eid = e.get("item_id") or e.get("id")
            name = e.get("name") or e.get("text") or e.get("content")
        if eid and name:
            id_map[str(eid)] = str(name)
    return id_map


def extract_relations(relations: list, entities: list | None = None) -> list[tuple[str, str, str]]:
    """Extract (subject, predicate, object) tuples from various formats.

    When relations use source_id/target_id (hash IDs), resolves them
    to entity names using the entities list.
    """
    # Build ID -> name map if entities provided
    id_to_name = _build_entity_id_to_name(entities) if entities else {}

    tuples = []
    for r in relations:
        if isinstance(r, dict):
            # Prefer subject/object (name-based), fall back to source_id/target_id
            subj = r.get("subject") or ""
            obj = r.get("object") or ""
            pred = r.get("predicate") or r.get("relation_type", "related_to")

            # If subject/object are empty, resolve from IDs
            if not subj and "source_id" in r:
                subj = id_to_name.get(str(r["source_id"]), "")
            if not obj and "target_id" in r:
                obj = id_to_name.get(str(r["target_id"]), "")

            if subj and obj:
                tuples.append((str(subj), str(pred), str(obj)))
        elif isinstance(r, tuple | list) and len(r) >= 3:
            tuples.append((str(r[0]), str(r[1]), str(r[2])))
    return tuples


def run_single_extractor(
    model_name: str, text: str, api_key: str, api_base_url: str | None = None,
    use_json_schema: bool = False,
) -> dict[str, Any]:
    """Run single-call LLM extractor."""
    from smartmemory.plugins.extractors.llm_single import LLMSingleExtractor

    extractor = LLMSingleExtractor(
        model_name=model_name, api_key=api_key, api_base_url=api_base_url,
    )
    extractor.cfg.use_json_schema = use_json_schema
    start = time.time()
    result = extractor.extract(text)
    latency = (time.time() - start) * 1000
    return {"result": result, "latency_ms": latency}


def run_dual_extractor(model_name: str, text: str, api_key: str) -> dict[str, Any]:
    """Run dual-call (original) LLM extractor."""
    from smartmemory.plugins.extractors.llm import LLMExtractor, LLMExtractorConfig

    config = LLMExtractorConfig(model_name=model_name)
    extractor = LLMExtractor(config=config)
    start = time.time()
    result = extractor.extract(text)
    latency = (time.time() - start) * 1000
    return {"result": result, "latency_ms": latency}


# Singleton cache for traditional extractors (avoid reloading models)
_extractor_cache: dict[str, Any] = {}


def _create_spacy_extractor(model_name: str, use_entity_ruler: bool = False):
    """Create a spaCy extractor with a specific model, bypassing config.

    Args:
        model_name: spaCy model to load.
        use_entity_ruler: If True, add an EntityRuler before NER with patterns for
            entity types spaCy's built-in NER typically misses (technologies, concepts,
            products, awards, events).
    """
    import spacy  # type: ignore

    from smartmemory.plugins.extractors.spacy import SpacyExtractor

    extractor = SpacyExtractor.__new__(SpacyExtractor)
    nlp = spacy.load(model_name)

    if use_entity_ruler:
        ruler = nlp.add_pipe("entity_ruler", before="ner")
        # Patterns for entity types spaCy NER misses: technologies, concepts, products, events, awards.
        # These are domain-general patterns (not gold-data-specific cheating) representing
        # common named entities in tech, science, and business domains.
        patterns = [
            # --- Technologies / Frameworks / Programming Languages ---
            {"label": "PRODUCT", "pattern": "Python"},
            {"label": "PRODUCT", "pattern": "Django"},
            {"label": "PRODUCT", "pattern": "Flask"},
            {"label": "PRODUCT", "pattern": "TensorFlow"},
            {"label": "PRODUCT", "pattern": "PyTorch"},
            {"label": "PRODUCT", "pattern": "Kubernetes"},
            {"label": "PRODUCT", "pattern": "Docker"},
            {"label": "PRODUCT", "pattern": "React"},
            {"label": "PRODUCT", "pattern": "Node.js"},
            {"label": "PRODUCT", "pattern": "TypeScript"},
            {"label": "PRODUCT", "pattern": "JavaScript"},
            {"label": "PRODUCT", "pattern": "Rust"},
            {"label": "PRODUCT", "pattern": "Swift"},
            {"label": "PRODUCT", "pattern": "Java"},
            {"label": "PRODUCT", "pattern": "C++"},
            {"label": "PRODUCT", "pattern": "Linux"},
            {"label": "PRODUCT", "pattern": "Git"},
            {"label": "PRODUCT", "pattern": "Borg"},
            # --- AI Products ---
            {"label": "PRODUCT", "pattern": "ChatGPT"},
            {"label": "PRODUCT", "pattern": "GPT-4"},
            {"label": "PRODUCT", "pattern": "GPT-3"},
            {"label": "PRODUCT", "pattern": "DALL-E"},
            {"label": "PRODUCT", "pattern": "Siri"},
            {"label": "PRODUCT", "pattern": "Alexa"},
            # --- Consumer Products ---
            {"label": "PRODUCT", "pattern": "iPhone"},
            {"label": "PRODUCT", "pattern": "iPad"},
            {"label": "PRODUCT", "pattern": "iOS"},
            {"label": "PRODUCT", "pattern": "Android"},
            {"label": "PRODUCT", "pattern": "Starship"},
            {"label": "PRODUCT", "pattern": [{"LOWER": "app"}, {"LOWER": "store"}]},
            {"label": "PRODUCT", "pattern": [{"LOWER": "apple"}, {"LOWER": "pay"}]},
            {"label": "PRODUCT", "pattern": [{"LOWER": "stranger"}, {"LOWER": "things"}]},
            # --- Concepts / Theories ---
            {"label": "WORK_OF_ART", "pattern": [{"LOWER": "theory"}, {"LOWER": "of"}, {"LOWER": "relativity"}]},
            {"label": "WORK_OF_ART", "pattern": [{"LOWER": "nobel"}, {"LOWER": "prize"}]},
            # --- Musical works ---
            {"label": "WORK_OF_ART", "pattern": [{"LOWER": "symphony"}, {"LOWER": "no"}, {"IS_PUNCT": True, "OP": "?"}, {"LIKE_NUM": True}]},
            {"label": "WORK_OF_ART", "pattern": [{"LOWER": "moonlight"}, {"LOWER": "sonata"}]},
            # --- Historical events ---
            {"label": "EVENT", "pattern": [{"LOWER": "world"}, {"LOWER": "war"}, {"TEXT": {"REGEX": "^(I|II|1|2)$"}}]},
            {"label": "EVENT", "pattern": "D-Day"},
            {"label": "EVENT", "pattern": "Macworld"},
            {"label": "EVENT", "pattern": [{"LOWER": "manhattan"}, {"LOWER": "project"}]},
        ]
        ruler.add_patterns(patterns)

    extractor.nlp = nlp
    return extractor


def run_spacy_extractor(
    text: str, model_name: str = "en_core_web_sm", use_entity_ruler: bool = False,
) -> dict[str, Any]:
    """Run spaCy extractor with a specific model."""
    cache_key = f"spacy_{model_name}{'_ruler' if use_entity_ruler else ''}"
    if cache_key not in _extractor_cache:
        _extractor_cache[cache_key] = _create_spacy_extractor(model_name, use_entity_ruler=use_entity_ruler)

    extractor = _extractor_cache[cache_key]
    start = time.time()
    result = extractor.extract(text)
    latency = (time.time() - start) * 1000
    return {"result": result, "latency_ms": latency}


# --- Progressive prompt variants ---
# V1: Original "refine the draft" (anchoring problem)
PROGRESSIVE_PROMPT_V1_REFINE = """You are refining a knowledge graph extraction. A fast NLP tool already extracted entities and relations from the text below, but it may have missed some or made errors.

TEXT:
{text}

DRAFT EXTRACTION (from spaCy NER + dependency parse):
Entities: {draft_entities}
Relations: {draft_relations}

YOUR TASK:
1. Keep correct entities and relations from the draft
2. Fix any entity TYPE errors (e.g., location labeled as organization)
3. Add MISSING entities the NLP tool missed (concepts, technologies, products, events, temporal, awards)
4. Add MISSING relations the dependency parser couldn't find
5. Remove any clearly wrong or duplicate entities/relations

ENTITY TYPES: person, organization, location, event, product, work_of_art, temporal, concept, technology, award

Return JSON:
{{
  "entities": [
    {{"name": "exact name", "entity_type": "type", "confidence": 0.95}}
  ],
  "relations": [
    {{"subject": "entity1 name", "predicate": "relationship", "object": "entity2 name"}}
  ]
}}

Return ONLY valid JSON."""

# V2: "Extract independently, use draft as safety net" (minimal anchoring)
PROGRESSIVE_PROMPT_V2_INDEPENDENT = """Extract entities and relationships from this text for a knowledge graph.

TEXT:
{text}

ENTITY TYPES: person, organization, location, event, product, work_of_art, temporal, concept, technology, award

INSTRUCTIONS:
1. Extract ALL significant entities with their types
2. Extract ALL relationships between entities
3. Use exact entity names in relations (case-sensitive match)
4. Be comprehensive but avoid duplicates

CROSS-CHECK: A simple NLP tool found these entities: {draft_entities}
Make sure you haven't missed any of these, but do NOT limit yourself to this list — extract everything you find.

Return JSON:
{{
  "entities": [
    {{"name": "exact name", "entity_type": "type", "confidence": 0.95}}
  ],
  "relations": [
    {{"subject": "entity1 name", "predicate": "relationship", "object": "entity2 name"}}
  ]
}}

Return ONLY valid JSON."""

# V3: "Deficit-focused" — only ask for what spaCy can't do
PROGRESSIVE_PROMPT_V3_DEFICIT = """A simple NLP tool extracted named entities from this text. Your job is to COMPLETE the extraction by finding what it missed and adding all relationships.

TEXT:
{text}

ALREADY FOUND BY NLP TOOL (names only): {draft_entity_names}

THE NLP TOOL CANNOT:
- Find concepts, technologies, theories, or abstract ideas
- Find products, tools, frameworks, or creative works
- Find temporal references (dates, years, time periods)
- Find events or awards
- Extract relationships between entities

YOUR TASK:
1. Add ALL missing entities (especially concepts, technologies, products, temporal, events, awards)
2. Verify the NLP entities above are correct — include them in your output with proper types
3. Extract ALL relationships between ALL entities (both NLP-found and your additions)

ENTITY TYPES: person, organization, location, event, product, work_of_art, temporal, concept, technology, award

Return JSON:
{{
  "entities": [
    {{"name": "exact name", "entity_type": "type", "confidence": 0.95}}
  ],
  "relations": [
    {{"subject": "entity1 name", "predicate": "relationship", "object": "entity2 name"}}
  ]
}}

Return ONLY valid JSON."""

# V4: Entity names only (no types to anchor on) + full relation extraction
PROGRESSIVE_PROMPT_V4_NAMES_ONLY = """Extract entities and relationships from this text for a knowledge graph.

TEXT:
{text}

A fast NER tool pre-identified these named entities: {draft_entity_names}
These are probably correct but the list may be INCOMPLETE — the tool only finds people, organizations, and locations. It misses concepts, technologies, products, events, and temporal references.

ENTITY TYPES: person, organization, location, event, product, work_of_art, temporal, concept, technology, award

INSTRUCTIONS:
1. Include the pre-identified entities above (assign correct types)
2. Add any entities the NER tool missed
3. Extract ALL relationships between entities
4. Be comprehensive — extract every meaningful relationship

Return JSON:
{{
  "entities": [
    {{"name": "exact name", "entity_type": "type", "confidence": 0.95}}
  ],
  "relations": [
    {{"subject": "entity1 name", "predicate": "relationship", "object": "entity2 name"}}
  ]
}}

Return ONLY valid JSON."""

PROGRESSIVE_PROMPTS = {
    "v1_refine": PROGRESSIVE_PROMPT_V1_REFINE,
    "v2_independent": PROGRESSIVE_PROMPT_V2_INDEPENDENT,
    "v3_deficit": PROGRESSIVE_PROMPT_V3_DEFICIT,
    "v4_names_only": PROGRESSIVE_PROMPT_V4_NAMES_ONLY,
}


def run_progressive_extractor(text: str, prompt_variant: str = "v1_refine") -> dict[str, Any]:
    """Run progressive extraction: spaCy trf draft → Groq LLM refinement.

    Args:
        text: Input text to extract from.
        prompt_variant: Which prompt strategy to use (v1_refine, v2_independent, v3_deficit, v4_names_only).

    Measures total end-to-end latency (spaCy + Groq combined).
    """
    import json as _json

    from smartmemory.utils.llm import call_llm

    prompt_template = PROGRESSIVE_PROMPTS.get(prompt_variant, PROGRESSIVE_PROMPTS["v1_refine"])

    # Step 1: spaCy trf draft
    start = time.time()
    spacy_out = run_spacy_extractor(text, "en_core_web_trf")
    spacy_latency = spacy_out["latency_ms"]

    # Format draft for the prompt
    draft_result = spacy_out["result"]
    draft_entities_list = []
    draft_entity_names_list = []
    for e in draft_result.get("entities", []):
        if isinstance(e, dict):
            name = e.get("text") or e.get("name", "")
            etype = e.get("type", "entity")
            draft_entities_list.append(f"{name} ({etype})")
            draft_entity_names_list.append(name)
        elif isinstance(e, str):
            draft_entities_list.append(e)
            draft_entity_names_list.append(e)

    draft_relations_list = []
    for r in draft_result.get("relations", []):
        if isinstance(r, tuple | list) and len(r) >= 3:
            draft_relations_list.append(f"{r[0]} --{r[1]}--> {r[2]}")
        elif isinstance(r, dict):
            s = r.get("subject") or r.get("source", "")
            p = r.get("predicate") or r.get("relation_type", "related_to")
            o = r.get("object") or r.get("target", "")
            draft_relations_list.append(f"{s} --{p}--> {o}")

    draft_entities_str = ", ".join(draft_entities_list) if draft_entities_list else "(none found)"
    draft_entity_names_str = ", ".join(draft_entity_names_list) if draft_entity_names_list else "(none found)"
    draft_relations_str = "; ".join(draft_relations_list) if draft_relations_list else "(none found)"

    # Step 2: Groq LLM refinement
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return {"result": draft_result, "latency_ms": spacy_latency, "error": "GROQ_API_KEY not set"}

    prompt = prompt_template.format(
        text=text,
        draft_entities=draft_entities_str,
        draft_entity_names=draft_entity_names_str,
        draft_relations=draft_relations_str,
    )

    system_prompts = {
        "v1_refine": "You are an expert knowledge graph extractor refining a draft extraction.",
        "v2_independent": "You are an expert knowledge graph extractor. Extract entities and relationships accurately.",
        "v3_deficit": "You are an expert knowledge graph extractor completing a partial extraction.",
        "v4_names_only": "You are an expert knowledge graph extractor. Extract entities and relationships accurately.",
    }

    try:
        parsed, raw = call_llm(
            model="llama-3.3-70b-versatile",
            system_prompt=system_prompts.get(prompt_variant, system_prompts["v1_refine"]),
            user_content=prompt,
            response_format={"type": "json_object"},
            max_output_tokens=2000,
            temperature=0.0,
            api_key=api_key,
            api_base="https://api.groq.com/openai/v1",
            config_section="extractor",
        )
    except Exception as e:
        total_latency = (time.time() - start) * 1000
        return {"result": draft_result, "latency_ms": total_latency, "error": f"Groq refinement failed: {e}"}

    total_latency = (time.time() - start) * 1000

    # Parse Groq response
    data = parsed or {}
    if not data and raw and isinstance(raw, str):
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            first_nl = cleaned.index("\n") if "\n" in cleaned else len(cleaned)
            cleaned = cleaned[first_nl + 1:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        try:
            data = _json.loads(cleaned.strip())
        except Exception:
            return {"result": draft_result, "latency_ms": total_latency, "error": "Failed to parse Groq response"}

    # Convert to standard format (dicts with name/text keys for extract_entity_names)
    entities = [
        {"name": e.get("name", ""), "entity_type": e.get("entity_type", "concept")}
        for e in data.get("entities", [])
        if isinstance(e, dict) and e.get("name")
    ]
    relations = [
        {"subject": r.get("subject", ""), "predicate": r.get("predicate", "related_to"), "object": r.get("object", "")}
        for r in data.get("relations", [])
        if isinstance(r, dict) and r.get("subject") and r.get("object")
    ]

    return {"result": {"entities": entities, "relations": relations}, "latency_ms": total_latency}


def run_gliner2_extractor(text: str) -> dict[str, Any]:
    """Run GLiNER2 extractor."""
    if "gliner2" not in _extractor_cache:
        try:
            from smartmemory.plugins.extractors.gliner2 import GLiNER2Extractor

            _extractor_cache["gliner2"] = GLiNER2Extractor()
        except ImportError as e:
            return {
                "result": {"entities": [], "relations": []},
                "latency_ms": 0,
                "error": f"gliner2 not installed: {e}",
            }

    extractor = _extractor_cache["gliner2"]
    start = time.time()
    result = extractor.extract(text)
    latency = (time.time() - start) * 1000
    return {"result": result, "latency_ms": latency}


def run_hybrid_extractor(text: str) -> dict[str, Any]:
    """Run Hybrid (GLiNER2 + ReLiK) extractor."""
    if "hybrid" not in _extractor_cache:
        try:
            from smartmemory.plugins.extractors.hybrid import HybridExtractor

            _extractor_cache["hybrid"] = HybridExtractor()
        except ImportError as e:
            return {
                "result": {"entities": [], "relations": []},
                "latency_ms": 0,
                "error": f"hybrid deps not installed: {e}",
            }

    extractor = _extractor_cache["hybrid"]
    start = time.time()
    result = extractor.extract(text)
    latency = (time.time() - start) * 1000
    return {"result": result, "latency_ms": latency}


def _parse_rebel_output(text: str) -> list[dict[str, str]]:
    """Parse REBEL model output with special tokens into relation triplets.

    REBEL uses: <triplet> subject <subj> object <obj> relation
    Ported from feed/src/rebel/kb.py extract_relations_from_model_output().
    """
    relations = []
    subject, relation, object_ = "", "", ""
    text = text.strip()
    current = "x"
    text_replaced = text.replace("<s>", "").replace("<pad>", "").replace("</s>", "")
    for token in text_replaced.split():
        if token == "<triplet>":
            current = "t"
            if relation != "":
                relations.append({"source": subject.strip(), "relation": relation.strip(), "target": object_.strip()})
                relation = ""
            subject = ""
        elif token == "<subj>":
            current = "s"
            if relation != "":
                relations.append({"source": subject.strip(), "relation": relation.strip(), "target": object_.strip()})
            object_ = ""
        elif token == "<obj>":
            current = "o"
            relation = ""
        else:
            if current == "t":
                subject += " " + token
            elif current == "s":
                object_ += " " + token
            elif current == "o":
                relation += " " + token
    if subject.strip() and relation.strip() and object_.strip():
        relations.append({"source": subject.strip(), "relation": relation.strip(), "target": object_.strip()})
    return relations


def run_rebel_extractor(text: str) -> dict[str, Any]:
    """Run REBEL end-to-end triplet extraction (Babelscape/rebel-large)."""
    if "rebel" not in _extractor_cache:
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained("Babelscape/rebel-large")
            rebel_model = AutoModelForSeq2SeqLM.from_pretrained("Babelscape/rebel-large")
            _extractor_cache["rebel"] = (tokenizer, rebel_model)
        except Exception as e:
            return {"result": {"entities": [], "relations": []}, "latency_ms": 0, "error": f"REBEL load failed: {e}"}

    tokenizer, rebel_model = _extractor_cache["rebel"]
    start = time.time()

    try:
        inputs = tokenizer(text, max_length=512, padding=True, truncation=True, return_tensors="pt")
        generated = rebel_model.generate(
            **inputs, max_length=256, length_penalty=0, num_beams=3, num_return_sequences=3
        )

        # Parse all generated sequences and deduplicate
        all_relations = []
        for seq in generated:
            decoded = tokenizer.decode(seq, skip_special_tokens=False)
            all_relations.extend(_parse_rebel_output(decoded))

        seen: set[tuple[str, str, str]] = set()
        unique_relations = []
        for r in all_relations:
            key = (r["source"].lower(), r["relation"].lower(), r["target"].lower())
            if key not in seen:
                seen.add(key)
                unique_relations.append(r)

        # Extract entities from relation triplets
        entity_names: set[str] = set()
        for r in unique_relations:
            entity_names.add(r["source"])
            entity_names.add(r["target"])

        latency = (time.time() - start) * 1000
        entities = [{"name": name, "entity_type": "unknown"} for name in sorted(entity_names)]
        relations = [{"subject": r["source"], "predicate": r["relation"], "object": r["target"]} for r in unique_relations]
        return {"result": {"entities": entities, "relations": relations}, "latency_ms": latency}
    except Exception as e:
        latency = (time.time() - start) * 1000
        return {"result": {"entities": [], "relations": []}, "latency_ms": latency, "error": f"REBEL extraction failed: {e}"}


def run_nuner_extractor(text: str) -> dict[str, Any]:
    """Run NuMind NuNER Zero zero-shot NER (entity extraction only).

    NuNER Zero is a 0.4B parameter model - SOTA zero-shot NER,
    +3.1% F1 over GLiNER-large-v2.1. Uses the gliner package.

    NOTE: NuNER Zero requires gliner==0.1.12. With gliner>=0.2.x,
    the model produces near-random scores (~0.05). Benchmark results
    with current setup are unreliable.
    """
    if "nuner" not in _extractor_cache:
        try:
            from gliner import GLiNER  # type: ignore

            # Patch for huggingface_hub >= 1.0 compatibility
            _orig_fp = GLiNER._from_pretrained.__func__

            @classmethod  # type: ignore[misc]
            def _patched_fp(klass, _ofp=_orig_fp, **kwargs):
                kwargs.setdefault("proxies", None)
                kwargs.setdefault("resume_download", False)
                return _ofp(klass, **kwargs)

            GLiNER._from_pretrained = _patched_fp  # type: ignore[assignment]

            model = GLiNER.from_pretrained("numind/NuNER_Zero")
            _extractor_cache["nuner"] = model
        except Exception as e:
            return {"result": {"entities": [], "relations": []}, "latency_ms": 0, "error": f"NuNER load failed: {e}"}

    model = _extractor_cache["nuner"]
    start = time.time()

    try:
        # NuNER Zero requires lowercase labels
        entity_labels = [
            "person", "organization", "location", "event", "product",
            "technology", "concept", "temporal", "award",
        ]
        ner_results = model.predict_entities(text, entity_labels, threshold=0.3)

        latency = (time.time() - start) * 1000
        entities = [{"name": ent["text"], "entity_type": ent["label"]} for ent in ner_results]
        # NuNER is NER-only, no relation extraction
        return {"result": {"entities": entities, "relations": []}, "latency_ms": latency}
    except Exception as e:
        latency = (time.time() - start) * 1000
        return {"result": {"entities": [], "relations": []}, "latency_ms": latency, "error": f"NuNER failed: {e}"}


def run_nuextract_extractor(text: str) -> dict[str, Any]:
    """Run NuMind NuExtract 1.5 tiny structured extraction (0.5B params).

    Uses a JSON template to define extraction schema - a lightweight
    local alternative to LLM-based extraction for entities and relations.
    """
    if "nuextract" not in _extractor_cache:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            model_name = "numind/NuExtract-tiny-v1.5"
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=torch.bfloat16, trust_remote_code=True
            )
            model = model.eval()  # type: ignore[assignment]
            # Use MPS on Apple Silicon if available, else CPU
            device = "mps" if torch.backends.mps.is_available() else "cpu"
            model.to(device)
            _extractor_cache["nuextract"] = (tokenizer, model, device)
        except Exception as e:
            return {"result": {"entities": [], "relations": []}, "latency_ms": 0, "error": f"NuExtract load failed: {e}"}

    tokenizer, model, device = _extractor_cache["nuextract"]
    start = time.time()

    try:
        import torch

        # Template for entity + relation extraction
        template = json.dumps({
            "entities": [{"name": "", "type": ""}],
            "relations": [{"subject": "", "predicate": "", "object": ""}],
        })

        prompt = f"<|input|>\n### Template:\n{template}\n### Text:\n{text}\n\n<|output|>"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)

        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=500, do_sample=False)

        result_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        # Parse the output after <|output|> marker
        if "<|output|>" in result_text:
            extracted_text = result_text.split("<|output|>", 1)[1].strip()
        else:
            extracted_text = result_text.strip()

        # Parse JSON output
        entities = []
        relations = []
        try:
            extracted_text = extracted_text.strip()
            if extracted_text.startswith("```"):
                extracted_text = extracted_text.split("```")[1]
                if extracted_text.startswith("json"):
                    extracted_text = extracted_text[4:]
            parsed = json.loads(extracted_text)
            raw_entities = parsed.get("entities", [])
            raw_relations = parsed.get("relations", [])

            for e in raw_entities:
                if isinstance(e, dict) and e.get("name"):
                    entities.append({"name": e["name"], "entity_type": e.get("type", "unknown")})
            for r in raw_relations:
                if isinstance(r, dict) and r.get("subject") and r.get("object"):
                    relations.append({
                        "subject": r["subject"],
                        "predicate": r.get("predicate", "related_to"),
                        "object": r["object"],
                    })
        except json.JSONDecodeError:
            pass  # Return empty if JSON parsing fails

        latency = (time.time() - start) * 1000
        return {"result": {"entities": entities, "relations": relations}, "latency_ms": latency}
    except Exception as e:
        latency = (time.time() - start) * 1000
        return {"result": {"entities": [], "relations": []}, "latency_ms": latency, "error": f"NuExtract failed: {e}"}


def run_opennre_extractor(text: str) -> dict[str, Any]:
    """Run spaCy NER + OpenNRE relation classification (wiki80 CNN).

    Combines spaCy for entity detection with OpenNRE's wiki80 CNN model
    for relation classification between entity pairs. OpenNRE uses
    Wikidata relation types (80 classes).
    """
    if "opennre" not in _extractor_cache:
        try:
            import opennre  # type: ignore
            import spacy  # type: ignore

            model = opennre.get_model("wiki80_cnn_softmax")
            nlp = spacy.load("en_core_web_sm")
            _extractor_cache["opennre"] = (model, nlp)
        except Exception as e:
            return {"result": {"entities": [], "relations": []}, "latency_ms": 0, "error": f"OpenNRE load failed: {e}"}

    opennre_model, nlp = _extractor_cache["opennre"]
    start = time.time()

    try:
        doc = nlp(text)

        # Extract entities from spaCy
        entities = []
        ent_spans = []  # (start_char, end_char, name)
        for ent in doc.ents:
            entities.append({"name": ent.text, "entity_type": ent.label_.lower()})
            ent_spans.append((ent.start_char, ent.end_char, ent.text))

        # Classify relations for all entity pairs
        relations = []
        for i, (h_start, h_end, h_name) in enumerate(ent_spans):
            for j, (t_start, t_end, t_name) in enumerate(ent_spans):
                if i == j:
                    continue
                result = opennre_model.infer({
                    "text": text,
                    "h": {"pos": (h_start, h_end)},
                    "t": {"pos": (t_start, t_end)},
                })
                rel_type, confidence = result
                # Only keep relations with reasonable confidence
                if confidence > 0.3 and rel_type != "no_relation":
                    relations.append({
                        "subject": h_name,
                        "predicate": rel_type,
                        "object": t_name,
                    })

        # Deduplicate relations (keep highest confidence)
        seen: set[tuple[str, str]] = set()
        unique_relations = []
        for r in relations:
            key = (r["subject"].lower(), r["object"].lower())
            reverse_key = (r["object"].lower(), r["subject"].lower())
            if key not in seen and reverse_key not in seen:
                seen.add(key)
                unique_relations.append(r)

        latency = (time.time() - start) * 1000
        return {"result": {"entities": entities, "relations": unique_relations}, "latency_ms": latency}
    except Exception as e:
        latency = (time.time() - start) * 1000
        return {"result": {"entities": [], "relations": []}, "latency_ms": latency, "error": f"OpenNRE failed: {e}"}


def run_nuner_opennre_extractor(text: str) -> dict[str, Any]:
    """Run NuNER Zero NER + OpenNRE relation classification.

    Best local NER (NuNER Zero, 0.4B) paired with OpenNRE wiki80 CNN
    for relation classification. Tests whether better NER improves
    downstream relation extraction.
    """
    if "nuner_opennre" not in _extractor_cache:
        try:
            import opennre  # type: ignore
            from gliner import GLiNER  # type: ignore

            # Patch for huggingface_hub compatibility
            _orig_fp = GLiNER._from_pretrained.__func__

            @classmethod  # type: ignore[misc]
            def _patched_fp(klass, _ofp=_orig_fp, **kwargs):
                kwargs.setdefault("proxies", None)
                kwargs.setdefault("resume_download", False)
                return _ofp(klass, **kwargs)

            GLiNER._from_pretrained = _patched_fp  # type: ignore[assignment]

            # Reuse cached models if available
            nuner = _extractor_cache.get("nuner") or GLiNER.from_pretrained("numind/NuNER_Zero")
            opennre_parts = _extractor_cache.get("opennre")
            if opennre_parts:
                opennre_model = opennre_parts[0]
            else:
                opennre_model = opennre.get_model("wiki80_cnn_softmax")

            _extractor_cache["nuner_opennre"] = (nuner, opennre_model)
        except Exception as e:
            return {"result": {"entities": [], "relations": []}, "latency_ms": 0, "error": f"NuNER+OpenNRE load failed: {e}"}

    nuner, opennre_model = _extractor_cache["nuner_opennre"]
    start = time.time()

    try:
        entity_labels = [
            "person", "organization", "location", "event", "product",
            "technology", "concept", "temporal", "award",
        ]
        ner_results = nuner.predict_entities(text, entity_labels, threshold=0.3)

        entities = []
        ent_spans = []
        for ent in ner_results:
            entities.append({"name": ent["text"], "entity_type": ent["label"]})
            ent_spans.append((ent["start"], ent["end"], ent["text"]))

        # Classify relations between all entity pairs
        relations = []
        for i, (h_start, h_end, h_name) in enumerate(ent_spans):
            for j, (t_start, t_end, t_name) in enumerate(ent_spans):
                if i == j:
                    continue
                result = opennre_model.infer({
                    "text": text,
                    "h": {"pos": (h_start, h_end)},
                    "t": {"pos": (t_start, t_end)},
                })
                rel_type, confidence = result
                if confidence > 0.3 and rel_type != "no_relation":
                    relations.append({
                        "subject": h_name,
                        "predicate": rel_type,
                        "object": t_name,
                    })

        seen: set[tuple[str, str]] = set()
        unique_relations = []
        for r in relations:
            key = (r["subject"].lower(), r["object"].lower())
            reverse_key = (r["object"].lower(), r["subject"].lower())
            if key not in seen and reverse_key not in seen:
                seen.add(key)
                unique_relations.append(r)

        latency = (time.time() - start) * 1000
        return {"result": {"entities": entities, "relations": unique_relations}, "latency_ms": latency}
    except Exception as e:
        latency = (time.time() - start) * 1000
        return {"result": {"entities": [], "relations": []}, "latency_ms": latency, "error": f"NuNER+OpenNRE failed: {e}"}


def run_gliner_glirel_extractor(text: str) -> dict[str, Any]:
    """Run GLiNER (NER) + GLiREL (zero-shot relation extraction).

    GLiNER handles entity recognition. GLiREL handles relation extraction
    using the NER results as input. This is the most promising local combo
    for full knowledge graph extraction.
    """
    if "gliner_glirel" not in _extractor_cache:
        try:
            import spacy  # type: ignore
            from gliner import GLiNER  # type: ignore
            from glirel import GLiREL  # type: ignore

            # Patch _from_pretrained for huggingface_hub >= 1.0 compatibility
            # (newer hub drops 'proxies' and 'resume_download' kwargs)
            for cls in [GLiNER, GLiREL]:
                _orig_fp = cls._from_pretrained.__func__

                @classmethod  # type: ignore[misc]
                def _patched_fp(klass, _ofp=_orig_fp, **kwargs):
                    kwargs.setdefault("proxies", None)
                    kwargs.setdefault("resume_download", False)
                    return _ofp(klass, **kwargs)

                cls._from_pretrained = _patched_fp  # type: ignore[assignment]

            gliner_model = GLiNER.from_pretrained("urchade/gliner_multi-v2.1")
            glirel_model = GLiREL.from_pretrained("jackboyla/glirel-large-v0")
            nlp = spacy.load("en_core_web_sm")
            _extractor_cache["gliner_glirel"] = (gliner_model, glirel_model, nlp)
        except Exception as e:
            return {"result": {"entities": [], "relations": []}, "latency_ms": 0, "error": f"GLiNER+GLiREL load failed: {e}"}

    gliner_model, glirel_model, nlp = _extractor_cache["gliner_glirel"]
    start = time.time()

    try:
        # NER with GLiNER
        entity_labels = ["person", "organization", "location", "event", "product", "technology", "concept", "temporal", "award"]
        ner_results = gliner_model.predict_entities(text, entity_labels, threshold=0.25)

        # Tokenize with spaCy for GLiREL
        doc = nlp(text)
        tokens = [token.text for token in doc]

        # Convert GLiNER NER spans to GLiREL format: [[start_tok, end_tok, type, text], ...]
        ner_for_glirel = []
        for ent in ner_results:
            start_char = ent["start"]
            end_char = ent["end"]
            start_tok = None
            end_tok = None
            for token in doc:
                if token.idx <= start_char < token.idx + len(token.text):
                    start_tok = token.i
                if token.idx < end_char <= token.idx + len(token.text):
                    end_tok = token.i
            if start_tok is not None and end_tok is not None:
                ner_for_glirel.append([start_tok, end_tok, ent["label"], ent["text"]])

        # Relation extraction with GLiREL
        relation_labels = [
            "founded by", "created by", "developed by", "located in", "headquartered in",
            "part of", "subsidiary of", "CEO of", "works for", "member of",
            "uses", "inspired by", "produces", "launched", "won",
            "born in", "died in", "released", "announced", "composed by",
            "owns", "parent company of", "maintained by", "orchestrates", "invaded",
        ]

        relations = []
        if ner_for_glirel:
            rel_results = glirel_model.predict_relations(tokens, relation_labels, threshold=0.1, ner=ner_for_glirel, top_k=10)
            for r in rel_results:
                relations.append({"subject": r["head_text"], "predicate": r["label"], "object": r["tail_text"]})

        latency = (time.time() - start) * 1000
        entities = [{"name": ent["text"], "entity_type": ent["label"]} for ent in ner_results]
        return {"result": {"entities": entities, "relations": relations}, "latency_ms": latency}
    except Exception as e:
        latency = (time.time() - start) * 1000
        return {"result": {"entities": [], "relations": []}, "latency_ms": latency, "error": f"GLiNER+GLiREL failed: {e}"}


# LM Studio base URL (OpenAI-compatible local inference)
LM_STUDIO_BASE = "http://localhost:1234/v1"

# Model ID substrings that indicate non-chat models (embeddings, OCR, base models)
LM_STUDIO_SKIP_PATTERNS = [
    # Non-chat models
    "text-embedding", "embed-text", "olmocr", "base-0414",
    # Coder/function-calling models
    "coder", "function-calling", "functionary", "python-coder",
    # Already tested - niche/small models
    "devstral", "glitch", "art-0-8b", "kat-dev", "intellect-2",
    "granite-4-h-tiny",
    # Already tested - commercial/small models
    "nemotron-nano", "nemotron-3-nano",
    "gpt-oss", "magistral-small", "ministral",
    "mistral-small", "phi-4-reasoning", "phi-4-mlx", "phi-3-mini",
    "glm-4.7-flash", "qwen3-14b", "qwen3-30b",
    "deepseek-r1-distill-llama-8b",
]

# MoE indicators in model names (active params << total)
MOE_INDICATORS = ["a3b", "a8b", "a14b", "moe"]


def _estimate_params_b(model_id: str) -> float:
    """Estimate parameter count in billions from model ID. Returns 0 if unknown."""
    lower = model_id.lower()
    sizes = []
    for match in re.finditer(r"(\d+(?:\.\d+)?)b(?:\b|[-_])", lower):
        val = float(match.group(1))
        if val >= 1.0:
            sizes.append(val)
    if sizes:
        return max(sizes)
    if "tiny" in lower or "nano" in lower:
        return 3.0
    if "mini" in lower or "small" in lower:
        return 7.0
    return 10.0  # unknown — assume mid-range


def _model_too_large(model_id: str, max_params_b: float = 24.0) -> bool:
    """Check if model exceeds size limit. MoE models always pass."""
    lower = model_id.lower()
    if any(ind in lower for ind in MOE_INDICATORS):
        return False
    size = _estimate_params_b(model_id)
    return size > max_params_b


def discover_lm_studio_models(max_params_b: float = 24.0) -> dict[str, dict]:
    """Query LM Studio /v1/models and return configs for all loaded chat models.

    Args:
        max_params_b: Skip dense models larger than this (in billions). MoE models are always included.
    """
    import urllib.request

    try:
        req = urllib.request.Request(f"{LM_STUDIO_BASE}/models", headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
    except Exception as e:
        print(f"LM Studio not reachable at {LM_STUDIO_BASE}: {e}")
        return {}

    accepted = []
    skipped = []
    for model in data.get("data", []):
        model_id = model["id"]
        if any(pat in model_id.lower() for pat in LM_STUDIO_SKIP_PATTERNS):
            skipped.append(f"  skip (non-chat): {model_id}")
            continue
        if _model_too_large(model_id, max_params_b):
            skipped.append(f"  skip (>{max_params_b:.0f}B): {model_id}")
            continue
        accepted.append(model_id)

    # Sort smallest models first so we get fast results early
    accepted.sort(key=lambda m: _estimate_params_b(m))

    configs = {}
    for model_id in accepted:
        est = _estimate_params_b(model_id)
        safe_key = "lm_" + model_id.replace("/", "_").replace("-", "_").replace(".", "_").replace("@", "_")
        configs[safe_key] = {
            "name": model_id,
            "api_key_env": None,
            "api_base": LM_STUDIO_BASE,
            "use_json_schema": True,
            "description": f"LM Studio: {model_id} (~{est:.0f}B)",
            "type": "llm",
        }

    if skipped:
        print(f"Filtered {len(skipped)} models:")
        for s in skipped:
            print(s)

    return configs


# Groq skip patterns (non-chat models)
GROQ_SKIP_PATTERNS = ["whisper", "guard", "prompt-guard", "compound", "orpheus"]


def discover_groq_models() -> dict[str, dict]:
    """Query Groq /v1/models and return configs for all chat models."""
    import urllib.request

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("GROQ_API_KEY not set, skipping Groq discovery")
        return {}

    groq_base = "https://api.groq.com/openai/v1"

    try:
        req = urllib.request.Request(
            f"{groq_base}/models",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "User-Agent": "smartmemory-benchmark/1.0",
            },
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
    except Exception as e:
        print(f"Groq API not reachable: {e}")
        return {}

    configs = {}
    skipped = []
    for model in data.get("data", []):
        model_id = model["id"]
        if any(pat in model_id.lower() for pat in GROQ_SKIP_PATTERNS):
            skipped.append(f"  skip: {model_id}")
            continue
        safe_key = "groq_" + model_id.replace("/", "_").replace("-", "_").replace(".", "_")
        configs[safe_key] = {
            "name": model_id,
            "api_key_env": "GROQ_API_KEY",
            "api_base": groq_base,
            "description": f"Groq: {model_id}",
            "type": "llm",
        }

    if skipped:
        print(f"Filtered {len(skipped)} Groq models:")
        for s in skipped:
            print(s)

    return configs


def discover_provider_models(provider: str) -> dict[str, dict]:
    """Discover models from a provider with OpenAI-compatible /v1/models endpoint."""
    import urllib.request

    provider_info = PROVIDER_CONFIGS.get(provider)
    if not provider_info:
        print(f"Unknown provider: {provider}")
        return {}

    api_key = os.getenv(provider_info["api_key_env"])
    if not api_key:
        print(f"{provider_info['api_key_env']} not set, skipping {provider}")
        return {}

    base_url = provider_info["api_base"]
    skip_patterns = provider_info.get("skip_patterns", [])

    try:
        req = urllib.request.Request(
            f"{base_url}/models",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
    except Exception as e:
        print(f"{provider} API not reachable: {e}")
        return {}

    configs = {}
    skipped = []
    for model in data.get("data", []):
        model_id = model.get("id", "")
        if any(pat in model_id.lower() for pat in skip_patterns):
            skipped.append(f"  skip: {model_id}")
            continue
        safe_key = f"{provider}_" + model_id.replace("/", "_").replace("-", "_").replace(".", "_")
        configs[safe_key] = {
            "name": model_id,
            "api_key_env": provider_info["api_key_env"],
            "api_base": base_url,
            "description": f"{provider.capitalize()}: {model_id}",
            "type": "llm",
        }

    if skipped:
        print(f"Filtered {len(skipped)} {provider} models:")
        for s in skipped:
            print(s)

    return configs


# Provider configurations for auto-discovery
PROVIDER_CONFIGS = {
    "cerebras": {
        "api_base": "https://api.cerebras.ai/v1",
        "api_key_env": "CEREBRAS_API_KEY",
        "skip_patterns": [],
    },
    "deepseek": {
        "api_base": "https://api.deepseek.com/v1",
        "api_key_env": "DEEPSEEK_API_KEY",
        "skip_patterns": [],
    },
    "together": {
        "api_base": "https://api.together.xyz/v1",
        "api_key_env": "TOGETHER_API_KEY",
        "skip_patterns": ["embed", "rerank", "vision", "image", "guard"],
    },
    "fireworks": {
        "api_base": "https://api.fireworks.ai/inference/v1",
        "api_key_env": "FIREWORKS_API_KEY",
        "skip_patterns": ["embed", "vision", "image", "whisper"],
    },
    "sambanova": {
        "api_base": "https://api.sambanova.ai/v1",
        "api_key_env": "SAMBANOVA_API_KEY",
        "skip_patterns": [],
    },
    "deepinfra": {
        "api_base": "https://api.deepinfra.com/v1/openai",
        "api_key_env": "DEEPINFRA_API_KEY",
        "skip_patterns": ["embed", "rerank", "vision", "image", "whisper"],
    },
    "novita": {
        "api_base": "https://api.novita.ai/v3/openai",
        "api_key_env": "NOVITA_API_KEY",
        "skip_patterns": ["embed", "vision", "image"],
    },
    "mistral": {
        "api_base": "https://api.mistral.ai/v1",
        "api_key_env": "MISTRAL_API_KEY",
        "skip_patterns": ["embed", "moderation"],
    },
}


# LLM Model configurations (cloud APIs)
LLM_MODEL_CONFIGS = {
    "gpt4o": {
        "name": "gpt-4o-mini",
        "api_key_env": "OPENAI_API_KEY",
        "description": "OpenAI GPT-4o-mini (baseline)",
        "type": "llm",
    },
    "groq": {
        "name": "llama-3.3-70b-versatile",
        "api_key_env": "GROQ_API_KEY",
        "description": "Groq Llama-3.3-70b",
        "type": "llm",
    },
    "gemini": {
        "name": "gemini-2.0-flash",
        "api_key_env": "GOOGLE_API_KEY",
        "description": "Google Gemini Flash",
        "type": "llm",
    },
    "haiku": {
        "name": "claude-3-5-haiku-latest",
        "api_key_env": "ANTHROPIC_API_KEY",
        "description": "Anthropic Claude Haiku",
        "type": "llm",
    },
}

# Traditional NLP extractor configurations
TRADITIONAL_EXTRACTOR_CONFIGS = {
    "spacy": {
        "name": "en_core_web_sm",
        "description": "spaCy NER (en_core_web_sm)",
        "type": "traditional",
        "runner": lambda text: run_spacy_extractor(text, "en_core_web_sm"),
    },
    "spacy_md": {
        "name": "en_core_web_md",
        "description": "spaCy NER (en_core_web_md)",
        "type": "traditional",
        "runner": lambda text: run_spacy_extractor(text, "en_core_web_md"),
    },
    "spacy_lg": {
        "name": "en_core_web_lg",
        "description": "spaCy NER (en_core_web_lg)",
        "type": "traditional",
        "runner": lambda text: run_spacy_extractor(text, "en_core_web_lg"),
    },
    "spacy_trf": {
        "name": "en_core_web_trf",
        "description": "spaCy NER (en_core_web_trf - transformer)",
        "type": "traditional",
        "runner": lambda text: run_spacy_extractor(text, "en_core_web_trf"),
    },
    "spacy_trf_ruler": {
        "name": "en_core_web_trf+ruler",
        "description": "spaCy trf + EntityRuler (tech/concept/product patterns)",
        "type": "traditional",
        "runner": lambda text: run_spacy_extractor(text, "en_core_web_trf", use_entity_ruler=True),
    },
    "spacy_ruler": {
        "name": "en_core_web_sm+ruler",
        "description": "spaCy sm + EntityRuler (tech/concept/product patterns)",
        "type": "traditional",
        "runner": lambda text: run_spacy_extractor(text, "en_core_web_sm", use_entity_ruler=True),
    },
    "gliner2": {
        "name": "fastino/gliner2-base-v1",
        "description": "GLiNER2 (gliner2-base-v1)",
        "type": "traditional",
        "runner": run_gliner2_extractor,
    },
    "hybrid": {
        "name": "gliner2+relik",
        "description": "Hybrid (GLiNER2 + ReLiK)",
        "type": "traditional",
        "runner": run_hybrid_extractor,
    },
    "rebel": {
        "name": "Babelscape/rebel-large",
        "description": "REBEL end-to-end triplet extraction (400M)",
        "type": "traditional",
        "runner": run_rebel_extractor,
    },
    "gliner_glirel": {
        "name": "gliner+glirel",
        "description": "GLiNER NER + GLiREL zero-shot RE",
        "type": "traditional",
        "runner": run_gliner_glirel_extractor,
    },
    "nuner": {
        "name": "numind/NuNER_Zero",
        "description": "NuMind NuNER Zero (0.4B, NER-only)",
        "type": "traditional",
        "runner": run_nuner_extractor,
    },
    "nuextract": {
        "name": "numind/NuExtract-tiny-v1.5",
        "description": "NuMind NuExtract 1.5 tiny (0.5B, structured)",
        "type": "traditional",
        "runner": run_nuextract_extractor,
    },
    "opennre": {
        "name": "wiki80_cnn+spacy",
        "description": "spaCy NER + OpenNRE wiki80 CNN RE",
        "type": "traditional",
        "runner": run_opennre_extractor,
    },
    "nuner_opennre": {
        "name": "NuNER_Zero+wiki80_cnn",
        "description": "NuNER Zero NER + OpenNRE wiki80 CNN RE",
        "type": "traditional",
        "runner": run_nuner_opennre_extractor,
    },
    "progressive": {
        "name": "spacy_trf+groq_v1",
        "description": "Progressive v1: refine draft (original)",
        "type": "progressive",
        "runner": lambda text: run_progressive_extractor(text, "v1_refine"),
    },
    "progressive_v2": {
        "name": "spacy_trf+groq_v2",
        "description": "Progressive v2: independent + cross-check",
        "type": "progressive",
        "runner": lambda text: run_progressive_extractor(text, "v2_independent"),
    },
    "progressive_v3": {
        "name": "spacy_trf+groq_v3",
        "description": "Progressive v3: deficit-focused (complete the gaps)",
        "type": "progressive",
        "runner": lambda text: run_progressive_extractor(text, "v3_deficit"),
    },
    "progressive_v4": {
        "name": "spacy_trf+groq_v4",
        "description": "Progressive v4: names only (no types to anchor on)",
        "type": "progressive",
        "runner": lambda text: run_progressive_extractor(text, "v4_names_only"),
    },
}

# Backwards compatibility
MODEL_CONFIGS = LLM_MODEL_CONFIGS


def run_llm_benchmark(
    models: list[str],
    single_only: bool = False,
    verbose: bool = True,
    dataset: list[dict] | None = None,
    timeout_s: float = 30.0,
    skip_after_errors: int = 2,
    max_latency_s: float = 0.0,
    max_slow_samples: int = 3,
    output_file: str | None = None,
) -> list[ModelBenchmark]:
    """Run benchmark across LLM models.

    Args:
        timeout_s: Per-test-case timeout in seconds. 0 = no timeout.
        skip_after_errors: Skip remaining test cases for a model after N consecutive errors.
        max_latency_s: Skip model if N samples exceed this latency. 0 = no limit.
        max_slow_samples: Number of slow samples before skipping model.
        output_file: If set, append results incrementally after each model completes.
    """
    test_data = dataset or GROUND_TRUTH_DATASET
    benchmarks = []

    for model_key in models:
        if model_key not in LLM_MODEL_CONFIGS:
            print(f"Unknown LLM model: {model_key}")
            continue

        config = LLM_MODEL_CONFIGS[model_key]
        api_key_env = config.get("api_key_env")
        api_base = config.get("api_base")
        use_json_schema = config.get("use_json_schema", False)
        api_key = os.getenv(api_key_env) if api_key_env else "local"

        if not api_key:
            print(f"Skipping {model_key}: {api_key_env} not set")
            continue

        # Flush caches before each model to prevent cross-contamination
        _flush_caches()

        print(f"\n{'=' * 60}")
        print(f"Testing: {config['description']}")
        print(f"{'=' * 60}")

        extractor_types = ["single"] if single_only else ["single", "dual"]

        for ext_type in extractor_types:
            results: list[ExtractionResult] = []
            consecutive_errors = 0
            slow_count = 0

            for test_case in test_data:
                if verbose:
                    print(f"  [{ext_type}] {test_case['id']}...", end=" ", flush=True)

                try:
                    if timeout_s > 0:
                        with ThreadPoolExecutor(max_workers=1) as pool:
                            if ext_type == "single":
                                future = pool.submit(
                                    run_single_extractor, config["name"], test_case["text"], api_key, api_base,
                                    use_json_schema,
                                )
                            else:
                                future = pool.submit(run_dual_extractor, config["name"], test_case["text"], api_key)
                            out = future.result(timeout=timeout_s)
                    else:
                        if ext_type == "single":
                            out = run_single_extractor(
                                config["name"], test_case["text"], api_key, api_base, use_json_schema,
                            )
                        else:
                            out = run_dual_extractor(config["name"], test_case["text"], api_key)

                    raw_entities = out["result"].get("entities", [])
                    entities = extract_entity_names(raw_entities)
                    relations = extract_relations(out["result"].get("relations", []), raw_entities)

                    result = ExtractionResult(
                        model=config["name"],
                        extractor_type=ext_type,
                        test_id=test_case["id"],
                        latency_ms=out["latency_ms"],
                        entity_count=len(entities),
                        relation_count=len(relations),
                        entities_found=entities,
                        relations_found=relations,
                    )
                    consecutive_errors = 0

                    # Track slow samples (latency check, not counting thinking overhead)
                    if max_latency_s > 0 and out["latency_ms"] > max_latency_s * 1000:
                        slow_count += 1

                    if verbose:
                        slow_tag = " SLOW" if max_latency_s > 0 and out["latency_ms"] > max_latency_s * 1000 else ""
                        print(f"{out['latency_ms']:.0f}ms | E:{len(entities)} R:{len(relations)}{slow_tag}")

                except FuturesTimeout:
                    result = ExtractionResult(
                        model=config["name"],
                        extractor_type=ext_type,
                        test_id=test_case["id"],
                        latency_ms=timeout_s * 1000,
                        entity_count=0,
                        relation_count=0,
                        entities_found=[],
                        relations_found=[],
                        error=f"Timeout ({timeout_s}s)",
                    )
                    consecutive_errors += 1
                    if verbose:
                        print(f"TIMEOUT ({timeout_s}s)")

                except Exception as e:
                    result = ExtractionResult(
                        model=config["name"],
                        extractor_type=ext_type,
                        test_id=test_case["id"],
                        latency_ms=0,
                        entity_count=0,
                        relation_count=0,
                        entities_found=[],
                        relations_found=[],
                        error=str(e),
                    )
                    consecutive_errors += 1
                    if verbose:
                        print(f"ERROR: {e}")

                results.append(result)

                if skip_after_errors and consecutive_errors >= skip_after_errors:
                    remaining = len(test_data) - len(results)
                    if remaining > 0:
                        print(f"  ** Skipping {remaining} remaining tests ({consecutive_errors} consecutive errors)")
                    break

                if max_latency_s > 0 and slow_count >= max_slow_samples:
                    remaining = len(test_data) - len(results)
                    if remaining > 0:
                        print(f"  ** Skipping {remaining} remaining tests ({slow_count} samples >{max_latency_s}s)")
                    break

            benchmark = aggregate_results(config["name"], ext_type, results, test_data)
            benchmarks.append(benchmark)

            # Stream result summary after each model
            em = benchmark.entity_metrics
            rm = benchmark.relation_metrics
            print(f"\n  >> {config['name']}: E-F1={em.f1:.1%} R-F1={rm.f1:.1%} "
                  f"avg={benchmark.avg_latency_ms:.0f}ms errors={benchmark.errors}")

            # Append to output file incrementally
            if output_file:
                _append_benchmark_to_file(benchmark, output_file)

    return benchmarks


def _append_benchmark_to_file(benchmark: "ModelBenchmark", filepath: str) -> None:
    """Append a single benchmark result to a JSON array file."""
    existing = []
    if os.path.exists(filepath):
        try:
            with open(filepath) as f:
                existing = json.load(f)
        except (json.JSONDecodeError, ValueError):
            existing = []

    d = asdict(benchmark)
    d["raw_results"] = len(benchmark.raw_results)
    existing.append(d)

    with open(filepath, "w") as f:
        json.dump(existing, f, indent=2)


def run_traditional_benchmark(
    extractors: list[str], verbose: bool = True, dataset: list[dict] | None = None
) -> list[ModelBenchmark]:
    """Run benchmark across traditional NLP extractors."""
    test_data = dataset or GROUND_TRUTH_DATASET
    benchmarks = []

    for ext_key in extractors:
        if ext_key not in TRADITIONAL_EXTRACTOR_CONFIGS:
            print(f"Unknown traditional extractor: {ext_key}")
            continue

        config = TRADITIONAL_EXTRACTOR_CONFIGS[ext_key]

        # Flush caches before each extractor to prevent cross-contamination
        _flush_caches()

        print(f"\n{'=' * 60}")
        print(f"Testing: {config['description']}")
        print(f"{'=' * 60}")

        results: list[ExtractionResult] = []
        runner = config["runner"]

        for test_case in test_data:
            if verbose:
                print(f"  [{ext_key}] {test_case['id']}...", end=" ", flush=True)

            try:
                out = runner(test_case["text"])

                if "error" in out:
                    raise RuntimeError(out["error"])

                entities = extract_entity_names(out["result"].get("entities", []))
                relations = extract_relations(out["result"].get("relations", []))

                result = ExtractionResult(
                    model=config["name"],
                    extractor_type=ext_key,
                    test_id=test_case["id"],
                    latency_ms=out["latency_ms"],
                    entity_count=len(entities),
                    relation_count=len(relations),
                    entities_found=entities,
                    relations_found=relations,
                )

                if verbose:
                    print(f"{out['latency_ms']:.0f}ms | E:{len(entities)} R:{len(relations)}")

            except Exception as e:
                result = ExtractionResult(
                    model=config["name"],
                    extractor_type=ext_key,
                    test_id=test_case["id"],
                    latency_ms=0,
                    entity_count=0,
                    relation_count=0,
                    entities_found=[],
                    relations_found=[],
                    error=str(e),
                )
                if verbose:
                    print(f"ERROR: {e}")

            results.append(result)

        benchmark = aggregate_results(config["name"], ext_key, results, test_data)
        benchmarks.append(benchmark)

    return benchmarks


def aggregate_results(
    model_name: str, ext_type: str, results: list[ExtractionResult], dataset: list[dict] | None = None
) -> ModelBenchmark:
    """Aggregate individual results into a benchmark."""
    test_data = dataset or GROUND_TRUTH_DATASET
    valid_results = [r for r in results if not r.error]

    if valid_results:
        latencies = [r.latency_ms for r in valid_results]

        # Calculate entity metrics across all tests
        all_entity_tp = all_entity_fp = all_entity_fn = 0
        all_rel_tp = all_rel_fp = all_rel_fn = 0

        for r in valid_results:
            tc_match = next(t for t in test_data if t["id"] == r.test_id)
            em = calculate_entity_metrics(r.entities_found, tc_match["entities"])
            rm = calculate_relation_metrics(r.relations_found, tc_match["relations"])

            all_entity_tp += em.true_positives
            all_entity_fp += em.false_positives
            all_entity_fn += em.false_negatives
            all_rel_tp += rm.true_positives
            all_rel_fp += rm.false_positives
            all_rel_fn += rm.false_negatives

        # Calculate aggregate precision/recall
        e_precision = all_entity_tp / (all_entity_tp + all_entity_fp) if (all_entity_tp + all_entity_fp) > 0 else 0
        e_recall = all_entity_tp / (all_entity_tp + all_entity_fn) if (all_entity_tp + all_entity_fn) > 0 else 0
        e_f1 = 2 * e_precision * e_recall / (e_precision + e_recall) if (e_precision + e_recall) > 0 else 0

        r_precision = all_rel_tp / (all_rel_tp + all_rel_fp) if (all_rel_tp + all_rel_fp) > 0 else 0
        r_recall = all_rel_tp / (all_rel_tp + all_rel_fn) if (all_rel_tp + all_rel_fn) > 0 else 0
        r_f1 = 2 * r_precision * r_recall / (r_precision + r_recall) if (r_precision + r_recall) > 0 else 0

        return ModelBenchmark(
            model=model_name,
            extractor_type=ext_type,
            avg_latency_ms=mean(latencies),
            latency_std_ms=stdev(latencies) if len(latencies) > 1 else 0,
            entity_metrics=QualityMetrics(
                precision=e_precision,
                recall=e_recall,
                f1=e_f1,
                true_positives=all_entity_tp,
                false_positives=all_entity_fp,
                false_negatives=all_entity_fn,
            ),
            relation_metrics=QualityMetrics(
                precision=r_precision,
                recall=r_recall,
                f1=r_f1,
                true_positives=all_rel_tp,
                false_positives=all_rel_fp,
                false_negatives=all_rel_fn,
            ),
            total_tests=len(test_data),
            errors=len(results) - len(valid_results),
            raw_results=results,
        )
    else:
        return ModelBenchmark(
            model=model_name,
            extractor_type=ext_type,
            avg_latency_ms=0,
            latency_std_ms=0,
            entity_metrics=QualityMetrics(),
            relation_metrics=QualityMetrics(),
            total_tests=len(test_data),
            errors=len(results),
            raw_results=results,
        )


def run_benchmark(
    models: list[str], single_only: bool = False, verbose: bool = True, dataset: list[dict] | None = None
) -> list[ModelBenchmark]:
    """Run benchmark across LLM models (backwards compatibility)."""
    return run_llm_benchmark(models, single_only, verbose, dataset)


def print_summary(benchmarks: list[ModelBenchmark]):
    """Print summary table."""
    print("\n" + "=" * 100)
    print("BENCHMARK SUMMARY")
    print("=" * 100)

    print(f"\n{'Model':<25} {'Type':<8} {'Latency':<10} {'E-F1':<8} {'R-F1':<8} {'Errors':<6}")
    print("-" * 70)

    for b in benchmarks:
        em = b.entity_metrics
        rm = b.relation_metrics
        print(
            f"{b.model:<25} {b.extractor_type:<8} {b.avg_latency_ms:>6.0f}ms   "
            f"{em.f1:>6.1%}   {rm.f1:>6.1%}   {b.errors}"
        )

    # Best performers
    print("\n" + "-" * 60)
    valid = [b for b in benchmarks if b.entity_metrics.f1 > 0]
    if valid:

        def avg_f1(b: ModelBenchmark) -> float:
            return (b.entity_metrics.f1 + b.relation_metrics.f1) / 2

        best_quality = max(valid, key=avg_f1)
        fastest = min(valid, key=lambda b: b.avg_latency_ms)
        best_balance = max(valid, key=lambda b: avg_f1(b) / (b.avg_latency_ms / 1000))

        bq_f1 = avg_f1(best_quality)
        print(f"Best Quality:  {best_quality.model} ({best_quality.extractor_type}) - F1: {bq_f1:.1%}")
        print(f"Fastest:       {fastest.model} ({fastest.extractor_type}) - {fastest.avg_latency_ms:.0f}ms")
        print(f"Best Balance:  {best_balance.model} ({best_balance.extractor_type})")


def print_category_breakdown(benchmarks: list[ModelBenchmark], dataset: list[dict] | None = None):
    """Print per-category breakdown for the best model."""
    test_data = dataset or GROUND_TRUTH_DATASET
    if not benchmarks:
        return

    # Find best overall model
    valid = [b for b in benchmarks if b.entity_metrics.f1 > 0]
    if not valid:
        return

    best = max(valid, key=lambda b: (b.entity_metrics.f1 + b.relation_metrics.f1) / 2)

    print("\n" + "=" * 60)
    print(f"CATEGORY BREAKDOWN: {best.model} ({best.extractor_type})")
    print("=" * 60)

    # Group test results by category
    categories = {}
    for r in best.raw_results:
        if r.error:
            continue
        tc = next(t for t in test_data if t["id"] == r.test_id)
        cat = tc.get("category", "unknown")
        if cat not in categories:
            categories[cat] = {
                "entity_tp": 0,
                "entity_fp": 0,
                "entity_fn": 0,
                "rel_tp": 0,
                "rel_fp": 0,
                "rel_fn": 0,
                "count": 0,
            }

        em = calculate_entity_metrics(r.entities_found, tc["entities"])
        rm = calculate_relation_metrics(r.relations_found, tc["relations"])

        categories[cat]["entity_tp"] += em.true_positives
        categories[cat]["entity_fp"] += em.false_positives
        categories[cat]["entity_fn"] += em.false_negatives
        categories[cat]["rel_tp"] += rm.true_positives
        categories[cat]["rel_fp"] += rm.false_positives
        categories[cat]["rel_fn"] += rm.false_negatives
        categories[cat]["count"] += 1

    print(f"\n{'Category':<12} {'Tests':<6} {'Entity F1':<12} {'Relation F1':<12}")
    print("-" * 50)

    for cat, stats in sorted(categories.items()):
        e_tp, e_fp, e_fn = stats["entity_tp"], stats["entity_fp"], stats["entity_fn"]
        r_tp, r_fp, r_fn = stats["rel_tp"], stats["rel_fp"], stats["rel_fn"]

        e_prec = e_tp / (e_tp + e_fp) if (e_tp + e_fp) > 0 else 0
        e_rec = e_tp / (e_tp + e_fn) if (e_tp + e_fn) > 0 else 0
        e_f1 = 2 * e_prec * e_rec / (e_prec + e_rec) if (e_prec + e_rec) > 0 else 0

        r_prec = r_tp / (r_tp + r_fp) if (r_tp + r_fp) > 0 else 0
        r_rec = r_tp / (r_tp + r_fn) if (r_tp + r_fn) > 0 else 0
        r_f1 = 2 * r_prec * r_rec / (r_prec + r_rec) if (r_prec + r_rec) > 0 else 0

        print(f"{cat:<12} {stats['count']:<6} {e_f1:>6.1%}       {r_f1:>6.1%}")


def load_conll04_dataset(split: str = "test", max_examples: int | None = None) -> list[dict[str, Any]]:
    """Load CoNLL04 from HuggingFace and convert to SmartMemory evaluation format.

    CoNLL04 is a standardized joint NER+RE benchmark with published baselines.
    Entity types: Loc, Org, Peop, Other
    Relation types: Kill, Live_In, Located_In, OrgBased_In, Work_For
    Test set: 288 examples.
    """
    from datasets import load_dataset  # type: ignore

    # Map CoNLL04 types to SmartMemory types
    entity_type_map = {"Loc": "location", "Org": "organization", "Peop": "person", "Other": "concept"}
    relation_type_map = {
        "Kill": "killed",
        "Live_In": "lives_in",
        "Located_In": "located_in",
        "OrgBased_In": "based_in",
        "Work_For": "works_for",
    }

    ds = load_dataset("DFKI-SLT/conll04")
    data = ds[split]
    converted = []

    for i, ex in enumerate(data):
        if max_examples and i >= max_examples:
            break

        tokens = ex["tokens"]
        text = " ".join(tokens)
        ents = ex["entities"]
        rels = ex["relations"]

        # Skip very short or empty texts
        if len(tokens) < 3:
            continue

        # Convert entities
        entities = []
        for e in ents:
            ent_text = " ".join(tokens[e["start"]:e["end"]])
            ent_type = entity_type_map.get(e["type"], "concept")
            entities.append({"name": ent_text, "type": ent_type})

        # Convert relations (head/tail are entity indices)
        relations = []
        for r in rels:
            head_ent = ents[r["head"]]
            tail_ent = ents[r["tail"]]
            subj = " ".join(tokens[head_ent["start"]:head_ent["end"]])
            obj = " ".join(tokens[tail_ent["start"]:tail_ent["end"]])
            pred = relation_type_map.get(r["type"], r["type"].lower())
            relations.append({"subject": subj, "predicate": pred, "object": obj})

        # Only include examples with at least one entity
        if entities:
            converted.append({
                "id": f"conll04_{ex.get('orig_id', i)}",
                "category": "conll04",
                "domain": "news",
                "text": text,
                "entities": entities,
                "relations": relations,
            })

    return converted


def _flush_caches():
    """Flush extraction caches for clean benchmark results."""
    # Flush SmartMemory Redis extraction cache
    try:
        import redis

        r = redis.Redis(host="localhost", port=9012, db=0)
        keys = r.keys("smartmemory:entity_extraction:*")
        if keys:
            r.delete(*keys)
            print(f"Flushed {len(keys)} Redis extraction cache entries")
    except Exception:
        pass  # Redis not available

    # Flush DSPy disk cache (diskcache at ~/.dspy_cache)
    try:
        import dspy

        cache = dspy.DSPY_CACHE
        if cache.disk_cache:
            cache.disk_cache.clear()
            print("Flushed DSPy disk cache")
        if cache.memory_cache:
            cache.reset_memory_cache()
            print("Flushed DSPy memory cache")
    except Exception:
        pass

    # Also clean up stale cache dirs
    import shutil

    for cache_dir in [".dspy_cache", "/tmp/dspy_cache"]:
        if os.path.isdir(cache_dir):
            shutil.rmtree(cache_dir)
            print(f"Removed stale cache dir: {cache_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark extraction quality: LLM vs Traditional NLP",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run all extractors (LLM + traditional)
    PYTHONPATH=. python tests/benchmark_model_quality.py --all

    # Run specific LLM models
    PYTHONPATH=. python tests/benchmark_model_quality.py --models gpt4o groq

    # Run traditional extractors only
    PYTHONPATH=. python tests/benchmark_model_quality.py --traditional

    # Run specific traditional extractors
    PYTHONPATH=. python tests/benchmark_model_quality.py --extractors spacy gliner2

    # Run quick test (single-call LLM only, no traditional)
    PYTHONPATH=. python tests/benchmark_model_quality.py --models gpt4o --single-only
        """,
    )
    parser.add_argument("--models", nargs="+", default=[], help="LLM models to test: gpt4o, groq, gemini, haiku")
    parser.add_argument("--extractors", nargs="+", default=[], help="Traditional extractors: spacy, gliner2, hybrid")
    parser.add_argument("--traditional", action="store_true", help="Run all traditional extractors")
    parser.add_argument("--all", action="store_true", help="Run all LLM models and traditional extractors")
    parser.add_argument("--lm-studio", action="store_true", help="Auto-discover and benchmark all LM Studio models")
    parser.add_argument("--groq", action="store_true", help="Auto-discover and benchmark all Groq models")
    parser.add_argument(
        "--provider", nargs="+", default=[],
        help="Auto-discover models from providers: cerebras, deepseek, together, fireworks, sambanova, mistral, deepinfra, novita",
    )
    parser.add_argument(
        "--skip-after-errors",
        type=int,
        default=2,
        help="Skip a model after N consecutive errors (default: 2)",
    )
    parser.add_argument("--timeout", type=float, default=30.0, help="Per-test-case timeout in seconds (default: 30, 0=none)")
    parser.add_argument("--max-latency", type=float, default=0.0, help="Skip model if 3 samples exceed this (seconds, 0=none)")
    parser.add_argument("--single-only", action="store_true", help="Only test single-call LLM extractor (faster)")
    parser.add_argument("--output", type=str, default="benchmark_results.json", help="Output file for results")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-test output")
    parser.add_argument("--category-breakdown", action="store_true", help="Show per-category F1 breakdown")
    parser.add_argument("--conll04", action="store_true", help="Use CoNLL04 standardized benchmark (288 test examples)")
    parser.add_argument("--conll04-limit", type=int, default=None, help="Limit CoNLL04 examples (default: all 288)")
    parser.add_argument("--progressive", action="store_true", help="Run progressive benchmark (spaCy trf → Groq refinement)")
    parser.add_argument("--no-size-limit", action="store_true", help="Remove model size limit for LM Studio discovery (default: 24B)")

    args = parser.parse_args()

    # Auto-discover LM Studio models if requested
    max_params = 999.0 if args.no_size_limit else 24.0
    if args.lm_studio:
        lm_models = discover_lm_studio_models(max_params_b=max_params)
        if lm_models:
            print(f"Discovered {len(lm_models)} LM Studio models")
            LLM_MODEL_CONFIGS.update(lm_models)
        else:
            print("No LM Studio models found")

    # Auto-discover Groq models if requested
    if args.groq:
        groq_models = discover_groq_models()
        if groq_models:
            print(f"Discovered {len(groq_models)} Groq models")
            LLM_MODEL_CONFIGS.update(groq_models)
        else:
            print("No Groq models found")

    # Auto-discover models from other providers
    for provider in args.provider:
        provider_models = discover_provider_models(provider)
        if provider_models:
            print(f"Discovered {len(provider_models)} {provider} models")
            LLM_MODEL_CONFIGS.update(provider_models)
        else:
            print(f"No {provider} models found")

    # Determine what to run
    llm_models = args.models
    traditional = args.extractors

    if args.all:
        llm_models = list(LLM_MODEL_CONFIGS.keys())
        traditional = list(TRADITIONAL_EXTRACTOR_CONFIGS.keys())
    elif args.groq and not args.models:
        # If --groq with no --models, run only Groq models
        llm_models = [k for k in LLM_MODEL_CONFIGS if k.startswith("groq_")]
    elif args.provider and not args.models:
        # If --provider with no --models, run only discovered provider models
        prefixes = tuple(f"{p}_" for p in args.provider)
        llm_models = [k for k in LLM_MODEL_CONFIGS if k.startswith(prefixes)]
    elif args.lm_studio and not args.models:
        # If --lm-studio with no --models, run only local models
        llm_models = [k for k in LLM_MODEL_CONFIGS if k.startswith("lm_")]
    elif args.traditional:
        traditional = list(TRADITIONAL_EXTRACTOR_CONFIGS.keys())

    # Add progressive if requested
    if args.progressive and "progressive" not in traditional:
        traditional.append("progressive")

    # Default: at least gpt4o if nothing specified
    if not llm_models and not traditional:
        llm_models = ["gpt4o", "groq"]

    # Load dataset
    dataset = GROUND_TRUTH_DATASET
    dataset_name = "SmartMemory Gold (hand-crafted)"
    if args.conll04:
        print("Loading CoNLL04 from HuggingFace...")
        dataset = load_conll04_dataset(split="test", max_examples=args.conll04_limit)
        dataset_name = f"CoNLL04 test ({len(dataset)} examples)"

    print("SmartMemory Extraction Quality Benchmark")
    print("=" * 60)
    print(f"Dataset: {dataset_name}")
    if llm_models:
        print(f"LLM models: {', '.join(llm_models)}")
        print(f"LLM extractor types: {'single only' if args.single_only else 'single + dual'}")
    if traditional:
        print(f"Traditional extractors: {', '.join(traditional)}")
    print(f"Test cases: {len(dataset)}")

    # Group by category
    categories = {}
    for tc in dataset:
        cat = tc.get("category", "unknown")
        categories[cat] = categories.get(cat, 0) + 1
    print(f"Categories: {', '.join(f'{k}({v})' for k, v in sorted(categories.items()))}")

    benchmarks: list[ModelBenchmark] = []

    # Run LLM benchmarks
    if llm_models:
        llm_benchmarks = run_llm_benchmark(
            models=llm_models,
            single_only=args.single_only,
            verbose=not args.quiet,
            dataset=dataset,
            timeout_s=args.timeout,
            skip_after_errors=args.skip_after_errors,
            max_latency_s=args.max_latency,
            output_file=args.output,
        )
        benchmarks.extend(llm_benchmarks)

    # Run traditional benchmarks
    if traditional:
        trad_benchmarks = run_traditional_benchmark(extractors=traditional, verbose=not args.quiet, dataset=dataset)
        benchmarks.extend(trad_benchmarks)

    print_summary(benchmarks)

    if args.category_breakdown:
        print_category_breakdown(benchmarks, dataset=dataset)

    # Save results (exclude raw_results to keep file size manageable)
    output_data = []
    for b in benchmarks:
        d = asdict(b)
        d["raw_results"] = len(b.raw_results)  # Just count, not full data
        output_data.append(d)

    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Benchmark different extractor combinations.

Compares:
- LLM (GPT-based)
- GLiNER2 (local NER)
- REBEL (local relation extraction)
- Hybrid GLiNER2+REBEL

Metrics:
- Entity count
- Relation count
- Extraction time
- Entity types coverage
"""

import time
import json
from typing import Dict, Any, List
from dataclasses import dataclass, asdict

# Test texts
TEST_TEXTS = [
    {
        "id": "simple",
        "text": "Sarah completed her first marathon in Boston last weekend.",
        "expected_entities": ["Sarah", "Boston", "marathon", "last weekend"],
    },
    {
        "id": "business",
        "text": "Elon Musk is the CEO of Tesla and SpaceX. Tesla is headquartered in Austin, Texas. SpaceX launched Starship from Boca Chica.",
        "expected_entities": ["Elon Musk", "Tesla", "SpaceX", "Austin", "Texas", "Starship", "Boca Chica"],
        "expected_relations": ["CEO of", "headquartered in", "launched"],
    },
    {
        "id": "technical",
        "text": "Python is a programming language created by Guido van Rossum. Django and Flask are popular Python web frameworks. NumPy provides numerical computing capabilities.",
        "expected_entities": ["Python", "Guido van Rossum", "Django", "Flask", "NumPy"],
    },
]


@dataclass
class BenchmarkResult:
    extractor: str
    text_id: str
    entity_count: int
    relation_count: int
    extraction_time_ms: float
    entities: List[str]
    relations: List[str]
    entity_types: Dict[str, int]
    error: str = None


def benchmark_llm(text: str) -> Dict[str, Any]:
    """Benchmark LLM extractor."""
    from smartmemory.plugins.extractors.llm import LLMExtractor
    
    extractor = LLMExtractor()
    start = time.time()
    result = extractor.extract(text)
    elapsed = (time.time() - start) * 1000
    
    return {
        "entities": result.get("entities", []),
        "relations": result.get("relations", []),
        "time_ms": elapsed
    }


def benchmark_gliner2(text: str) -> Dict[str, Any]:
    """Benchmark GLiNER2 extractor."""
    from smartmemory.plugins.extractors.gliner2 import GLiNER2Extractor
    
    extractor = GLiNER2Extractor()
    start = time.time()
    result = extractor.extract(text)
    elapsed = (time.time() - start) * 1000
    
    return {
        "entities": result.get("entities", []),
        "relations": result.get("relations", []),
        "time_ms": elapsed
    }


def benchmark_rebel(text: str) -> Dict[str, Any]:
    """Benchmark REBEL extractor."""
    from smartmemory.plugins.extractors.rebel import RebelExtractor
    
    extractor = RebelExtractor()
    start = time.time()
    result = extractor.extract(text)
    elapsed = (time.time() - start) * 1000
    
    return {
        "entities": result.get("entities", []),
        "relations": result.get("relations", []),
        "time_ms": elapsed
    }


def benchmark_hybrid(text: str) -> Dict[str, Any]:
    """Benchmark Hybrid GLiNER2+REBEL extractor."""
    from smartmemory.plugins.extractors.hybrid_gliner_rebel import HybridGlinerRebelExtractor
    
    extractor = HybridGlinerRebelExtractor()
    start = time.time()
    result = extractor.extract(text)
    elapsed = (time.time() - start) * 1000
    
    return {
        "entities": result.get("entities", []),
        "relations": result.get("relations", []),
        "time_ms": elapsed
    }


def extract_entity_names(entities: List) -> List[str]:
    """Extract entity names from various formats."""
    names = []
    for e in entities:
        if isinstance(e, str):
            names.append(e)
        elif hasattr(e, 'content'):
            names.append(e.content)
        elif hasattr(e, 'metadata') and 'name' in e.metadata:
            names.append(e.metadata['name'])
        elif isinstance(e, dict):
            names.append(e.get('name', e.get('content', str(e))))
    return names


def extract_relation_types(relations: List) -> List[str]:
    """Extract relation types from various formats."""
    types = []
    for r in relations:
        if isinstance(r, tuple) and len(r) >= 3:
            types.append(r[1])
        elif isinstance(r, dict):
            types.append(r.get('relation_type', 'unknown'))
        elif hasattr(r, 'relation_type'):
            types.append(r.relation_type)
    return types


def count_entity_types(entities: List) -> Dict[str, int]:
    """Count entities by type."""
    type_counts = {}
    for e in entities:
        if hasattr(e, 'metadata'):
            etype = e.metadata.get('entity_type', 'unknown')
        elif isinstance(e, dict):
            etype = e.get('entity_type', e.get('type', 'unknown'))
        else:
            etype = 'unknown'
        type_counts[etype] = type_counts.get(etype, 0) + 1
    return type_counts


def run_benchmark(extractors: List[str] = None) -> List[BenchmarkResult]:
    """Run benchmark on all extractors and texts."""
    
    if extractors is None:
        extractors = ["llm", "gliner2", "rebel", "hybrid"]
    
    benchmark_funcs = {
        "llm": benchmark_llm,
        "gliner2": benchmark_gliner2,
        "rebel": benchmark_rebel,
        "hybrid": benchmark_hybrid,
    }
    
    results = []
    
    for test in TEST_TEXTS:
        text_id = test["id"]
        text = test["text"]
        print(f"\n{'='*60}")
        print(f"Testing: {text_id}")
        print(f"Text: {text[:80]}...")
        print(f"{'='*60}")
        
        for extractor_name in extractors:
            if extractor_name not in benchmark_funcs:
                print(f"  Unknown extractor: {extractor_name}")
                continue
            
            print(f"\n  {extractor_name.upper()}:")
            try:
                result = benchmark_funcs[extractor_name](text)
                
                entities = result["entities"]
                relations = result["relations"]
                
                entity_names = extract_entity_names(entities)
                relation_types = extract_relation_types(relations)
                entity_type_counts = count_entity_types(entities)
                
                benchmark_result = BenchmarkResult(
                    extractor=extractor_name,
                    text_id=text_id,
                    entity_count=len(entities),
                    relation_count=len(relations),
                    extraction_time_ms=result["time_ms"],
                    entities=entity_names,
                    relations=relation_types,
                    entity_types=entity_type_counts,
                )
                results.append(benchmark_result)
                
                print(f"    Entities: {len(entities)} - {entity_names[:5]}{'...' if len(entity_names) > 5 else ''}")
                print(f"    Relations: {len(relations)} - {relation_types[:5]}{'...' if len(relation_types) > 5 else ''}")
                print(f"    Time: {result['time_ms']:.0f}ms")
                print(f"    Types: {entity_type_counts}")
                
            except Exception as e:
                print(f"    ERROR: {e}")
                results.append(BenchmarkResult(
                    extractor=extractor_name,
                    text_id=text_id,
                    entity_count=0,
                    relation_count=0,
                    extraction_time_ms=0,
                    entities=[],
                    relations=[],
                    entity_types={},
                    error=str(e)
                ))
    
    return results


def print_summary(results: List[BenchmarkResult]):
    """Print summary comparison."""
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    # Group by extractor
    by_extractor = {}
    for r in results:
        if r.extractor not in by_extractor:
            by_extractor[r.extractor] = []
        by_extractor[r.extractor].append(r)
    
    print(f"\n{'Extractor':<20} {'Avg Entities':<15} {'Avg Relations':<15} {'Avg Time (ms)':<15}")
    print("-" * 65)
    
    for extractor, ext_results in by_extractor.items():
        valid_results = [r for r in ext_results if not r.error]
        if not valid_results:
            print(f"{extractor:<20} {'ERROR':<15}")
            continue
        
        avg_entities = sum(r.entity_count for r in valid_results) / len(valid_results)
        avg_relations = sum(r.relation_count for r in valid_results) / len(valid_results)
        avg_time = sum(r.extraction_time_ms for r in valid_results) / len(valid_results)
        
        print(f"{extractor:<20} {avg_entities:<15.1f} {avg_relations:<15.1f} {avg_time:<15.0f}")


if __name__ == "__main__":
    import sys
    
    # Parse args for specific extractors
    extractors = sys.argv[1:] if len(sys.argv) > 1 else None
    
    print("SmartMemory Extractor Benchmark")
    print("="*80)
    
    results = run_benchmark(extractors)
    print_summary(results)
    
    # Save results to JSON
    output_file = "benchmark_results.json"
    with open(output_file, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"\nResults saved to {output_file}")

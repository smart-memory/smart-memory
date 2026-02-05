#!/usr/bin/env python3
"""
Detailed error analysis for Gemma-3-27b-it extraction.

Runs extraction on all 16 benchmark test cases and produces per-test-case
breakdown of: matched entities, missed entities (FN), hallucinated entities (FP),
matched relations, missed relations, hallucinated relations.

Saves full results to benchmark_error_analysis.json for inspection.
"""

import json
import sys
import time

sys.path.insert(0, ".")

from tests.benchmark_model_quality import GROUND_TRUTH_DATASET, normalize_name
from smartmemory.plugins.extractors.llm_single import LLMSingleExtractor


def fuzzy_name_match(a: str, b: str) -> bool:
    """Check if two entity names match (case-insensitive, substring)."""
    a, b = a.lower().strip(), b.lower().strip()
    if a == b:
        return True
    if len(a) >= 4 and len(b) >= 4:
        return a in b or b in a
    return False


def analyze_entity_errors(predicted: list[dict], gold_entities: list[dict]) -> dict:
    """Detailed entity error analysis. predicted is list of {name, type}."""
    pred_items = {normalize_name(p["name"]): p for p in predicted}
    gold_items = {normalize_name(g["name"]): g for g in gold_entities}

    matched = []
    matched_pred = set()
    matched_gold = set()

    for gn, ge in gold_items.items():
        for pn, pe in pred_items.items():
            if gn == pn and gn not in matched_gold and pn not in matched_pred:
                matched.append({
                    "gold_name": ge["name"], "gold_type": ge["type"],
                    "pred_name": pe["name"], "pred_type": pe["type"],
                    "type_match": ge["type"] == pe["type"],
                })
                matched_gold.add(gn)
                matched_pred.add(pn)
                break

    missed = [{"name": ge["name"], "type": ge["type"]}
              for gn, ge in gold_items.items() if gn not in matched_gold]
    hallucinated = [{"name": pe["name"], "type": pe["type"]}
                    for pn, pe in pred_items.items() if pn not in matched_pred]

    return {
        "matched": matched,
        "missed": missed,
        "hallucinated": hallucinated,
        "tp": len(matched), "fn": len(missed), "fp": len(hallucinated),
    }


def analyze_relation_errors(predicted_triples: list[tuple], gold_relations: list[dict]) -> dict:
    """Detailed relation error analysis."""
    gold_pairs = [(r["subject"], r["predicate"], r["object"]) for r in gold_relations]

    matched = []
    missed = []
    hallucinated = []
    matched_gold = set()
    matched_pred = set()

    for gi, (gs, gp, go) in enumerate(gold_pairs):
        for pi, (ps, pp, po) in enumerate(predicted_triples):
            if gi in matched_gold or pi in matched_pred:
                continue
            fwd = fuzzy_name_match(ps, gs) and fuzzy_name_match(po, go)
            rev = fuzzy_name_match(ps, go) and fuzzy_name_match(po, gs)
            if fwd or rev:
                matched.append({
                    "gold": f"{gs} --[{gp}]--> {go}",
                    "predicted": f"{ps} --[{pp}]--> {po}",
                    "predicate_match": normalize_name(gp).replace("_", " ") == normalize_name(pp).replace("_", " "),
                })
                matched_gold.add(gi)
                matched_pred.add(pi)
                break

    for gi, (gs, gp, go) in enumerate(gold_pairs):
        if gi not in matched_gold:
            missed.append(f"{gs} --[{gp}]--> {go}")

    for pi, (ps, pp, po) in enumerate(predicted_triples):
        if pi not in matched_pred:
            hallucinated.append(f"{ps} --[{pp}]--> {po}")

    return {
        "matched": matched,
        "missed": missed,
        "hallucinated": hallucinated,
        "tp": len(matched), "fn": len(missed), "fp": len(hallucinated),
    }


def main():
    model = "gemma-3-27b-it"
    api_base = "http://localhost:1234/v1"
    api_key = "lm-studio"

    extractor = LLMSingleExtractor(
        model_name=model, api_key=api_key, api_base_url=api_base,
    )
    extractor.cfg.use_json_schema = True

    all_results = []
    totals = {"entity_tp": 0, "entity_fp": 0, "entity_fn": 0, "rel_tp": 0, "rel_fp": 0, "rel_fn": 0}

    print(f"Running {model} on {len(GROUND_TRUTH_DATASET)} test cases with json_schema mode...")
    print("=" * 80)

    for i, tc in enumerate(GROUND_TRUTH_DATASET):
        print(f"\n[{i+1}/{len(GROUND_TRUTH_DATASET)}] {tc['id']} ({tc['category']}/{tc['domain']})")
        print(f"  Text: {tc['text'][:100]}...")

        try:
            start = time.time()
            result = extractor.extract(tc["text"])
            latency = (time.time() - start) * 1000

            # Extract entity names and types
            raw_entities = result.get("entities", [])
            pred_entities = []
            for e in raw_entities:
                name = e.metadata.get("name", e.content) if hasattr(e, "metadata") else str(e)
                etype = e.metadata.get("entity_type", "unknown") if hasattr(e, "metadata") else "unknown"
                pred_entities.append({"name": name, "type": etype})

            # Extract relation triples
            raw_relations = result.get("relations", [])
            # Build id -> name map
            id_to_name = {}
            for e in raw_entities:
                if hasattr(e, "item_id") and hasattr(e, "metadata"):
                    id_to_name[e.item_id] = e.metadata.get("name", e.content)

            pred_triples = []
            for r in raw_relations:
                if isinstance(r, dict):
                    sid = r.get("source_id", "")
                    oid = r.get("target_id", "")
                    pred = r.get("relation_type", "")
                    subj = id_to_name.get(sid, sid)
                    obj = id_to_name.get(oid, oid)
                    pred_triples.append((subj, pred, obj))

            # Analyze errors
            ent_analysis = analyze_entity_errors(pred_entities, tc["entities"])
            rel_analysis = analyze_relation_errors(pred_triples, tc["relations"])

            totals["entity_tp"] += ent_analysis["tp"]
            totals["entity_fp"] += ent_analysis["fp"]
            totals["entity_fn"] += ent_analysis["fn"]
            totals["rel_tp"] += rel_analysis["tp"]
            totals["rel_fp"] += rel_analysis["fp"]
            totals["rel_fn"] += rel_analysis["fn"]

            # Print summary
            print(f"  Latency: {latency:.0f}ms")
            print(f"  Entities extracted: {[e['name'] for e in pred_entities]}")
            print(f"  Entities: {ent_analysis['tp']} TP, {ent_analysis['fp']} FP, {ent_analysis['fn']} FN")
            if ent_analysis["missed"]:
                for m in ent_analysis["missed"]:
                    print(f"    MISSED ENTITY: {m['name']} ({m['type']})")
            if ent_analysis["hallucinated"]:
                for h in ent_analysis["hallucinated"]:
                    print(f"    HALLUC ENTITY: {h['name']} ({h['type']})")
            # Type mismatches
            for m in ent_analysis["matched"]:
                if not m["type_match"]:
                    print(f"    TYPE MISMATCH: {m['gold_name']} gold={m['gold_type']} pred={m['pred_type']}")

            print(f"  Relations: {rel_analysis['tp']} TP, {rel_analysis['fp']} FP, {rel_analysis['fn']} FN")
            if rel_analysis["missed"]:
                for m in rel_analysis["missed"]:
                    print(f"    MISSED REL: {m}")
            if rel_analysis["hallucinated"]:
                for h in rel_analysis["hallucinated"]:
                    print(f"    HALLUC REL: {h}")

            all_results.append({
                "test_id": tc["id"],
                "category": tc["category"],
                "domain": tc["domain"],
                "text": tc["text"],
                "latency_ms": latency,
                "predicted_entities": pred_entities,
                "predicted_relations": [f"{s} --[{p}]--> {o}" for s, p, o in pred_triples],
                "entity_analysis": ent_analysis,
                "relation_analysis": rel_analysis,
            })

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            all_results.append({"test_id": tc["id"], "error": str(e)})

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    e_tp, e_fp, e_fn = totals["entity_tp"], totals["entity_fp"], totals["entity_fn"]
    r_tp, r_fp, r_fn = totals["rel_tp"], totals["rel_fp"], totals["rel_fn"]

    e_prec = e_tp / (e_tp + e_fp) if (e_tp + e_fp) > 0 else 0
    e_rec = e_tp / (e_tp + e_fn) if (e_tp + e_fn) > 0 else 0
    e_f1 = 2 * e_prec * e_rec / (e_prec + e_rec) if (e_prec + e_rec) > 0 else 0

    r_prec = r_tp / (r_tp + r_fp) if (r_tp + r_fp) > 0 else 0
    r_rec = r_tp / (r_tp + r_fn) if (r_tp + r_fn) > 0 else 0
    r_f1 = 2 * r_prec * r_rec / (r_prec + r_rec) if (r_prec + r_rec) > 0 else 0

    print(f"\nEntities: P={e_prec:.1%} R={e_rec:.1%} F1={e_f1:.1%}  (TP={e_tp} FP={e_fp} FN={e_fn})")
    print(f"Relations: P={r_prec:.1%} R={r_rec:.1%} F1={r_f1:.1%}  (TP={r_tp} FP={r_fp} FN={r_fn})")

    # Aggregate errors
    all_missed_ent = []
    all_halluc_ent = []
    all_missed_rel = []
    all_halluc_rel = []
    all_type_mismatches = []

    for r in all_results:
        if "error" in r:
            continue
        tid = r["test_id"]
        for m in r["entity_analysis"]["missed"]:
            all_missed_ent.append({**m, "test_id": tid})
        for h in r["entity_analysis"]["hallucinated"]:
            all_halluc_ent.append({**h, "test_id": tid})
        for m in r["entity_analysis"]["matched"]:
            if not m["type_match"]:
                all_type_mismatches.append({**m, "test_id": tid})
        for m in r["relation_analysis"]["missed"]:
            all_missed_rel.append({"relation": m, "test_id": tid})
        for h in r["relation_analysis"]["hallucinated"]:
            all_halluc_rel.append({"relation": h, "test_id": tid})

    print(f"\n{'='*60}")
    print(f"ALL MISSED ENTITIES ({len(all_missed_ent)})")
    print(f"{'='*60}")
    for m in all_missed_ent:
        print(f"  [{m['test_id']}] {m['name']} ({m['type']})")

    print(f"\n{'='*60}")
    print(f"ALL HALLUCINATED ENTITIES ({len(all_halluc_ent)})")
    print(f"{'='*60}")
    for h in all_halluc_ent:
        print(f"  [{h['test_id']}] {h['name']} ({h['type']})")

    print(f"\n{'='*60}")
    print(f"ALL TYPE MISMATCHES ({len(all_type_mismatches)})")
    print(f"{'='*60}")
    for m in all_type_mismatches:
        print(f"  [{m['test_id']}] {m['gold_name']}: gold={m['gold_type']} pred={m['pred_type']}")

    print(f"\n{'='*60}")
    print(f"ALL MISSED RELATIONS ({len(all_missed_rel)})")
    print(f"{'='*60}")
    for m in all_missed_rel:
        print(f"  [{m['test_id']}] {m['relation']}")

    print(f"\n{'='*60}")
    print(f"ALL HALLUCINATED RELATIONS ({len(all_halluc_rel)})")
    print(f"{'='*60}")
    for h in all_halluc_rel:
        print(f"  [{h['test_id']}] {h['relation']}")

    # Error pattern categorization
    missed_by_type = {}
    for m in all_missed_ent:
        t = m["type"]
        missed_by_type[t] = missed_by_type.get(t, 0) + 1

    if missed_by_type:
        print(f"\n{'='*60}")
        print("MISSED ENTITY TYPE DISTRIBUTION")
        print(f"{'='*60}")
        for t, count in sorted(missed_by_type.items(), key=lambda x: -x[1]):
            print(f"  {t}: {count}")

    mismatch_patterns = {}
    for m in all_type_mismatches:
        key = f"{m['gold_type']} -> {m['pred_type']}"
        mismatch_patterns[key] = mismatch_patterns.get(key, 0) + 1

    if mismatch_patterns:
        print(f"\n{'='*60}")
        print("TYPE MISMATCH PATTERNS")
        print(f"{'='*60}")
        for pattern, count in sorted(mismatch_patterns.items(), key=lambda x: -x[1]):
            print(f"  {pattern}: {count}")

    # Save full results
    output_path = "benchmark_error_analysis.json"
    with open(output_path, "w") as f:
        json.dump({
            "model": model,
            "totals": totals,
            "entity_metrics": {"precision": e_prec, "recall": e_rec, "f1": e_f1},
            "relation_metrics": {"precision": r_prec, "recall": r_rec, "f1": r_f1},
            "per_test_results": all_results,
            "all_missed_entities": all_missed_ent,
            "all_hallucinated_entities": all_halluc_ent,
            "all_type_mismatches": all_type_mismatches,
            "all_missed_relations": all_missed_rel,
            "all_hallucinated_relations": all_halluc_rel,
        }, f, indent=2, default=str)
    print(f"\nFull results saved to {output_path}")


if __name__ == "__main__":
    main()

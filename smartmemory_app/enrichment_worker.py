"""Enrichment worker — separate process that drains the SQLite queue.

Polls enrichment_queue table, runs Tier 2 LLM extraction for each job,
writes results back to the graph. Runs as its own launchd-managed process.

Usage:
    python -m smartmemory_app.enrichment_worker          # run once (drain queue)
    python -m smartmemory_app.enrichment_worker --loop    # poll continuously
"""

import logging
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def process_one_job(job: dict) -> dict:
    """Run Tier 2 LLM extraction for a single job."""
    from smartmemory.background.extraction_worker import (
        _get_content_from_item,
        _run_llm_extraction,
        process_extract_job,
    )
    from smartmemory_app.storage import get_memory

    item_id = job["item_id"]
    mem = get_memory()
    item = mem.get(item_id)

    if item is None:
        return {"status": "item_not_found", "new_entities": 0, "new_relations": 0}

    content = _get_content_from_item(item)
    if not content.strip():
        return {"status": "no_text", "new_entities": 0, "new_relations": 0}

    llm_result = _run_llm_extraction(content)
    if llm_result["status"] != "ok":
        return {"status": "llm_failed", "new_entities": 0, "new_relations": 0}

    with mem._di_context():
        result = process_extract_job(
            mem,
            job,
            redis_client=None,
            item_override=mem.get(item_id),  # re-fetch for freshness
            extraction_override=llm_result.get("extraction"),
        )

    return result


def drain_queue() -> int:
    """Process all pending jobs. Returns count of jobs processed."""
    from smartmemory_app.enrichment_queue import dequeue, mark_done, mark_failed

    processed = 0
    while True:
        jobs = dequeue(batch_size=1)
        if not jobs:
            break

        for job in jobs:
            item_id = job["item_id"]
            queue_id = job["queue_id"]
            try:
                result = process_one_job(job)
                mark_done(queue_id)
                processed += 1
                status = result.get("status", "?")
                new_e = result.get("new_entities", 0)
                new_r = result.get("new_relations", 0)
                if status == "ok" and (new_e or new_r):
                    logger.info("Enriched %s: +%d entities, +%d relations", item_id, new_e, new_r)
                else:
                    logger.info("Enrichment %s: status=%s", item_id, status)
            except Exception as e:
                mark_failed(queue_id, str(e))
                logger.warning("Enrichment failed for %s: %s", item_id, e)

    return processed


def run_loop(poll_interval: float = 2.0) -> None:
    """Poll the queue continuously."""
    logger.info("Enrichment worker started (poll every %.1fs)", poll_interval)
    while True:
        try:
            n = drain_queue()
            if n > 0:
                logger.info("Processed %d jobs", n)
        except Exception:
            logger.warning("Drain cycle failed", exc_info=True)
        time.sleep(poll_interval)


def main():
    loop = "--loop" in sys.argv
    if loop:
        run_loop()
    else:
        n = drain_queue()
        logger.info("Drained %d jobs", n)


if __name__ == "__main__":
    main()

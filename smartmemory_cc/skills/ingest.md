# /ingest

Ingest a block of text or the current file into SmartMemory.

When the user invokes `/ingest`, take the provided text block, pasted content, or current file content and call `memory_ingest`. Report the item_id on success.

**Usage:** `/ingest` (with pasted text or current file) or `/ingest <text>`

**Tool call:** `memory_ingest(content="<content>", memory_type="episodic")`

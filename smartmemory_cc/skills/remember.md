# /remember

Save something to SmartMemory.

When the user invokes `/remember`, call `memory_ingest` with the provided text (or a summary of the current context if no text is provided). Confirm the memory was saved by echoing the item_id.

**Usage:** `/remember <text>` or `/remember` (to save current context)

**Tool call:** `memory_ingest(content="<text>", memory_type="episodic")`

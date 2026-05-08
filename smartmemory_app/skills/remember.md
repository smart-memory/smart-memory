# /remember

Save something to SmartMemory.

When the user invokes `/remember`, call `memory_ingest` with the provided text (or a summary of the current context if no text is provided). Confirm the memory was saved by echoing the item_id.

**Usage:** `/remember <text>` or `/remember` (to save current context)

**Tool call:** `memory_ingest(content="<text>", memory_type="episodic")`

## Expertise capture (CORE-EXPERTISE-1)

When the captured content is a *decision*, *constraint*, or *hard-won lesson*, prefer the typed wrapper over `memory_ingest`/`memory_add` — it stores structured fields the recall layer can surface as expertise.

| Shape of statement | Tool call |
|---|---|
| "We chose X over Y because Z" → a *decision* | `add_decision(content="...", rejected_alternatives=[...], rationale="...", constraints=[...])` |
| "Must / cannot / always / never" → a *constraint* | `add_constraint(content="...", domain="...", source="...")` |
| "Turns out / we learned the hard way / discovered that" → a *learned lesson* | `add_learning(content="...", category="...", incident_link="...")` |
| "I think / my opinion / I prefer" → an *opinion* | `add_opinion(content="...", subject="...", confidence=0.8)` |

Free-text or unstructured prose still goes through `memory_ingest`. Only call the typed wrappers when the statement is clearly one of the four shapes above.

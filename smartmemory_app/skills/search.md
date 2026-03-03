# /search

Search SmartMemory for relevant memories.

When the user invokes `/search <query>`, call `memory_search` with the provided query and display the results in a readable list format showing memory type, content preview, and relevance score.

**Usage:** `/search <query>`

**Tool call:** `memory_search(query="<query>", top_k=5)`

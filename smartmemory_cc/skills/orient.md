# /orient

Recall recent and relevant memories for the current working directory.

When the user invokes `/orient`, call `memory_recall` with the current working directory and display the top results as context. This helps orient yourself at the start of a session or when switching to a new area of the codebase.

**Usage:** `/orient`

**Tool call:** `memory_recall(cwd="<current_directory>", top_k=10)`

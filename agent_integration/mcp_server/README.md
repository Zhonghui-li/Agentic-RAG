# UCSC Slug Advisor — MCP server

Exposes the UCSC course tools over the **Model Context Protocol (MCP)** so any MCP
client (Claude Desktop, Cursor, …) can let *its own* LLM query the catalog. It
reuses the same functions the in-app agent uses — the value is the RAG over real
UCSC data; MCP is just the standard doorway to it.

Tools exposed:
- `search_catalog(query)` — hybrid BM25+FAISS search over the real CSE catalog
- `lookup_schedule(course_code, term)` — when/where a course meets, seats
- `get_academic_calendar(term)` — quarter dates, finals, enrollment

The vectorstore, data, and retrieval run **server-side**; only the query action is
exposed (not the index or the data dump).

## Run / test

```bash
cd agent_integration
pip install "mcp[cli]"          # on top of the project deps
export OPENAI_API_KEY=sk-...  EMB_MODEL=text-embedding-3-small

# stdio server (what Claude Desktop launches):
python mcp_server/server.py

# or inspect interactively:
mcp dev mcp_server/server.py
```

`RERANK` defaults to `0` here (no torch — lighter local launch); set `RERANK=1`
for CrossEncoder reranking.

## Add to Claude Desktop

Edit `claude_desktop_config.json` (Settings → Developer → Edit Config):

```json
{
  "mcpServers": {
    "ucsc-slug-advisor": {
      "command": "/ABS/PATH/agent_rl/venv/bin/python",
      "args": ["/ABS/PATH/agent_rl/agent_integration/mcp_server/server.py"],
      "env": {
        "OPENAI_API_KEY": "sk-...",
        "EMB_MODEL": "text-embedding-3-small"
      }
    }
  }
}
```

Restart Claude Desktop, then ask: *"What are the prerequisites for CSE 142, and is
it offered this fall?"* — Claude will call `search_catalog` and `lookup_schedule`
on this server. Paths resolve relative to the script, so the launch cwd doesn't
matter.

## Agent vs MCP server (two ways to consume the same tools)

- **In-app agent** (`slug_service`) — *our* LLM orchestrates the tools, packaged as
  a product (the web demo).
- **MCP server** (this) — exposes the same tools so *other* clients' LLMs can use
  them. Same tool functions, two consumption modes.

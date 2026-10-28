# BitGuide Multi-Agent Platform

A configuration-driven multi-agent platform written from scratch in Python 3.11 `asyncio`,
**without LangChain / LangGraph**. It includes a hand-written ReAct loop, tool calling, RAG,
MCP (JSON-RPC over stdio), a reference-counted agent registry, and a graph workflow engine with
parallel super-steps, checkpoints, and human-in-the-loop pause / resume.

```mermaid
flowchart LR
    CLI[CLI] --> P[Platform]
    API[Gin gateway<br/>Go] -- LPUSH --> Q[(Redis<br/>queue + status)] -- BRPOP --> WK[worker] --> P
    YAML[(platform.yaml)] --> F
    ETCD[(etcd)] -.-> F
    P --> F[AgentFactory] --> R[AgentRegistry<br/>ref-count + safe replace]
    P --> W[WorkflowService]
    W --> E[WorkflowRunner<br/>BSP super-steps] --> CP[(Checkpoint<br/>file / Redis)]
    E -- AgentNode acquire --> R
    R --> A[Agent · ReAct loop]
    A --> LLM[LLM<br/>Ollama / OpenAI / offline]
    A --> T[Tools] --> MCP[MCP server<br/>stdio subprocess]
    A --> RAG[RAG<br/>Milvus / in-memory]
    A --> S[Skills<br/>SKILL.md]
```

## Quick start

```bash
uv sync
uv run bitguide --offline agents                         # list agents
uv run bitguide --offline chat -m "周末带 4 岁孩子玩什么？顺便算 12*(3+4)"
uv run bitguide --offline workflow list                  # 8 demo workflows
uv run bitguide --offline workflow run expert_review -i "孩子想学编程" --thread demo
uv run bitguide --offline workflow resume demo --approve # resumes from checkpoint, even in a new process
```

With a real model: `ollama pull qwen2.5`, then drop `--offline`. To use a cloud model, add an
OpenAI-compatible entry under `models:` in `config/platform.yaml` (API key via `apiKeyEnv`).

## HTTP API (Gin gateway)

The Go gateway in `gateway/` only accepts tasks, queues them, and reports status; the Python
`bitguide worker` executes them. The two sides talk only through Redis: a List is the queue and
a Hash per task holds `queued → running → completed / interrupted / failed`. HTTP requests return
`202` immediately instead of waiting for a workflow that may run for a minute or stop for
approval. The task id doubles as the workflow thread id, so `resume` reaches the same checkpoint.

```bash
redis-server                                      # localhost:6379
uv run bitguide --offline worker                  # --concurrency 4; start more processes to scale
cd gateway && go run .                            # :8080 (env ADDR, REDIS_ADDR)

curl -X POST localhost:8080/tasks -d '{"workflow":"human_approval","input":"周五团建"}'
# → 202 {"id":"e2a6f8d6910d","status":"queued"}
curl localhost:8080/tasks/e2a6f8d6910d            # status, output, interrupt (JSON string), error
curl -X POST localhost:8080/tasks/e2a6f8d6910d/resume -d '{"approved":true,"comment":"ok"}'
```

`resume` returns `409` unless the task is `interrupted` (body required) or `failed` (retry, body
optional). Unknown workflow names are accepted and show up as `failed` once the worker picks them up.

## Highlights

| Area | What it does |
|---|---|
| ReAct agent | reason → tool call → observe loop, max-iteration guard, memory written only on success |
| Tools | exception-isolated `ToolManager`, AST-based safe calculator (no `eval`) |
| MCP | hand-written JSON-RPC 2.0 stdio client + server; remote tools adapted to the local `Tool` protocol |
| RAG | boundary-aware splitter with overlap, pluggable embedder / vector store (Milvus or in-memory), fail-open |
| Registry | `async with registry.acquire(id)`; `replace()` swaps instances atomically and retires the old one only after in-flight requests finish |
| Factory | turns YAML / etcd declarations into live agents, cleans up partial resources on failure |
| Workflow engine | `Graph` + `Channel` (LastValue / Append / Ephemeral), BFS validation, BSP super-steps with `asyncio.gather` + `Semaphore` |
| Checkpoint & HITL | snapshot after every step; `Interrupt` pauses, `resume()` continues; failed nodes retried without re-running finished siblings |
| Demos | pipeline, router, parallel experts, review loop, supervisor, human approval, tool pipeline, expert review |

## Project layout

```
config/platform.yaml     models, agents, RAG, MCP, workflow settings
gateway/main.go          Gin HTTP API: submit / query / resume tasks via a Redis queue
src/bitguide/
  models.py tools.py memory.py contracts.py agent.py   # single agent core
  llm.py config.py observability.py skills.py rag.py mcp.py
  infrastructure.py registry.py factory.py application.py cli.py
  workflow_core.py workflow_runtime.py workflow_checkpoint.py
  workflow_nodes.py workflow_registry.py workflow_demos.py workflow.py
  worker.py                                            # consumes tasks queued by the gateway
mcp_servers/             a minimal MCP server (no SDK)
knowledge/  skills/      sample knowledge base and SKILL.md files
tests/                   180 tests, 93% coverage, no network needed
```

## Development

```bash
uv run pytest -q          # tests
uv run ruff check .       # lint
uv run pyright            # type check
uv sync --extra milvus    # optional: Milvus Lite vector store
uv sync --extra redis     # optional: Redis checkpoint store
```

License: for learning purposes. Rebuilt independently with reference to an open-source teaching
project.

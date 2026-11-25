# agent-mem

A local, single-binary memory engine shared by coding agents: **Claude Code, OpenCode, OpenAI Codex CLI and Google Antigravity** all talk to one warm daemon, so a fact one agent learns is available to the others.

- Hybrid retrieval: HNSW vector search (usearch, cosine) + SQLite FTS5 BM25, fused with Reciprocal Rank Fusion, then weighted by exponential time decay and usage.
- Fully local after the first run: SQLite (WAL) for storage, quantized `BAAI/bge-small-en-v1.5` via ONNX on CPU for embeddings.
- MCP over stdio (`agent-mem mcp`) and MCP Streamable HTTP (`POST /mcp`), plus a small REST API.

```
 Claude Code ─┐                                   ┌─ SQLite (WAL + FTS5)   source of truth
 OpenCode ────┼─ agent-mem mcp (stdio proxy) ─►  daemon 127.0.0.1:4321 ─┤
 Codex CLI ───┤   auto-starts the daemon          └─ usearch HNSW index    rebuildable cache
 Antigravity ─┘                                      warm ONNX model
```

## Install

**One line (macOS / Linux):**

```sh
curl -fsSL https://raw.githubusercontent.com/hemanth090/agent-mem/main/install.sh | sh
```

That downloads the prebuilt binary for your machine, verifies its SHA-256, copies it to `~/.agent-mem/bin/agent-mem`, registers it with every agent it finds (Claude Code, OpenCode, Codex CLI, Antigravity), and pre-downloads the ~35 MB embedding model. Restart your agent(s) and the `remember` / `recall` / `forget` / `inspect_stats` tools are available in **every project**. Pass options after `sh -s --`, e.g. `... | sh -s -- --clients claude,opencode` or `--dry-run`.

**From source** (needs a Rust toolchain):

```sh
cargo build --release
target/release/agent-mem install        # copies itself to ~/.agent-mem/bin and registers that copy
```

Why a stable copy: client configs get an absolute path. Registering a build or download directory would break when it is cleaned or moved; `~/.agent-mem/bin/agent-mem` survives. Re-running `install` (or the one-liner) upgrades the binary and restarts the daemon on next use.

Useful flags: `--dry-run` (show changes only), `--clients a,b` (also configures a client that was not auto-detected), `--uninstall`, `--no-model` (skip the model download), `--no-copy` (register the current binary in place, for development). Quit Claude Code before installing so it does not overwrite `~/.claude.json` while the installer edits it.

Memory then works across sessions and agents on its own: the first agent to start launches the shared daemon in the background, later sessions reuse it, and everything is persisted in `~/.agent-mem`. After a reboot the next session simply starts it again.

The installer edits these files atomically, with a `*.bak.<unixtime>` backup of any existing file:

| Client | File | Entry |
|---|---|---|
| Claude Code | `~/.claude.json` | `mcpServers.agent-mem` (stdio) |
| OpenCode | `~/.config/opencode/opencode.json` | `mcp.agent-mem` (`type: local`) |
| Codex CLI | `~/.codex/config.toml` | `[mcp_servers.agent-mem]` (comments preserved) |
| Antigravity | `~/.gemini/config/mcp_config.json` or `~/.gemini/antigravity/mcp_config.json` | `mcpServers.agent-mem` |

It is idempotent, leaves unparseable files untouched, skips clients it cannot detect, and `--uninstall` removes only its own entry. `--clients claude,codex` limits the set. Restart the agent afterwards.

> The Antigravity config location could not be verified; the installer updates whichever of the two paths exists and creates the first if only `~/.gemini` exists. Claude Code rewrites `~/.claude.json` itself, so quit it before running `install`.

Add the usage instructions to a project:

```sh
agent-mem install --instructions-only --write-instructions ./AGENTS.md   # or ./CLAUDE.md
agent-mem template            # just print the snippet
```

The block is delimited by `<!-- agent-mem:begin/end -->`, so re-running updates it in place. Nothing is written to any instruction file unless you pass `--write-instructions`.

## MCP tools

| Tool | Purpose |
|---|---|
| `remember(content, memory_type?, project_id?, tags?)` | Store a decision, bug root cause, convention or preference. Near-duplicates (cosine > 0.95, same project and type) are merged: timestamp, weight and access count are bumped, tags merged. |
| `recall(query, project_id?, top_k?, memory_type?, tags?, include_global?)` | Vector ANN and BM25 run concurrently, are fused with RRF and weighted by decay. |
| `forget(memory_id)` | Delete an obsolete memory. |
| `inspect_stats()` | Totals, index size, average and p95 recall latency. |

`project_id` defaults to the git-root name of the agent's working directory (filled in by the stdio proxy). `recall` searches that project **plus the shared `global` namespace**; pass `include_global=false` for the project only. Use `project_id: "global"` for cross-project preferences. Resources: `agent-mem://stats`, `agent-mem://instructions`, `agent-mem://project/{id}/recent`.

### Scoring

```
FinalScore = RRF × weight × exp(-ln2 · Δt_days / half_life) × (1 + ln(max(1, access_count)))
```

`RRF = Σ 1/(60 + rank)` over the vector and BM25 lists. Half-lives: episodic 14 d, semantic 180 d, procedural 365 d (`Δt` is time since last access). Note `max(1, …)`: a plain `ln(access_count)` is 0 for a first access and would zero every new memory.

Vector candidates below `min_similarity` (default 0.55) are discarded so an unrelated query returns nothing instead of the nearest neighbour. That default was calibrated on a small sample with the quantized model (relevant pairs scored 0.62–0.78, unrelated 0.29–0.57); tune it in `config.toml` if recall feels too strict or too loose. BM25 matches are never filtered by it.

## Commands

```
agent-mem daemon [--foreground]   start the shared daemon (detached by default)
agent-mem mcp                     stdio MCP server (what clients launch); starts the daemon if needed
agent-mem install [...]           configure clients
agent-mem status | stop           inspect / stop the daemon
agent-mem fetch-model             pre-download the model
agent-mem reindex                 re-embed everything after a model change (daemon stopped)
agent-mem template                print the AGENTS.md / CLAUDE.md snippet
agent-mem bench-recall            latency benchmark on a throwaway database
```

Global flags: `--data-dir`, `--addr`, `--model-dir`. Env: `AGENT_MEM_DATA_DIR`, `AGENT_MEM_ADDR`, `AGENT_MEM_MODEL_DIR`, `AGENT_MEM_MIN_SIMILARITY`, `AGENT_MEM_DEDUP_THRESHOLD`, `AGENT_MEM_ALLOW_SECRETS`, `AGENT_MEM_LOG`. Optional `<data-dir>/config.toml` (see `mem-core/src/config.rs` for every key).

Data lives in `~/.agent-mem`: `memory.db`, `index.usearch`, `token`, `daemon.log` (rotated at 5 MB), `models/`.

### Offline use

The first run downloads the quantized model from Hugging Face into `<data-dir>/models`. To go offline, run `fetch-model` once, or copy that directory to the other machine and point `--model-dir` / `AGENT_MEM_MODEL_DIR` at it. After that nothing leaves your machine.

## Security

- The daemon refuses to bind anything but loopback and rejects requests whose `Host` or `Origin` is not loopback (DNS-rebinding defence).
- Every route except `/health` needs the bearer token in `<data-dir>/token` (mode 0600, created on first start). Client configs never contain it; the stdio proxy reads it from disk.
- `remember` rejects content that looks like credentials (AWS/GitHub/Slack/Google keys, private keys, JWTs, `password=…`, URLs with credentials, long high-entropy tokens). This is a best-effort filter, not a guarantee.
- Request bodies are capped at 64 KiB, requests time out after 10 s.

## Reliability notes

- SQLite is the source of truth and stores each embedding. If `index.usearch` is missing, corrupt, or disagrees with the database it is rebuilt automatically on start. The index is checkpointed every 64 writes / 30 s and on shutdown (temp file + rename).
- Only the daemon opens the database, so there is no multi-process write contention. Concurrent proxies start exactly one daemon (file lock + loopback port bind as the final arbiter).
- Access-count updates from `recall` are batched (every 5 s) rather than written on each read.
- Changing the embedding model is refused with a clear error until you run `agent-mem reindex`.

## Performance

Measured on an Apple-silicon laptop, release build, quantized bge-small, 5 000 synthetic memories (`agent-mem bench-recall --records 5000`):

| | p50 | p99 |
|---|---|---|
| `recall`, cold query (includes ONNX embedding) | 7.5 ms | 8.1 ms |
| `recall`, query embedding cached | 2.6 ms | 3.5 ms |
| `remember` (includes embedding and dedup) | 6.4 ms | 9.9 ms |

Retrieval (ANN + BM25 + fusion) excluding embedding is the cached row. Numbers vary by machine; run the benchmark yourself. `cargo bench` runs criterion micro-benchmarks (fusion math; store retrieval with a fake embedder). BM25 cost grows with how many documents match a query's terms: queries made only of words present in every document are slower (about 10 ms at 20 000 near-identical synthetic records), which realistic, varied memories do not hit. Stopwords are removed from the BM25 query for this reason.

## Development

```sh
cargo test --workspace                          # unit + end-to-end (fake embedder, no download)
cargo test -p mem-embed --features model-tests  # loads the real model
cargo clippy --workspace --all-targets -- -D warnings
```

`AGENT_MEM_FAKE_EMBEDDER=1` swaps in a deterministic hashing embedder (tests and smoke runs only; its vectors have no real semantics).

### Publishing a release (maintainers)

Push a tag (`git tag v0.1.0 && git push origin v0.1.0`). `.github/workflows/release.yml` builds archives for Apple-silicon macOS, Linux x86_64 and Linux arm64 (Intel macOS is best-effort), writes `.sha256` files and attaches them to the release; `install.sh` downloads from `releases/latest`. Windows is not supported yet.

Release builds disable LTO on purpose: with LTO the macOS link fails on a symbol the prebuilt ONNX Runtime needs. See the comment in the workspace `Cargo.toml`.

Layout: `mem-core` (schemas, RRF/decay, config) · `mem-embed` (embedder trait, fastembed engine, LRU query cache) · `mem-store` (SQLite + FTS5 + HNSW, dedup) · `mem-cli` (the `agent-mem` binary). See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the file-level map, request paths and invariants.

Not in v1: Unix domain sockets, SSE streaming, authentication beyond the local token, sync between machines.

# Architecture

## Workspace layout

```
agent-mem/
├── Cargo.toml                workspace, shared dependency versions, release profile
├── install.sh                one-line installer (downloads a release, runs `agent-mem install`)
├── .github/workflows/        ci.yml (fmt, clippy, tests) · release.yml (prebuilt binaries)
├── docs/ARCHITECTURE.md      this file
├── AGENTS.md · CLAUDE.md     instructions for agents working on this repo (include the memory block)
└── crates/
    ├── mem-core/             no I/O: schemas, validation, RRF + decay math, config
    │   └── src/  schema.rs · rank.rs · config.rs · error.rs
    ├── mem-embed/            embeddings behind a trait
    │   └── src/  engine.rs (fastembed, bge-small-en-v1.5 quantized) · cache.rs (LRU) · fake.rs (tests)
    ├── mem-store/            persistence and retrieval
    │   └── src/
    │       ├── sqlite.rs     schema, migrations, WAL + busy_timeout, row mapping
    │       ├── hnsw.rs       usearch wrapper (filtered search, atomic save)
    │       ├── secrets.rs    credential detection for `remember`
    │       └── store/        the `Store` type
    │           ├── mod.rs    options, open/rebuild/reindex, checkpoint, access flush, maintenance
    │           ├── write.rs  remember (cosine dedup) and forget
    │           ├── read.rs   recall (ANN ∥ BM25 → RRF → decay), get, recent, stats
    │           ├── fts.rs    free text → safe FTS5 query (stopwords removed)
    │           └── tests.rs
    └── mem-cli/              the `agent-mem` binary
        ├── templates/AGENTS.md   the instruction block embedded in the binary
        ├── tests/e2e.rs          drives the real binary (fake embedder)
        └── src/
            ├── main.rs · cli.rs          entrypoint and clap definitions
            ├── server/                   the daemon
            │   ├── daemon.rs             bind, token, model load, graceful shutdown, stop
            │   └── rest.rs               axum router, loopback/token guard, REST handlers
            ├── mcp/                      Model Context Protocol
            │   ├── proto.rs              JSON-RPC handling, tools, resources (shared)
            │   ├── http.rs               POST /mcp (Streamable HTTP)
            │   └── stdio.rs              `agent-mem mcp`: stdio → daemon proxy
            ├── lifecycle/                client-side daemon management
            │   ├── client.rs             health probes, token, project-name detection
            │   └── spawn.rs              detached start under a file lock
            ├── commands/                 one-shot commands
            │   ├── install/              mod.rs (per-client config edits) · command.rs · tests.rs
            │   ├── model.rs              fetch-model, reindex
            │   ├── status.rs · bench.rs
            ├── embedder.rs               builds the real or fake embedder
            └── template.rs               insert/replace the instruction block in a file
```

Dependency direction: `mem-core` ← `mem-embed` ← `mem-store` ← `mem-cli`. `mem-core` never does I/O except reading its config file.

## Runtime model

```
agent ──stdio──► agent-mem mcp ──HTTP + bearer token──► daemon (127.0.0.1:4321)
                  (stateless proxy)                       ├─ SQLite  (source of truth, incl. embeddings)
                  starts the daemon if needed             ├─ usearch HNSW (cache, rebuilt if inconsistent)
                  fills in project_id from the cwd        └─ warm ONNX model + query LRU
```

Only the daemon opens the database and index, so concurrent agents never contend on writes. Concurrent proxies start exactly one daemon: a file lock serializes spawning and the loopback port bind is the final arbiter.

## Request paths

- **remember:** validate → secret filter → embed (no locks held) → under the writer mutex: nearest neighbour in the same project and type; if cosine > 0.95 merge into it, else insert row + index add in one transaction.
- **recall:** embed query (LRU) → `tokio::join!` of filtered ANN and FTS5/BM25 (each in `spawn_blocking`) → RRF → load rows → `RRF × weight × exp(−λΔt) × (1 + ln max(1, uses))` → top-k. Access counts are queued and flushed in batches.
- **Scoping:** a project plus the shared `global` namespace unless `include_global=false`. ANN filtering happens inside the graph search via an in-memory key → (project, type, tags) map.

## Invariants worth preserving

- SQLite rows are authoritative. The index file may be missing, stale or corrupt; `Store::open` rebuilds it whenever its key set differs from the database.
- Lock order: never acquire the metadata map lock while holding the index lock for writing.
- Nothing is written to stdout by the proxy except protocol messages; logs go to stderr.
- No `unwrap`/`expect`/`todo!` outside tests (enforced by clippy at workspace level).
- The daemon binds loopback only, and every route except `/health` requires the token.
- Release builds keep LTO off (see `Cargo.toml`).

## Testing

`cargo test --workspace` uses the hashing fake embedder, so it needs no network. `mem-embed`'s `model-tests` feature loads the real model. `crates/mem-cli/tests/e2e.rs` starts real daemons on random ports in temp directories.

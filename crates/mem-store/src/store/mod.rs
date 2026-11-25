use std::collections::{HashMap, HashSet, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use mem_core::rank::{cmp_scored, final_score, rrf_fuse, HalfLives};
use mem_core::schema::{
    normalize_agent, normalize_project, normalize_tags, validate_content, GLOBAL_PROJECT, MAX_TAGS,
};
use mem_core::{
    Config, Error, MemoryRecord, MemoryType, RecallHit, RecallRequest, RecallResponse,
    RememberRequest, RememberResponse, Result, Stats,
};
use mem_embed::Embedder;
use parking_lot::{Mutex, RwLock};
use rusqlite::types::Value;
use rusqlite::{params, params_from_iter, Connection};

use crate::hnsw::VectorIndex;
use crate::secrets::find_secret;
use crate::sqlite::{
    blob_to_embedding, embedding_to_blob, get_meta, open_reader, open_writer, row_to_record,
    set_meta, sq, SELECT_COLS,
};

const READER_POOL: usize = 4;
const LATENCY_WINDOW: usize = 1000;
const MAX_TOP_K: usize = 50;
const MAX_WEIGHT: f64 = 5.0;
const DEDUP_WEIGHT_BUMP: f64 = 0.2;

#[derive(Debug, Clone)]
pub struct StoreOptions {
    pub db_path: PathBuf,
    pub index_path: PathBuf,
    pub dedup_threshold: f32,
    pub min_similarity: f32,
    pub rrf_k: f32,
    pub candidate_pool: usize,
    pub half_lives: HalfLives,
    pub max_content_bytes: usize,
    pub checkpoint_every_writes: u64,
    pub checkpoint_every_secs: u64,
    pub access_flush_secs: u64,
    pub allow_secrets: bool,
}

impl StoreOptions {
    /// Defaults rooted at `dir`.
    pub fn new(dir: &Path) -> Self {
        Self::from_config(&Config {
            data_dir: dir.to_path_buf(),
            ..Config::default()
        })
    }

    pub fn from_config(c: &Config) -> Self {
        Self {
            db_path: c.db_path(),
            index_path: c.index_path(),
            dedup_threshold: c.dedup_threshold,
            min_similarity: c.min_similarity,
            rrf_k: c.rrf_k,
            candidate_pool: c.candidate_pool,
            half_lives: c.half_lives(),
            max_content_bytes: c.max_content_bytes,
            checkpoint_every_writes: c.checkpoint_every_writes,
            checkpoint_every_secs: c.checkpoint_every_secs,
            access_flush_secs: c.access_flush_secs,
            allow_secrets: c.allow_secrets,
        }
    }
}

/// In-memory per-key metadata so ANN filtering happens *inside* the graph search.
#[derive(Debug, Clone)]
struct KeyMeta {
    project: Arc<str>,
    mtype: MemoryType,
    tags: Arc<Vec<String>>,
}

#[derive(Debug, Clone)]
struct Scope {
    projects: Vec<String>,
    mtype: Option<MemoryType>,
    tags: Vec<String>,
}

impl Scope {
    fn matches(&self, m: &KeyMeta) -> bool {
        self.projects.iter().any(|p| p.as_str() == &*m.project)
            && self.mtype.is_none_or(|t| t == m.mtype)
            && self.tags.iter().all(|t| m.tags.contains(t))
    }
}

pub struct Store {
    opts: StoreOptions,
    embedder: Arc<dyn Embedder>,
    writer: Mutex<Connection>,
    readers: Vec<Mutex<Connection>>,
    rr: AtomicUsize,
    index: VectorIndex,
    meta: RwLock<HashMap<u64, KeyMeta>>,
    pending_access: Mutex<Vec<(u64, i64)>>,
    writes_since_ckpt: AtomicU64,
    last_ckpt: Mutex<Instant>,
    latencies: Mutex<VecDeque<f64>>,
    started: Instant,
}

pub fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

fn join_err(e: tokio::task::JoinError) -> Error {
    Error::Internal(format!("task failed: {e}"))
}

mod fts;
mod read;
mod write;

#[cfg(test)]
mod tests;

pub use fts::fts_query;

impl Store {
    /// Opens (creating if needed) the database and index. Rebuilds the index from
    /// SQLite whenever it is missing, unreadable, or inconsistent with the DB.
    pub fn open(opts: StoreOptions, embedder: Arc<dyn Embedder>) -> Result<Arc<Store>> {
        if let Some(parent) = opts.db_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        if let Some(parent) = opts.index_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let writer = open_writer(&opts.db_path)?;
        let dim = embedder.dim();

        match (get_meta(&writer, "embed_model")?, get_meta(&writer, "dim")?) {
            (Some(m), Some(d)) => {
                if m != embedder.model_name() || d != dim.to_string() {
                    return Err(Error::Invalid(format!(
                        "database was built with model '{m}' (dim {d}) but the running model is '{}' \
                         (dim {dim}); run `agent-mem reindex` to re-embed all memories",
                        embedder.model_name()
                    )));
                }
            }
            _ => {
                set_meta(&writer, "embed_model", embedder.model_name())?;
                set_meta(&writer, "dim", &dim.to_string())?;
            }
        }

        // Load per-key metadata.
        let mut meta: HashMap<u64, KeyMeta> = HashMap::new();
        {
            let mut st = writer
                .prepare("SELECT key, project_id, memory_type, tags FROM memories")
                .map_err(sq)?;
            let rows = st
                .query_map([], |r| {
                    Ok((
                        r.get::<_, i64>(0)?,
                        r.get::<_, String>(1)?,
                        r.get::<_, String>(2)?,
                        r.get::<_, String>(3)?,
                    ))
                })
                .map_err(sq)?;
            for row in rows {
                let (k, p, t, tags) = row.map_err(sq)?;
                meta.insert(
                    k as u64,
                    KeyMeta {
                        project: Arc::from(p.as_str()),
                        mtype: MemoryType::parse(&t).unwrap_or_default(),
                        tags: Arc::new(serde_json::from_str(&tags).unwrap_or_default()),
                    },
                );
            }
        }

        let index = match VectorIndex::load(&opts.index_path, dim) {
            Ok(Some(ix)) if ix.len() == meta.len() && meta.keys().all(|k| ix.contains(*k)) => ix,
            other => {
                match &other {
                    Ok(Some(_)) => tracing::warn!("index inconsistent with database; rebuilding"),
                    Ok(None) => tracing::info!("no index file; building from database"),
                    Err(e) => tracing::warn!("index unreadable ({e}); rebuilding"),
                }
                drop(other);
                Self::rebuild_index(&writer, &opts.index_path, dim)?
            }
        };

        let mut readers = Vec::with_capacity(READER_POOL);
        for _ in 0..READER_POOL {
            readers.push(Mutex::new(open_reader(&opts.db_path)?));
        }

        Ok(Arc::new(Store {
            opts,
            embedder,
            writer: Mutex::new(writer),
            readers,
            rr: AtomicUsize::new(0),
            index,
            meta: RwLock::new(meta),
            pending_access: Mutex::new(Vec::new()),
            writes_since_ckpt: AtomicU64::new(0),
            last_ckpt: Mutex::new(Instant::now()),
            latencies: Mutex::new(VecDeque::with_capacity(LATENCY_WINDOW)),
            started: Instant::now(),
        }))
    }

    fn rebuild_index(conn: &Connection, path: &Path, dim: usize) -> Result<VectorIndex> {
        let ix = VectorIndex::create_empty(path, dim)?;
        let mut st = conn
            .prepare("SELECT key, embedding FROM memories")
            .map_err(sq)?;
        let rows = st
            .query_map([], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, Vec<u8>>(1)?)))
            .map_err(sq)?;
        for row in rows {
            let (k, blob) = row.map_err(sq)?;
            let v = blob_to_embedding(&blob);
            if v.len() != dim {
                return Err(Error::Index(format!(
                    "stored embedding for key {k} has dim {}, expected {dim}; run `agent-mem reindex`",
                    v.len()
                )));
            }
            ix.add(k as u64, &v)?;
        }
        ix.save()?;
        Ok(ix)
    }

    /// Re-embeds every memory with `embedder`, updates the stored model metadata
    /// and removes the stale index file (it is rebuilt on next open).
    pub fn reindex(opts: &StoreOptions, embedder: &dyn Embedder) -> Result<usize> {
        let conn = open_writer(&opts.db_path)?;
        let rows: Vec<(i64, String)> = {
            let mut st = conn
                .prepare("SELECT key, content FROM memories ORDER BY key")
                .map_err(sq)?;
            let it = st
                .query_map([], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?)))
                .map_err(sq)?;
            it.collect::<rusqlite::Result<_>>().map_err(sq)?
        };
        conn.execute_batch("BEGIN IMMEDIATE").map_err(sq)?;
        let res: Result<()> = (|| {
            for chunk in rows.chunks(64) {
                let texts: Vec<String> = chunk.iter().map(|(_, c)| c.clone()).collect();
                let embs = embedder.embed_passages(&texts)?;
                for ((key, _), e) in chunk.iter().zip(embs) {
                    conn.execute(
                        "UPDATE memories SET embedding=?1 WHERE key=?2",
                        params![embedding_to_blob(&e), key],
                    )
                    .map_err(sq)?;
                }
            }
            set_meta(&conn, "embed_model", embedder.model_name())?;
            set_meta(&conn, "dim", &embedder.dim().to_string())?;
            Ok(())
        })();
        match res {
            Ok(()) => conn.execute_batch("COMMIT").map_err(sq)?,
            Err(e) => {
                let _ = conn.execute_batch("ROLLBACK");
                return Err(e);
            }
        }
        match std::fs::remove_file(&opts.index_path) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return Err(e.into()),
        }
        Ok(rows.len())
    }

    // ------------------------------------------------------------------ helpers

    fn with_reader<T>(&self, f: impl FnOnce(&Connection) -> Result<T>) -> Result<T> {
        let n = self.readers.len();
        let start = self.rr.fetch_add(1, Ordering::Relaxed);
        for i in 0..n {
            if let Some(g) = self.readers[(start + i) % n].try_lock() {
                return f(&g);
            }
        }
        let g = self.readers[start % n].lock();
        f(&g)
    }

    fn note_write(&self) {
        let n = self.writes_since_ckpt.fetch_add(1, Ordering::Relaxed) + 1;
        if n >= self.opts.checkpoint_every_writes.max(1) {
            if let Err(e) = self.checkpoint() {
                tracing::warn!("index checkpoint failed: {e}");
            }
        }
    }

    /// Persist the HNSW index atomically.
    pub fn checkpoint(&self) -> Result<()> {
        self.index.save()?;
        self.writes_since_ckpt.store(0, Ordering::Relaxed);
        *self.last_ckpt.lock() = Instant::now();
        Ok(())
    }

    pub fn maybe_checkpoint(&self) -> Result<()> {
        let dirty = self.writes_since_ckpt.load(Ordering::Relaxed) > 0;
        let due =
            self.last_ckpt.lock().elapsed() >= Duration::from_secs(self.opts.checkpoint_every_secs);
        if dirty && due {
            self.checkpoint()?;
        }
        Ok(())
    }

    /// Applies queued access-count updates in a single transaction.
    pub fn flush_access(&self) -> Result<usize> {
        let drained = std::mem::take(&mut *self.pending_access.lock());
        if drained.is_empty() {
            return Ok(0);
        }
        let mut agg: HashMap<u64, (i64, i64)> = HashMap::new();
        for (k, ts) in drained {
            let e = agg.entry(k).or_insert((0, 0));
            e.0 += 1;
            e.1 = e.1.max(ts);
        }
        let mut conn = self.writer.lock();
        let tx = conn.transaction().map_err(sq)?;
        {
            let mut st = tx
                .prepare(
                    "UPDATE memories SET access_count = access_count + ?1, \
                     last_accessed = MAX(last_accessed, ?2) WHERE key = ?3",
                )
                .map_err(sq)?;
            for (k, (n, ts)) in &agg {
                st.execute(params![n, ts, *k as i64]).map_err(sq)?;
            }
        }
        tx.commit().map_err(sq)?;
        Ok(agg.len())
    }

    /// Background task: flushes access counts and checkpoints on a timer.
    pub fn spawn_maintenance(self: &Arc<Self>) -> tokio::task::JoinHandle<()> {
        let s = self.clone();
        tokio::spawn(async move {
            let mut tick =
                tokio::time::interval(Duration::from_secs(s.opts.access_flush_secs.max(1)));
            tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
            loop {
                tick.tick().await;
                let s2 = s.clone();
                let r = tokio::task::spawn_blocking(move || {
                    if let Err(e) = s2.flush_access() {
                        tracing::warn!("access flush failed: {e}");
                    }
                    if let Err(e) = s2.maybe_checkpoint() {
                        tracing::warn!("checkpoint failed: {e}");
                    }
                })
                .await;
                if let Err(e) = r {
                    tracing::warn!("maintenance task failed: {e}");
                }
            }
        })
    }

    pub fn shutdown_sync(&self) -> Result<()> {
        self.flush_access()?;
        self.checkpoint()
    }

    pub async fn shutdown(self: &Arc<Self>) -> Result<()> {
        let s = self.clone();
        tokio::task::spawn_blocking(move || s.shutdown_sync())
            .await
            .map_err(join_err)?
    }

    pub fn options(&self) -> &StoreOptions {
        &self.opts
    }

    #[cfg(test)]
    fn writer_for_test(&self) -> parking_lot::MutexGuard<'_, Connection> {
        self.writer.lock()
    }
}

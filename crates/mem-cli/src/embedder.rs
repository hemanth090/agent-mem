use std::sync::Arc;

use mem_core::{Config, Result};
use mem_embed::{CachedEmbedder, Embedder, FakeEmbedder, FastEmbedEngine};

/// `AGENT_MEM_FAKE_EMBEDDER=1` swaps in a deterministic hashing embedder. It exists for
/// tests and smoke runs only; its vectors are not semantically meaningful.
pub fn fake_requested() -> bool {
    matches!(
        std::env::var("AGENT_MEM_FAKE_EMBEDDER").as_deref(),
        Ok("1" | "true" | "yes")
    )
}

pub fn build_embedder(
    cfg: &Config,
    show_progress: bool,
    force_fake: bool,
) -> Result<Arc<dyn Embedder>> {
    if force_fake || fake_requested() {
        return Ok(Arc::new(CachedEmbedder::new(
            FakeEmbedder::new(),
            cfg.query_cache_size,
        )));
    }
    let engine = FastEmbedEngine::load(cfg.model_cache_dir(), show_progress)?;
    Ok(Arc::new(CachedEmbedder::new(engine, cfg.query_cache_size)))
}

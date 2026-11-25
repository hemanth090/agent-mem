use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use lru::LruCache;
use mem_core::Result;
use parking_lot::Mutex;

use crate::Embedder;

/// Wraps any `Embedder` with an LRU cache for query embeddings.
pub struct CachedEmbedder<E: Embedder> {
    inner: E,
    cache: Mutex<LruCache<[u8; 32], Arc<Vec<f32>>>>,
    hits: AtomicU64,
    misses: AtomicU64,
}

impl<E: Embedder> CachedEmbedder<E> {
    pub fn new(inner: E, capacity: usize) -> Self {
        let cap = NonZeroUsize::new(capacity.max(1)).unwrap_or(NonZeroUsize::MIN);
        Self {
            inner,
            cache: Mutex::new(LruCache::new(cap)),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
        }
    }

    pub fn inner(&self) -> &E {
        &self.inner
    }
}

impl<E: Embedder> Embedder for CachedEmbedder<E> {
    fn embed_passages(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        self.inner.embed_passages(texts)
    }

    fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        let key: [u8; 32] = *blake3::hash(text.as_bytes()).as_bytes();
        if let Some(v) = self.cache.lock().get(&key) {
            self.hits.fetch_add(1, Ordering::Relaxed);
            return Ok(v.as_ref().clone());
        }
        self.misses.fetch_add(1, Ordering::Relaxed);
        // Compute outside the cache lock so slow inference never blocks cache hits.
        let v = self.inner.embed_query(text)?;
        self.cache.lock().put(key, Arc::new(v.clone()));
        Ok(v)
    }

    fn dim(&self) -> usize {
        self.inner.dim()
    }

    fn model_name(&self) -> &str {
        self.inner.model_name()
    }

    fn cache_counters(&self) -> (u64, u64) {
        (
            self.hits.load(Ordering::Relaxed),
            self.misses.load(Ordering::Relaxed),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FakeEmbedder;

    #[test]
    fn counts_hits_and_misses() {
        let c = CachedEmbedder::new(FakeEmbedder::new(), 2);
        c.embed_query("a").unwrap();
        c.embed_query("a").unwrap();
        c.embed_query("b").unwrap();
        assert_eq!(c.cache_counters(), (1, 2));
    }

    #[test]
    fn evicts_lru() {
        let c = CachedEmbedder::new(FakeEmbedder::new(), 2);
        c.embed_query("a").unwrap();
        c.embed_query("b").unwrap();
        c.embed_query("c").unwrap(); // evicts a
        c.embed_query("a").unwrap(); // miss again
        assert_eq!(c.cache_counters(), (0, 4));
    }

    #[test]
    fn cached_value_is_identical() {
        let c = CachedEmbedder::new(FakeEmbedder::new(), 4);
        let a = c.embed_query("hello").unwrap();
        let b = c.embed_query("hello").unwrap();
        assert_eq!(a, b);
    }
}

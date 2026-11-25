#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]
//! Embedding engine abstraction, fastembed implementation and LRU query cache.

mod cache;
mod engine;
mod fake;

pub use cache::CachedEmbedder;
pub use engine::{FastEmbedEngine, MODEL_NAME};
pub use fake::FakeEmbedder;

use mem_core::Result;

/// Dimension of BAAI/bge-small-en-v1.5.
pub const EMBED_DIM: usize = 384;

/// BGE retrieval instruction, applied to queries only (never passages).
pub const QUERY_PREFIX: &str = "Represent this sentence for searching relevant passages: ";

pub trait Embedder: Send + Sync {
    fn embed_passages(&self, texts: &[String]) -> Result<Vec<Vec<f32>>>;
    fn embed_query(&self, text: &str) -> Result<Vec<f32>>;
    fn dim(&self) -> usize;
    fn model_name(&self) -> &str;
    /// (hits, misses) of the query cache, if any.
    fn cache_counters(&self) -> (u64, u64) {
        (0, 0)
    }
}

/// L2-normalizes in place; a zero vector is left untouched.
pub fn l2_normalize(v: &mut [f32]) {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n > 0.0 {
        for x in v.iter_mut() {
            *x /= n;
        }
    }
}

/// Cosine similarity of two (not necessarily normalized) vectors.
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na * nb)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_unit() {
        let mut v = vec![3.0, 4.0];
        l2_normalize(&mut v);
        assert!((v[0] - 0.6).abs() < 1e-6 && (v[1] - 0.8).abs() < 1e-6);
        let mut z = vec![0.0, 0.0];
        l2_normalize(&mut z);
        assert_eq!(z, vec![0.0, 0.0]);
    }

    #[test]
    fn cosine_basic() {
        assert!((cosine(&[1.0, 0.0], &[1.0, 0.0]) - 1.0).abs() < 1e-6);
        assert!(cosine(&[1.0, 0.0], &[0.0, 1.0]).abs() < 1e-6);
        assert_eq!(cosine(&[0.0], &[1.0]), 0.0);
    }
}

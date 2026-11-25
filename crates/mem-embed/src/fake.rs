use mem_core::Result;

use crate::{l2_normalize, Embedder, EMBED_DIM};

/// Deterministic bag-of-words hashing embedder. Texts that share words have
/// high cosine similarity, which is enough to test retrieval logic without
/// loading a model. Also usable with `AGENT_MEM_FAKE_EMBEDDER=1` for smoke tests.
#[derive(Debug, Clone, Default)]
pub struct FakeEmbedder;

impl FakeEmbedder {
    pub fn new() -> Self {
        FakeEmbedder
    }

    fn embed(text: &str) -> Vec<f32> {
        let mut v = vec![0.0f32; EMBED_DIM];
        for word in text
            .to_lowercase()
            .split(|c: char| !c.is_alphanumeric())
            .filter(|w| !w.is_empty())
        {
            let h = blake3::hash(word.as_bytes());
            let b = h.as_bytes();
            let idx = (u16::from_le_bytes([b[0], b[1]]) as usize) % EMBED_DIM;
            v[idx] += 1.0;
        }
        if v.iter().all(|x| *x == 0.0) {
            v[0] = 1.0;
        }
        l2_normalize(&mut v);
        v
    }
}

impl Embedder for FakeEmbedder {
    fn embed_passages(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        Ok(texts.iter().map(|t| Self::embed(t)).collect())
    }

    fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        Ok(Self::embed(text))
    }

    fn dim(&self) -> usize {
        EMBED_DIM
    }

    fn model_name(&self) -> &str {
        "fake-hash-384"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cosine;

    #[test]
    fn deterministic_and_similarity_ordering() {
        let e = FakeEmbedder::new();
        let a = e.embed_query("sqlite wal mode").unwrap();
        assert_eq!(a, e.embed_query("sqlite wal mode").unwrap());
        let near = e.embed_query("sqlite wal").unwrap();
        let far = e.embed_query("banana smoothie recipe").unwrap();
        assert!(cosine(&a, &near) > cosine(&a, &far));
        assert_eq!(a.len(), 384);
    }
}

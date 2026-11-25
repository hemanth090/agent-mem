use std::path::PathBuf;

use fastembed::{AutoDevicePolicy, EmbeddingModel, TextEmbedding, TextInitOptions};
use mem_core::{Error, Result};
use parking_lot::Mutex;

use crate::{l2_normalize, Embedder, EMBED_DIM, QUERY_PREFIX};

pub const MODEL_NAME: &str = "BAAI/bge-small-en-v1.5-q";

/// Thread-safe wrapper around one ONNX session. Inference is serialized by a
/// mutex; callers should batch where possible.
pub struct FastEmbedEngine {
    inner: Mutex<TextEmbedding>,
}

impl FastEmbedEngine {
    /// Loads the quantized model, downloading it into `cache_dir` on first use.
    pub fn load(cache_dir: PathBuf, show_progress: bool) -> Result<Self> {
        std::fs::create_dir_all(&cache_dir)?;
        let opts = TextInitOptions::new(EmbeddingModel::BGESmallENV15Q)
            .with_cache_dir(cache_dir.clone())
            .with_show_download_progress(show_progress)
            .with_auto_device(AutoDevicePolicy::PreferCPU);
        let model = TextEmbedding::try_new(opts).map_err(|e| {
            Error::Embed(format!(
                "could not load embedding model from {}: {e}. \
                 The first run downloads the model from Hugging Face; for offline use copy a \
                 previously downloaded cache to that directory (or set AGENT_MEM_MODEL_DIR).",
                cache_dir.display()
            ))
        })?;
        Ok(Self {
            inner: Mutex::new(model),
        })
    }

    fn run(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let mut out = self
            .inner
            .lock()
            .embed(texts, Some(32))
            .map_err(|e| Error::Embed(e.to_string()))?;
        for v in out.iter_mut() {
            if v.len() != EMBED_DIM {
                return Err(Error::Embed(format!(
                    "model returned dimension {}, expected {EMBED_DIM}",
                    v.len()
                )));
            }
            l2_normalize(v);
        }
        Ok(out)
    }
}

impl Embedder for FastEmbedEngine {
    fn embed_passages(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        self.run(texts.to_vec())
    }

    fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        let mut v = self.run(vec![format!("{QUERY_PREFIX}{text}")])?;
        v.pop()
            .ok_or_else(|| Error::Embed("empty embedding result".into()))
    }

    fn dim(&self) -> usize {
        EMBED_DIM
    }

    fn model_name(&self) -> &str {
        MODEL_NAME
    }
}

#[cfg(all(test, feature = "model-tests"))]
mod model_tests {
    use super::*;
    use crate::cosine;

    #[test]
    fn real_model_properties() {
        let dir = std::env::var_os("AGENT_MEM_MODEL_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| std::env::temp_dir().join("agent-mem-test-models"));
        let e = FastEmbedEngine::load(dir, false).unwrap();
        let v = e
            .embed_passages(&[
                "We use SQLite in WAL mode for persistence.".into(),
                "The database is SQLite with write-ahead logging.".into(),
                "Bananas are yellow fruit.".into(),
            ])
            .unwrap();
        assert_eq!(v[0].len(), 384);
        let norm: f32 = v[0].iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-3);
        assert!(cosine(&v[0], &v[1]) > cosine(&v[0], &v[2]));
        let q = e.embed_query("what database do we use").unwrap();
        assert!(cosine(&q, &v[0]) > cosine(&q, &v[2]));
    }
}

#[cfg(all(test, feature = "model-tests"))]
mod calibration {
    use super::*;
    use crate::cosine;

    /// Prints query/passage similarities used to choose the default relevance floor.
    #[test]
    fn print_similarity_distribution() {
        let dir = std::env::var_os("AGENT_MEM_MODEL_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| std::env::temp_dir().join("agent-mem-test-models"));
        let e = FastEmbedEngine::load(dir, false).unwrap();
        let passages: Vec<String> = [
            "The billing service retries failed webhooks three times with exponential backoff",
            "We use SQLite in WAL mode so multiple agents can share one database",
            "User prefers tabs over spaces everywhere",
            "Auth tokens expire after 15 minutes because of the API gateway timeout",
            "The CI pipeline caches cargo registry between runs to speed up builds",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        let pv = e.embed_passages(&passages).unwrap();
        let queries = [
            ("webhook retry policy", 0usize),
            ("which database do we use", 1),
            ("indentation style preference", 2),
            ("why do sessions log out quickly", 3),
            ("how to speed up CI", 4),
            ("what is the recipe for banana bread", 99),
            ("kubernetes ingress configuration", 99),
            ("how do I center a div", 99),
        ];
        for (q, expect) in queries {
            let qv = e.embed_query(q).unwrap();
            let sims: Vec<f32> = pv.iter().map(|p| cosine(&qv, p)).collect();
            let best = sims.iter().cloned().fold(f32::MIN, f32::max);
            let target = if expect < 99 { sims[expect] } else { f32::NAN };
            println!("CAL q={q:<40} best={best:.3} target={target:.3} all={sims:.3?}");
        }
    }
}

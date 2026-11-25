use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::rank::HalfLives;

pub const DEFAULT_ADDR: &str = "127.0.0.1:4321";

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct HalfLifeConfig {
    pub episodic: f64,
    pub semantic: f64,
    pub procedural: f64,
}

impl Default for HalfLifeConfig {
    fn default() -> Self {
        let h = HalfLives::default();
        Self {
            episodic: h.episodic,
            semantic: h.semantic,
            procedural: h.procedural,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    pub data_dir: PathBuf,
    pub addr: String,
    pub model_dir: Option<PathBuf>,
    pub dedup_threshold: f32,
    /// Vector hits with cosine similarity below this are not candidates (BM25 hits are unaffected).
    pub min_similarity: f32,
    pub rrf_k: f32,
    pub candidate_pool: usize,
    pub half_life_days: HalfLifeConfig,
    pub query_cache_size: usize,
    pub max_content_bytes: usize,
    pub checkpoint_every_writes: u64,
    pub checkpoint_every_secs: u64,
    pub access_flush_secs: u64,
    pub allow_secrets: bool,
}

pub fn default_data_dir() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".agent-mem")
}

impl Default for Config {
    fn default() -> Self {
        Self {
            data_dir: default_data_dir(),
            addr: DEFAULT_ADDR.to_string(),
            model_dir: None,
            dedup_threshold: 0.95,
            min_similarity: 0.55,
            rrf_k: 60.0,
            candidate_pool: 50,
            half_life_days: HalfLifeConfig::default(),
            query_cache_size: 1024,
            max_content_bytes: 8192,
            checkpoint_every_writes: 64,
            checkpoint_every_secs: 30,
            access_flush_secs: 5,
            allow_secrets: false,
        }
    }
}

impl Config {
    /// Load `<data_dir>/config.toml` (if present), then apply `AGENT_MEM_*` env overrides.
    /// `data_dir_override` (from the CLI) takes precedence over everything.
    pub fn load(data_dir_override: Option<&Path>) -> Result<Self> {
        let data_dir = data_dir_override
            .map(Path::to_path_buf)
            .or_else(|| std::env::var_os("AGENT_MEM_DATA_DIR").map(PathBuf::from))
            .unwrap_or_else(default_data_dir);

        let file = data_dir.join("config.toml");
        let mut cfg: Config = if file.exists() {
            let text = std::fs::read_to_string(&file)?;
            toml::from_str(&text).map_err(|e| Error::Invalid(format!("{}: {e}", file.display())))?
        } else {
            Config::default()
        };
        cfg.data_dir = data_dir;
        cfg.apply_env();
        cfg.validate()?;
        Ok(cfg)
    }

    fn apply_env(&mut self) {
        if let Ok(v) = std::env::var("AGENT_MEM_ADDR") {
            self.addr = v;
        }
        if let Some(v) = std::env::var_os("AGENT_MEM_MODEL_DIR") {
            self.model_dir = Some(PathBuf::from(v));
        }
        if let Ok(v) = std::env::var("AGENT_MEM_DEDUP_THRESHOLD") {
            if let Ok(f) = v.parse() {
                self.dedup_threshold = f;
            }
        }
        if let Ok(v) = std::env::var("AGENT_MEM_MIN_SIMILARITY") {
            if let Ok(f) = v.parse() {
                self.min_similarity = f;
            }
        }
        if let Ok(v) = std::env::var("AGENT_MEM_ALLOW_SECRETS") {
            self.allow_secrets = matches!(v.as_str(), "1" | "true" | "yes");
        }
    }

    pub fn validate(&self) -> Result<()> {
        if !(0.0..=1.0).contains(&self.dedup_threshold) {
            return Err(Error::Invalid("dedup_threshold must be in [0,1]".into()));
        }
        if !(0.0..=1.0).contains(&self.min_similarity) {
            return Err(Error::Invalid("min_similarity must be in [0,1]".into()));
        }
        if self.candidate_pool == 0 || self.query_cache_size == 0 || self.max_content_bytes == 0 {
            return Err(Error::Invalid(
                "candidate_pool, query_cache_size and max_content_bytes must be > 0".into(),
            ));
        }
        let hl = &self.half_life_days;
        if hl.episodic <= 0.0 || hl.semantic <= 0.0 || hl.procedural <= 0.0 {
            return Err(Error::Invalid("half_life_days values must be > 0".into()));
        }
        self.socket_addr()?;
        Ok(())
    }

    pub fn socket_addr(&self) -> Result<std::net::SocketAddr> {
        let a: std::net::SocketAddr = self
            .addr
            .parse()
            .map_err(|e| Error::Invalid(format!("bad addr '{}': {e}", self.addr)))?;
        if !a.ip().is_loopback() {
            return Err(Error::Invalid(format!(
                "refusing non-loopback address {a}; agent-mem only binds to loopback"
            )));
        }
        Ok(a)
    }

    pub fn half_lives(&self) -> HalfLives {
        HalfLives {
            episodic: self.half_life_days.episodic,
            semantic: self.half_life_days.semantic,
            procedural: self.half_life_days.procedural,
        }
    }

    pub fn model_cache_dir(&self) -> PathBuf {
        self.model_dir
            .clone()
            .unwrap_or_else(|| self.data_dir.join("models"))
    }
    pub fn db_path(&self) -> PathBuf {
        self.data_dir.join("memory.db")
    }
    pub fn index_path(&self) -> PathBuf {
        self.data_dir.join("index.usearch")
    }
    pub fn token_path(&self) -> PathBuf {
        self.data_dir.join("token")
    }
    pub fn pid_path(&self) -> PathBuf {
        self.data_dir.join("daemon.pid")
    }
    pub fn lock_path(&self) -> PathBuf {
        self.data_dir.join("spawn.lock")
    }
    pub fn log_path(&self) -> PathBuf {
        self.data_dir.join("daemon.log")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_validate() {
        Config::default().validate().unwrap();
    }

    #[test]
    fn rejects_non_loopback() {
        let c = Config {
            addr: "0.0.0.0:4321".into(),
            ..Config::default()
        };
        assert!(c.validate().is_err());
        let c = Config {
            addr: "[::1]:4321".into(),
            ..Config::default()
        };
        assert!(c.validate().is_ok());
    }

    #[test]
    fn rejects_bad_threshold() {
        let c = Config {
            dedup_threshold: 1.5,
            ..Config::default()
        };
        assert!(c.validate().is_err());
    }

    #[test]
    fn partial_toml_uses_defaults() {
        let c: Config =
            toml::from_str("dedup_threshold = 0.9\n[half_life_days]\nepisodic = 7.0").unwrap();
        assert_eq!(c.dedup_threshold, 0.9);
        assert_eq!(c.half_life_days.episodic, 7.0);
        assert_eq!(c.half_life_days.semantic, 180.0);
        assert_eq!(c.addr, DEFAULT_ADDR);
    }
}

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

pub const GLOBAL_PROJECT: &str = "global";
pub const MAX_TAGS: usize = 16;
pub const MAX_TAG_LEN: usize = 48;
pub const MAX_PROJECT_LEN: usize = 128;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum MemoryType {
    Episodic,
    #[default]
    Semantic,
    Procedural,
}

impl MemoryType {
    pub fn as_str(&self) -> &'static str {
        match self {
            MemoryType::Episodic => "episodic",
            MemoryType::Semantic => "semantic",
            MemoryType::Procedural => "procedural",
        }
    }

    pub fn parse(s: &str) -> Result<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "episodic" => Ok(MemoryType::Episodic),
            "semantic" => Ok(MemoryType::Semantic),
            "procedural" => Ok(MemoryType::Procedural),
            other => Err(Error::Invalid(format!(
                "unknown memory_type '{other}' (expected episodic|semantic|procedural)"
            ))),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MemoryRecord {
    pub id: String,
    pub content: String,
    pub memory_type: MemoryType,
    pub project_id: String,
    pub agent_source: String,
    pub tags: Vec<String>,
    /// Unix milliseconds.
    pub created_at: i64,
    /// Unix milliseconds.
    pub last_accessed: i64,
    pub access_count: u32,
    pub weight: f32,
}

/// Normalizes and validates a project id. `None`/empty maps to `global`.
pub fn normalize_project(p: Option<&str>) -> Result<String> {
    let p = match p.map(str::trim) {
        None | Some("") => return Ok(GLOBAL_PROJECT.to_string()),
        Some(p) => p,
    };
    if p.len() > MAX_PROJECT_LEN
        || !p
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '/' | '-'))
    {
        return Err(Error::Invalid(format!(
            "project_id '{p}' must match [A-Za-z0-9._/-]{{1,{MAX_PROJECT_LEN}}}"
        )));
    }
    Ok(p.to_string())
}

/// Lowercases, trims, de-duplicates and validates tags.
pub fn normalize_tags(tags: &[String]) -> Result<Vec<String>> {
    let mut out: Vec<String> = Vec::new();
    for t in tags {
        let t = t.trim().to_lowercase();
        if t.is_empty() {
            continue;
        }
        if t.chars().count() > MAX_TAG_LEN {
            return Err(Error::Invalid(format!(
                "tag longer than {MAX_TAG_LEN} characters"
            )));
        }
        if !out.contains(&t) {
            out.push(t);
        }
    }
    if out.len() > MAX_TAGS {
        return Err(Error::Invalid(format!("more than {MAX_TAGS} tags")));
    }
    Ok(out)
}

pub fn normalize_agent(a: Option<&str>) -> String {
    match a.map(|s| s.trim().to_ascii_lowercase()) {
        Some(s) if !s.is_empty() && s.len() <= 64 => s,
        _ => "unknown".to_string(),
    }
}

pub fn validate_content(content: &str, max_bytes: usize) -> Result<String> {
    let c = content.trim();
    if c.is_empty() {
        return Err(Error::Invalid("content is empty".into()));
    }
    if c.len() > max_bytes {
        return Err(Error::Invalid(format!(
            "content is {} bytes, limit is {max_bytes}",
            c.len()
        )));
    }
    Ok(c.to_string())
}

// ---------- request / response types shared by store, daemon and proxy ----------

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RememberRequest {
    pub content: String,
    #[serde(default)]
    pub memory_type: Option<String>,
    #[serde(default)]
    pub project_id: Option<String>,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub agent_source: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RememberResponse {
    pub id: String,
    pub deduplicated: bool,
    pub similarity: Option<f32>,
}

fn default_top_k() -> usize {
    5
}
fn default_true() -> bool {
    true
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecallRequest {
    pub query: String,
    #[serde(default)]
    pub project_id: Option<String>,
    #[serde(default = "default_top_k")]
    pub top_k: usize,
    #[serde(default)]
    pub memory_type: Option<String>,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default = "default_true")]
    pub include_global: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecallHit {
    pub id: String,
    pub content: String,
    pub memory_type: MemoryType,
    pub project_id: String,
    pub tags: Vec<String>,
    pub score: f32,
    pub created_at: i64,
    pub last_accessed: i64,
    pub access_count: u32,
    pub sources: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecallResponse {
    pub hits: Vec<RecallHit>,
    pub took_ms: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForgetRequest {
    pub memory_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Stats {
    pub total: u64,
    pub by_type: std::collections::BTreeMap<String, u64>,
    pub by_project: Vec<(String, u64)>,
    pub index_vectors: u64,
    pub index_bytes: u64,
    pub db_bytes: u64,
    pub avg_recall_ms: f64,
    pub p95_recall_ms: f64,
    pub recall_samples: u64,
    pub query_cache_hit_rate: f64,
    pub model: String,
    pub uptime_s: u64,
    pub version: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn project_normalization() {
        assert_eq!(normalize_project(None).unwrap(), "global");
        assert_eq!(normalize_project(Some("  ")).unwrap(), "global");
        assert_eq!(normalize_project(Some(" my-repo ")).unwrap(), "my-repo");
        assert!(normalize_project(Some("bad name")).is_err());
        assert!(normalize_project(Some(&"a".repeat(129))).is_err());
    }

    #[test]
    fn tags_normalized() {
        let t = normalize_tags(&["Rust".into(), " rust ".into(), "".into(), "DB".into()]).unwrap();
        assert_eq!(t, vec!["rust", "db"]);
        let many: Vec<String> = (0..17).map(|i| format!("t{i}")).collect();
        assert!(normalize_tags(&many).is_err());
        assert!(normalize_tags(&["x".repeat(49)]).is_err());
    }

    #[test]
    fn memory_type_roundtrip() {
        for t in [
            MemoryType::Episodic,
            MemoryType::Semantic,
            MemoryType::Procedural,
        ] {
            assert_eq!(MemoryType::parse(t.as_str()).unwrap(), t);
            let j = serde_json::to_string(&t).unwrap();
            assert_eq!(j, format!("\"{}\"", t.as_str()));
        }
        assert!(MemoryType::parse("nope").is_err());
    }

    #[test]
    fn content_validation() {
        assert!(validate_content("   ", 10).is_err());
        assert!(validate_content("hello world!!", 10).is_err());
        assert_eq!(validate_content(" hi ", 10).unwrap(), "hi");
    }

    #[test]
    fn record_serde_roundtrip() {
        let r = MemoryRecord {
            id: "x".into(),
            content: "c".into(),
            memory_type: MemoryType::Procedural,
            project_id: "p".into(),
            agent_source: "codex".into(),
            tags: vec!["a".into()],
            created_at: 1,
            last_accessed: 2,
            access_count: 3,
            weight: 1.5,
        };
        let j = serde_json::to_string(&r).unwrap();
        assert_eq!(serde_json::from_str::<MemoryRecord>(&j).unwrap(), r);
    }

    #[test]
    fn recall_request_defaults() {
        let r: RecallRequest = serde_json::from_str(r#"{"query":"q"}"#).unwrap();
        assert_eq!(r.top_k, 5);
        assert!(r.include_global);
    }
}

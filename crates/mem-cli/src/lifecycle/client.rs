//! Small helpers for talking to a running daemon (health probes, token, project detection).

use std::path::{Path, PathBuf};
use std::time::Duration;

use mem_core::Config;
use serde_json::Value;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Health {
    Ready,
    Loading,
    Failed(String),
    /// Something that is not agent-mem answers on the address.
    Foreign,
    Down,
}

pub fn http_client(timeout: Duration) -> reqwest::Client {
    reqwest::Client::builder()
        .timeout(timeout)
        .connect_timeout(Duration::from_secs(2))
        .no_proxy()
        .build()
        .unwrap_or_default()
}

pub async fn health(http: &reqwest::Client, addr: &str) -> Health {
    let resp = match http.get(format!("http://{addr}/health")).send().await {
        Ok(r) => r,
        Err(_) => return Health::Down,
    };
    let status = resp.status();
    let body: Value = match resp.json().await {
        Ok(v) => v,
        Err(_) => return Health::Foreign,
    };
    if body.get("service").and_then(Value::as_str) != Some("agent-mem") {
        return Health::Foreign;
    }
    match body.get("state").and_then(Value::as_str) {
        Some("ready") if status.is_success() => Health::Ready,
        Some("loading") => Health::Loading,
        Some("failed") => Health::Failed(
            body.get("error")
                .and_then(Value::as_str)
                .unwrap_or("unknown error")
                .to_string(),
        ),
        _ => Health::Foreign,
    }
}

pub fn read_token(cfg: &Config) -> Option<String> {
    let t = std::fs::read_to_string(cfg.token_path()).ok()?;
    let t = t.trim().to_string();
    if t.is_empty() {
        None
    } else {
        Some(t)
    }
}

pub fn tail_log(cfg: &Config, lines: usize) -> String {
    let text = std::fs::read_to_string(cfg.log_path()).unwrap_or_default();
    let all: Vec<&str> = text.lines().collect();
    let start = all.len().saturating_sub(lines);
    all[start..].join("\n")
}

fn sanitize_project(name: &str) -> Option<String> {
    let s: String = name
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-') {
                c
            } else {
                '-'
            }
        })
        .collect();
    let s = s.trim_matches('-').to_string();
    if s.is_empty() {
        None
    } else {
        Some(s.chars().take(128).collect())
    }
}

/// Basename of the nearest enclosing git repository, else of `dir` itself.
pub fn project_for_dir(dir: &Path) -> Option<String> {
    let mut cur: Option<&Path> = Some(dir);
    while let Some(d) = cur {
        if d.join(".git").exists() {
            return d
                .file_name()
                .and_then(|n| n.to_str())
                .and_then(sanitize_project);
        }
        cur = d.parent();
    }
    dir.file_name()
        .and_then(|n| n.to_str())
        .and_then(sanitize_project)
}

pub fn default_project() -> Option<String> {
    let cwd: PathBuf = std::env::current_dir().ok()?;
    project_for_dir(&cwd)
}

/// Maps an MCP `clientInfo.name` to a stable agent id.
pub fn agent_from_client_name(name: &str) -> String {
    let n = name.to_ascii_lowercase();
    if n.contains("claude") {
        "claude-code".into()
    } else if n.contains("opencode") {
        "opencode".into()
    } else if n.contains("codex") {
        "codex".into()
    } else if n.contains("antigravity") || n.contains("gemini") {
        "antigravity".into()
    } else {
        let s: String = n
            .chars()
            .map(|c| {
                if c.is_ascii_alphanumeric() || c == '-' {
                    c
                } else {
                    '-'
                }
            })
            .take(64)
            .collect();
        if s.trim_matches('-').is_empty() {
            "unknown".into()
        } else {
            s
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn project_from_git_root_and_fallback() {
        let d = tempfile::tempdir().unwrap();
        let repo = d.path().join("my repo!");
        std::fs::create_dir_all(repo.join(".git")).unwrap();
        let sub = repo.join("src/deep");
        std::fs::create_dir_all(&sub).unwrap();
        assert_eq!(project_for_dir(&sub).as_deref(), Some("my-repo"));
        let plain = d.path().join("plain");
        std::fs::create_dir_all(&plain).unwrap();
        assert_eq!(project_for_dir(&plain).as_deref(), Some("plain"));
    }

    #[test]
    fn git_file_worktree_counts() {
        let d = tempfile::tempdir().unwrap();
        let wt = d.path().join("wt");
        std::fs::create_dir_all(&wt).unwrap();
        std::fs::write(wt.join(".git"), "gitdir: /elsewhere").unwrap();
        assert_eq!(project_for_dir(&wt).as_deref(), Some("wt"));
    }

    #[test]
    fn agent_names() {
        assert_eq!(agent_from_client_name("claude-code"), "claude-code");
        assert_eq!(agent_from_client_name("Claude Desktop"), "claude-code");
        assert_eq!(agent_from_client_name("opencode"), "opencode");
        assert_eq!(agent_from_client_name("codex-mcp-client"), "codex");
        assert_eq!(agent_from_client_name("Antigravity"), "antigravity");
        assert_eq!(agent_from_client_name("My Tool"), "my-tool");
        assert_eq!(agent_from_client_name("***"), "unknown");
    }
}

//! `agent-mem install`: idempotent, atomic, backed-up registration of the MCP server in
//! Claude Code, OpenCode, Codex CLI and Google Antigravity configs.

pub mod command;

use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Context, Result};
use clap::ValueEnum;
use serde_json::{json, Map, Value};
use toml_edit::{value, Array, DocumentMut, Item, Table};

pub const SERVER_NAME: &str = "agent-mem";

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum ClientKind {
    #[value(alias = "claude-code")]
    Claude,
    Opencode,
    Codex,
    Antigravity,
}

impl ClientKind {
    pub const ALL: [ClientKind; 4] = [Self::Claude, Self::Opencode, Self::Codex, Self::Antigravity];

    pub fn label(&self) -> &'static str {
        match self {
            Self::Claude => "claude-code",
            Self::Opencode => "opencode",
            Self::Codex => "codex",
            Self::Antigravity => "antigravity",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Status {
    Added,
    Updated,
    Unchanged,
    Removed,
    NotInstalled,
    Skipped(String),
    WouldAdd(String),
    WouldUpdate(String),
    WouldRemove,
    Failed(String),
}

impl Status {
    pub fn is_failure(&self) -> bool {
        matches!(self, Status::Failed(_))
    }
}

#[derive(Debug, Clone)]
pub struct Outcome {
    pub client: ClientKind,
    pub path: Option<PathBuf>,
    pub status: Status,
}

#[derive(Debug, Clone)]
pub struct Options {
    pub dry_run: bool,
    pub uninstall: bool,
    /// Configure clients even if they were not detected (set when `--clients` is given).
    pub force: bool,
    /// Used in backup file names (`*.bak.<ts>`).
    pub timestamp: u64,
}

pub fn unix_ts() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

// ------------------------------------------------------------ stable binary

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CopyStatus {
    /// Already running from (or identical to) the destination.
    Same,
    Copied,
    WouldCopy,
}

pub fn stable_exe(data_dir: &Path) -> PathBuf {
    data_dir.join("bin").join(if cfg!(windows) {
        "agent-mem.exe"
    } else {
        "agent-mem"
    })
}

fn same_bytes(a: &Path, b: &Path) -> bool {
    let (Ok(ma), Ok(mb)) = (std::fs::metadata(a), std::fs::metadata(b)) else {
        return false;
    };
    ma.len() == mb.len() && matches!((std::fs::read(a), std::fs::read(b)), (Ok(x), Ok(y)) if x == y)
}

/// Copies the running binary to `dst` (atomically) so client configs never point into a
/// build directory or a temporary download folder.
pub fn install_binary(src: &Path, dst: &Path, dry_run: bool) -> Result<CopyStatus> {
    if let (Ok(a), Ok(b)) = (std::fs::canonicalize(src), std::fs::canonicalize(dst)) {
        if a == b {
            return Ok(CopyStatus::Same);
        }
    }
    if dst.exists() && same_bytes(src, dst) {
        return Ok(CopyStatus::Same);
    }
    if dry_run {
        return Ok(CopyStatus::WouldCopy);
    }
    let parent = dst
        .parent()
        .ok_or_else(|| anyhow!("invalid destination {}", dst.display()))?;
    std::fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
    let tmp = dst.with_extension("new");
    std::fs::copy(src, &tmp).with_context(|| format!("copying binary to {}", tmp.display()))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&tmp, std::fs::Permissions::from_mode(0o755))?;
    }
    // rename replaces the destination even if a daemon is currently running from it (unix).
    std::fs::rename(&tmp, dst).with_context(|| format!("replacing {}", dst.display()))?;
    Ok(CopyStatus::Copied)
}

// ---------------------------------------------------------------- detection

pub fn on_path(bin: &str) -> bool {
    let Some(paths) = std::env::var_os("PATH") else {
        return false;
    };
    std::env::split_paths(&paths).any(|dir| {
        let p = dir.join(bin);
        p.is_file() || (cfg!(windows) && dir.join(format!("{bin}.exe")).is_file())
    })
}

fn antigravity_candidates(home: &Path) -> [PathBuf; 2] {
    [
        home.join(".gemini/config/mcp_config.json"),
        home.join(".gemini/antigravity/mcp_config.json"),
    ]
}

// ------------------------------------------------------------------ entries

fn claude_entry(exe: &str) -> Value {
    json!({"type": "stdio", "command": exe, "args": ["mcp"], "env": {}})
}
fn opencode_entry(exe: &str) -> Value {
    json!({"type": "local", "command": [exe, "mcp"], "enabled": true})
}
fn antigravity_entry(exe: &str) -> Value {
    json!({"command": exe, "args": ["mcp"]})
}

// ----------------------------------------------------------------- file I/O

fn write_atomic(path: &Path, content: &str, existed: bool, ts: u64) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating {}", parent.display()))?;
    }
    if existed {
        let mut bak = path.as_os_str().to_os_string();
        bak.push(format!(".bak.{ts}"));
        std::fs::copy(path, PathBuf::from(&bak))
            .with_context(|| format!("backing up {}", path.display()))?;
    }
    let name = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("config");
    let tmp = path.with_file_name(format!(".{name}.agent-mem.tmp"));
    std::fs::write(&tmp, content).with_context(|| format!("writing {}", tmp.display()))?;
    if existed {
        if let Ok(meta) = std::fs::metadata(path) {
            let _ = std::fs::set_permissions(&tmp, meta.permissions());
        }
    }
    if let Err(e) = std::fs::rename(&tmp, path) {
        let _ = std::fs::remove_file(&tmp);
        return Err(anyhow!(e).context(format!("replacing {}", path.display())));
    }
    Ok(())
}

enum JsonOp<'a> {
    Set {
        section: &'a str,
        entry: Value,
        init: Option<(&'a str, Value)>,
    },
    Remove {
        section: &'a str,
    },
}

fn apply_json(path: &Path, op: JsonOp<'_>, opts: &Options) -> Result<Status> {
    let existing_text = match std::fs::read_to_string(path) {
        Ok(t) => Some(t),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
        Err(e) => return Err(anyhow!(e).context(format!("reading {}", path.display()))),
    };
    let mut root: Map<String, Value> = match &existing_text {
        Some(t) if !t.trim().is_empty() => match serde_json::from_str::<Value>(t) {
            Ok(Value::Object(m)) => m,
            Ok(_) => bail!(
                "{} is not a JSON object; leaving it untouched",
                path.display()
            ),
            Err(e) => bail!(
                "{} is not valid JSON ({e}); leaving it untouched",
                path.display()
            ),
        },
        _ => Map::new(),
    };
    let fresh = existing_text
        .as_deref()
        .map(|t| t.trim().is_empty())
        .unwrap_or(true);

    let status = match op {
        JsonOp::Set {
            section,
            entry,
            init,
        } => {
            if fresh {
                if let Some((k, v)) = init {
                    root.insert(k.to_string(), v);
                }
            }
            let sec = root
                .entry(section.to_string())
                .or_insert_with(|| Value::Object(Map::new()));
            let Some(sec) = sec.as_object_mut() else {
                bail!(
                    "\"{section}\" in {} is not an object; leaving it untouched",
                    path.display()
                );
            };
            match sec.get(SERVER_NAME) {
                Some(cur) if cur == &entry => return Ok(Status::Unchanged),
                cur => {
                    let was = cur.is_some();
                    let snippet = serde_json::to_string_pretty(
                        &json!({ section: { SERVER_NAME: entry.clone() } }),
                    )?;
                    sec.insert(SERVER_NAME.to_string(), entry);
                    if opts.dry_run {
                        return Ok(if was {
                            Status::WouldUpdate(snippet)
                        } else {
                            Status::WouldAdd(snippet)
                        });
                    }
                    if was {
                        Status::Updated
                    } else {
                        Status::Added
                    }
                }
            }
        }
        JsonOp::Remove { section } => {
            let removed = root
                .get_mut(section)
                .and_then(Value::as_object_mut)
                .and_then(|s| s.shift_remove(SERVER_NAME))
                .is_some();
            if !removed {
                return Ok(Status::NotInstalled);
            }
            if opts.dry_run {
                return Ok(Status::WouldRemove);
            }
            Status::Removed
        }
    };
    let mut out = serde_json::to_string_pretty(&Value::Object(root))?;
    out.push('\n');
    write_atomic(path, &out, existing_text.is_some(), opts.timestamp)?;
    Ok(status)
}

fn codex_expected(table: &dyn toml_edit::TableLike, exe: &str) -> bool {
    let cmd_ok = table.get("command").and_then(Item::as_str) == Some(exe);
    let args_ok = table
        .get("args")
        .and_then(Item::as_array)
        .map(|a| a.len() == 1 && a.get(0).and_then(|v| v.as_str()) == Some("mcp"))
        .unwrap_or(false);
    cmd_ok && args_ok
}

fn apply_codex(path: &Path, exe: &str, opts: &Options) -> Result<Status> {
    let existing_text = match std::fs::read_to_string(path) {
        Ok(t) => Some(t),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
        Err(e) => return Err(anyhow!(e).context(format!("reading {}", path.display()))),
    };
    let mut doc: DocumentMut = match &existing_text {
        Some(t) => t.parse().map_err(|e| {
            anyhow!(
                "{} is not valid TOML ({e}); leaving it untouched",
                path.display()
            )
        })?,
        None => DocumentMut::new(),
    };

    if opts.uninstall {
        let removed = doc
            .get_mut("mcp_servers")
            .and_then(Item::as_table_like_mut)
            .and_then(|t| t.remove(SERVER_NAME))
            .is_some();
        if !removed {
            return Ok(Status::NotInstalled);
        }
        if opts.dry_run {
            return Ok(Status::WouldRemove);
        }
        write_atomic(path, &doc.to_string(), true, opts.timestamp)?;
        return Ok(Status::Removed);
    }

    let mut entry = Table::new();
    entry.insert("command", value(exe));
    let mut args = Array::new();
    args.push("mcp");
    entry.insert("args", value(args));

    let servers_item = doc.entry("mcp_servers").or_insert_with(|| {
        let mut t = Table::new();
        t.set_implicit(true);
        Item::Table(t)
    });
    let Some(servers) = servers_item.as_table_like_mut() else {
        bail!(
            "`mcp_servers` in {} is not a table; leaving it untouched",
            path.display()
        );
    };
    let was = match servers.get(SERVER_NAME) {
        Some(cur) => {
            if cur.as_table_like().is_some_and(|t| codex_expected(t, exe)) {
                return Ok(Status::Unchanged);
            }
            true
        }
        None => false,
    };
    let snippet = format!(
        "[mcp_servers.{SERVER_NAME}]\ncommand = {}\nargs = [\"mcp\"]",
        value(exe)
            .into_value()
            .map(|v| v.to_string().trim().to_string())
            .unwrap_or_default()
    );
    servers.insert(SERVER_NAME, Item::Table(entry));
    if opts.dry_run {
        return Ok(if was {
            Status::WouldUpdate(snippet)
        } else {
            Status::WouldAdd(snippet)
        });
    }
    write_atomic(
        path,
        &doc.to_string(),
        existing_text.is_some(),
        opts.timestamp,
    )?;
    Ok(if was { Status::Updated } else { Status::Added })
}

// ------------------------------------------------------------- orchestration

fn json_install(client: ClientKind, path: &Path, exe: &str, opts: &Options) -> Result<Status> {
    let op = if opts.uninstall {
        JsonOp::Remove {
            section: if client == ClientKind::Opencode {
                "mcp"
            } else {
                "mcpServers"
            },
        }
    } else {
        match client {
            ClientKind::Claude => JsonOp::Set {
                section: "mcpServers",
                entry: claude_entry(exe),
                init: None,
            },
            ClientKind::Opencode => JsonOp::Set {
                section: "mcp",
                entry: opencode_entry(exe),
                init: Some(("$schema", json!("https://opencode.ai/config.json"))),
            },
            _ => JsonOp::Set {
                section: "mcpServers",
                entry: antigravity_entry(exe),
                init: None,
            },
        }
    };
    apply_json(path, op, opts)
}

fn run_one(client: ClientKind, path: &Path, exe: &str, opts: &Options) -> Outcome {
    let r = if client == ClientKind::Codex {
        apply_codex(path, exe, opts)
    } else {
        json_install(client, path, exe, opts)
    };
    Outcome {
        client,
        path: Some(path.to_path_buf()),
        status: r.unwrap_or_else(|e| Status::Failed(format!("{e:#}"))),
    }
}

fn skipped(client: ClientKind, why: &str) -> Outcome {
    Outcome {
        client,
        path: None,
        status: Status::Skipped(why.to_string()),
    }
}

/// Registers (or removes) the server for each requested client under `home`.
pub fn run(home: &Path, exe: &Path, clients: &[ClientKind], opts: &Options) -> Vec<Outcome> {
    let exe_s = exe.to_string_lossy().to_string();
    let mut out = Vec::new();
    for &client in clients {
        match client {
            ClientKind::Claude => {
                let path = home.join(".claude.json");
                let present = path.exists() || home.join(".claude").is_dir() || on_path("claude");
                if opts.uninstall && !path.exists() {
                    out.push(skipped(client, "no config file"));
                } else if !present && !opts.force {
                    out.push(skipped(client, "not detected"));
                } else {
                    out.push(run_one(client, &path, &exe_s, opts));
                }
            }
            ClientKind::Opencode => {
                let dir = home.join(".config/opencode");
                let path = dir.join("opencode.json");
                let present = dir.is_dir() || on_path("opencode");
                if opts.uninstall && !path.exists() {
                    out.push(skipped(client, "no config file"));
                } else if !present && !opts.force {
                    out.push(skipped(client, "not detected"));
                } else {
                    out.push(run_one(client, &path, &exe_s, opts));
                }
            }
            ClientKind::Codex => {
                let dir = home.join(".codex");
                let path = dir.join("config.toml");
                let present = dir.is_dir() || on_path("codex");
                if opts.uninstall && !path.exists() {
                    out.push(skipped(client, "no config file"));
                } else if !present && !opts.force {
                    out.push(skipped(client, "not detected"));
                } else {
                    out.push(run_one(client, &path, &exe_s, opts));
                }
            }
            ClientKind::Antigravity => {
                let cands = antigravity_candidates(home);
                let existing: Vec<&PathBuf> = cands.iter().filter(|p| p.is_file()).collect();
                if !existing.is_empty() {
                    for p in existing {
                        out.push(run_one(client, p, &exe_s, opts));
                    }
                } else if opts.uninstall {
                    out.push(skipped(client, "no config file"));
                } else {
                    let present = home.join(".gemini").is_dir() || on_path("antigravity");
                    if !present && !opts.force {
                        out.push(skipped(client, "not detected"));
                    } else {
                        // Prefer the candidate whose directory already exists, else the first.
                        let target = cands
                            .iter()
                            .find(|p| p.parent().is_some_and(Path::is_dir))
                            .unwrap_or(&cands[0]);
                        out.push(run_one(client, target, &exe_s, opts));
                    }
                }
            }
        }
    }
    out
}

pub fn describe(o: &Outcome) -> String {
    let path = o
        .path
        .as_ref()
        .map(|p| p.display().to_string())
        .unwrap_or_default();
    let (mark, text) = match &o.status {
        Status::Added => ("+", "added".to_string()),
        Status::Updated => ("~", "updated".to_string()),
        Status::Unchanged => ("=", "already up to date".to_string()),
        Status::Removed => ("-", "removed".to_string()),
        Status::NotInstalled => ("=", "not installed".to_string()),
        Status::Skipped(why) => ("·", format!("skipped ({why})")),
        Status::WouldAdd(s) => ("+", format!("would add:\n{}", indent(s))),
        Status::WouldUpdate(s) => ("~", format!("would update:\n{}", indent(s))),
        Status::WouldRemove => ("-", "would remove".to_string()),
        Status::Failed(e) => ("✗", format!("FAILED: {e}")),
    };
    format!("{mark} {:<12} {path}  {text}", o.client.label())
}

fn indent(s: &str) -> String {
    s.lines()
        .map(|l| format!("      {l}"))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Adds/updates the delimited instruction block in `path` (AGENTS.md / CLAUDE.md).
pub fn write_instructions(path: &Path, dry_run: bool, ts: u64) -> Result<String> {
    let existing = match std::fs::read_to_string(path) {
        Ok(t) => Some(t),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
        Err(e) => return Err(anyhow!(e).context(format!("reading {}", path.display()))),
    };
    let new = crate::template::upsert(existing.as_deref().unwrap_or(""));
    if existing.as_deref() == Some(new.as_str()) {
        return Ok(format!(
            "= instructions already up to date in {}",
            path.display()
        ));
    }
    if dry_run {
        return Ok(format!(
            "~ would write instructions block to {}",
            path.display()
        ));
    }
    write_atomic(path, &new, existing.is_some(), ts)?;
    Ok(format!("+ wrote instructions block to {}", path.display()))
}

#[cfg(test)]
mod tests;

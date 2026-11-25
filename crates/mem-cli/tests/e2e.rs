#![allow(clippy::unwrap_used, clippy::expect_used)]
//! End-to-end tests against the real `agent-mem` binary, using the hashing fake embedder
//! (AGENT_MEM_FAKE_EMBEDDER=1) so no model download is needed.

use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

use serde_json::{json, Value};

const BIN: &str = env!("CARGO_BIN_EXE_agent-mem");

struct Env {
    _tmp: tempfile::TempDir,
    data_dir: PathBuf,
    project_dir: PathBuf,
    addr: String,
}

impl Env {
    fn new(project_name: &str) -> Env {
        let tmp = tempfile::tempdir().unwrap();
        let data_dir = tmp.path().join("data");
        let project_dir = tmp.path().join(project_name);
        std::fs::create_dir_all(project_dir.join(".git")).unwrap();
        let port = {
            let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
            l.local_addr().unwrap().port()
        };
        Env {
            _tmp: tmp,
            data_dir,
            project_dir,
            addr: format!("127.0.0.1:{port}"),
        }
    }

    fn command(&self) -> Command {
        let mut c = Command::new(BIN);
        c.env("AGENT_MEM_DATA_DIR", &self.data_dir)
            .env("AGENT_MEM_ADDR", &self.addr)
            .env("AGENT_MEM_FAKE_EMBEDDER", "1")
            .env_remove("AGENT_MEM_LOG")
            .current_dir(&self.project_dir);
        c
    }

    fn stop(&self) {
        let _ = self.command().arg("stop").output();
    }
}

impl Drop for Env {
    fn drop(&mut self) {
        self.stop();
        // belt and braces: kill by pid file if graceful stop failed
        if let Ok(pid) = std::fs::read_to_string(self.data_dir.join("daemon.pid")) {
            let _ = Command::new("kill").arg(pid.trim()).output();
        }
    }
}

struct ProxyProc {
    child: Child,
    stdin: Option<ChildStdin>,
    stdout: BufReader<ChildStdout>,
    next_id: i64,
}

impl ProxyProc {
    fn spawn(env: &Env) -> ProxyProc {
        let mut child = env
            .command()
            .arg("mcp")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .unwrap();
        let stdin = child.stdin.take();
        let stdout = BufReader::new(child.stdout.take().unwrap());
        ProxyProc {
            child,
            stdin,
            stdout,
            next_id: 1,
        }
    }

    fn send(&mut self, v: &Value) {
        let s = self.stdin.as_mut().unwrap();
        writeln!(s, "{v}").unwrap();
        s.flush().unwrap();
    }

    fn rpc(&mut self, method: &str, params: Value) -> Value {
        let id = self.next_id;
        self.next_id += 1;
        self.send(&json!({"jsonrpc":"2.0","id":id,"method":method,"params":params}));
        let mut line = String::new();
        self.stdout.read_line(&mut line).unwrap();
        let v: Value =
            serde_json::from_str(line.trim()).unwrap_or_else(|e| panic!("bad line {line:?}: {e}"));
        assert_eq!(v["id"], id, "response id mismatch: {v}");
        v
    }

    fn handshake(&mut self, client: &str) {
        let r = self.rpc(
            "initialize",
            json!({"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":client,"version":"1"}}),
        );
        assert_eq!(r["result"]["serverInfo"]["name"], "agent-mem");
        self.send(&json!({"jsonrpc":"2.0","method":"notifications/initialized"}));
    }

    fn tool(&mut self, name: &str, args: Value) -> (bool, String) {
        let r = self.rpc("tools/call", json!({"name":name,"arguments":args}));
        let res = &r["result"];
        (
            res["isError"].as_bool().unwrap_or(true),
            res["content"][0]["text"].as_str().unwrap_or("").to_string(),
        )
    }

    fn close(mut self) {
        drop(self.stdin.take());
        let _ = self.child.wait();
    }
}

fn daemon_processes(data_dir: &Path) -> usize {
    let out = Command::new("ps")
        .args(["-axo", "command"])
        .output()
        .unwrap();
    let text = String::from_utf8_lossy(&out.stdout);
    let needle = data_dir.to_string_lossy().to_string();
    text.lines()
        .filter(|l| l.contains(&needle) && l.contains(" daemon ") && l.contains("--foreground"))
        .count()
}

#[test]
fn two_agents_share_one_daemon_and_memory() {
    let env = Env::new("shared-repo");
    let mut a = ProxyProc::spawn(&env);
    let mut b = ProxyProc::spawn(&env);
    a.handshake("claude-code");
    b.handshake("opencode");

    let (err, text) = a.tool(
        "remember",
        json!({"content":"The billing service retries failed webhooks three times with exponential backoff","tags":["billing"]}),
    );
    assert!(!err, "{text}");
    let id = text
        .trim_start_matches("Stored memory ")
        .trim_end_matches('.')
        .to_string();

    // Agent B sees agent A's memory; project defaulted from the git root name in both.
    let (err, text) = b.tool("recall", json!({"query":"webhook retries backoff"}));
    assert!(!err, "{text}");
    assert!(text.contains(&id) && text.contains("shared-repo"), "{text}");

    // Cross-project isolation + global visibility.
    let (_, text) = b.tool("recall", json!({"query":"webhook retries backoff","project_id":"other-project","include_global":false}));
    assert_eq!(text, "No relevant memories found.");
    b.tool(
        "remember",
        json!({"content":"User prefers tabs over spaces everywhere","project_id":"global"}),
    );
    let (_, text) = a.tool(
        "recall",
        json!({"query":"tabs spaces preference","project_id":"unrelated"}),
    );
    assert!(text.contains("tabs over spaces"), "{text}");

    // Provenance: the agent source comes from clientInfo.
    let token = std::fs::read_to_string(env.data_dir.join("token")).unwrap();
    let out = Command::new("curl")
        .args([
            "-s",
            "-H",
            &format!("Authorization: Bearer {}", token.trim()),
        ])
        .arg(format!("http://{}/v1/stats", env.addr))
        .output()
        .unwrap();
    let stats: Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(stats["total"], 2);

    let (err, text) = a.tool("forget", json!({"memory_id": id}));
    assert!(!err, "{text}");
    let (_, text) = b.tool("recall", json!({"query":"webhook retries backoff"}));
    assert_eq!(text, "No relevant memories found.");

    let (err, text) = b.tool("inspect_stats", json!({}));
    assert!(!err);
    let s: Value = serde_json::from_str(&text).unwrap();
    assert_eq!(s["total"], 1);
    assert_eq!(s["model"], "fake-hash-384");

    a.close();
    b.close();
    assert_eq!(
        daemon_processes(&env.data_dir),
        1,
        "daemon must outlive its proxies"
    );
}

#[test]
fn concurrent_proxies_start_exactly_one_daemon() {
    let env = Env::new("race-repo");
    let handles: Vec<_> = (0..8)
        .map(|i| {
            let mut p = ProxyProc::spawn(&env);
            std::thread::spawn(move || {
                p.handshake("codex");
                let (err, text) = p.tool(
                    "remember",
                    json!({"content": format!("concurrent fact number {i} about zebra{i} habitat{i}")}),
                );
                assert!(!err, "{text}");
                p.close();
            })
        })
        .collect();
    for h in handles {
        h.join().unwrap();
    }
    assert_eq!(daemon_processes(&env.data_dir), 1);
    let mut p = ProxyProc::spawn(&env);
    p.handshake("codex");
    let (_, text) = p.tool("inspect_stats", json!({}));
    let s: Value = serde_json::from_str(&text).unwrap();
    assert_eq!(s["total"], 8);
    p.close();
}

#[test]
fn memory_survives_daemon_restart() {
    let env = Env::new("persist-repo");
    let mut p = ProxyProc::spawn(&env);
    p.handshake("claude-code");
    let (err, _) = p.tool(
        "remember",
        json!({"content":"Persisted decision: use WAL mode for sqlite"}),
    );
    assert!(!err);
    p.close();

    let out = env.command().arg("stop").output().unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(daemon_processes(&env.data_dir), 0);

    // New proxy auto-spawns a fresh daemon that reloads index + db.
    let mut p = ProxyProc::spawn(&env);
    p.handshake("claude-code");
    let (err, text) = p.tool("recall", json!({"query":"sqlite WAL decision"}));
    assert!(!err, "{text}");
    assert!(text.contains("Persisted decision"), "{text}");
    p.close();
}

#[test]
fn daemon_subcommand_status_and_stop() {
    let env = Env::new("cli-repo");
    let out = env.command().arg("daemon").output().unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let out = env.command().arg("status").output().unwrap();
    let text = String::from_utf8_lossy(&out.stdout);
    assert!(text.contains("running"), "{text}");
    assert!(text.contains("\"total\": 0"), "{text}");
    // starting again is a no-op
    let out = env.command().arg("daemon").output().unwrap();
    assert!(String::from_utf8_lossy(&out.stdout).contains("already running"));
    let out = env.command().arg("stop").output().unwrap();
    assert!(out.status.success());
    let out = env.command().arg("status").output().unwrap();
    assert!(String::from_utf8_lossy(&out.stdout).contains("not running"));
}

#[test]
fn unauthenticated_http_is_rejected() {
    let env = Env::new("auth-repo");
    assert!(env
        .command()
        .arg("daemon")
        .output()
        .unwrap()
        .status
        .success());
    let code = |args: &[&str]| -> String {
        let out = Command::new("curl")
            .args(["-s", "-o", "/dev/null", "-w", "%{http_code}"])
            .args(args)
            .output()
            .unwrap();
        String::from_utf8_lossy(&out.stdout).to_string()
    };
    let base = format!("http://{}", env.addr);
    assert_eq!(code(&[&format!("{base}/health")]), "200");
    assert_eq!(code(&[&format!("{base}/v1/stats")]), "401");
    assert_eq!(
        code(&[
            "-H",
            "Origin: https://evil.example",
            &format!("{base}/health")
        ]),
        "403"
    );
    assert_eq!(
        code(&["-H", "Host: evil.example", &format!("{base}/health")]),
        "403"
    );
    let perms = std::fs::metadata(env.data_dir.join("token")).unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        assert_eq!(perms.permissions().mode() & 0o777, 0o600);
    }
    let _ = perms;
}

#[test]
fn non_loopback_address_is_refused() {
    let env = Env::new("bind-repo");
    let out = env
        .command()
        .args(["--addr", "0.0.0.0:4999", "daemon", "--foreground"])
        .output()
        .unwrap();
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("loopback"));
}

#[test]
fn install_dry_run_and_print_template_via_cli() {
    let env = Env::new("install-repo");
    let out = env
        .command()
        .args(["install", "--print-template"])
        .output()
        .unwrap();
    let text = String::from_utf8_lossy(&out.stdout);
    assert!(text.contains("agent-mem:begin") && text.contains("Never remember"));
    // dry-run must not touch the fake HOME's files
    let home = env.data_dir.parent().unwrap().join("home");
    std::fs::create_dir_all(home.join(".codex")).unwrap();
    let out = env
        .command()
        .env("HOME", &home)
        .env("PATH", "/usr/bin:/bin")
        .args(["install", "--dry-run", "--clients", "codex"])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(String::from_utf8_lossy(&out.stdout).contains("would add"));
    assert!(!home.join(".codex/config.toml").exists());
    // real install then uninstall
    assert!(env
        .command()
        .env("HOME", &home)
        .env("PATH", "/usr/bin:/bin")
        .args(["install", "--clients", "codex"])
        .output()
        .unwrap()
        .status
        .success());
    let t = std::fs::read_to_string(home.join(".codex/config.toml")).unwrap();
    assert!(
        t.contains("[mcp_servers.agent-mem]") && t.contains("args = [\"mcp\"]"),
        "{t}"
    );
    assert!(env
        .command()
        .env("HOME", &home)
        .args(["install", "--uninstall", "--clients", "codex"])
        .output()
        .unwrap()
        .status
        .success());
    assert!(!std::fs::read_to_string(home.join(".codex/config.toml"))
        .unwrap()
        .contains("agent-mem"));
}

#[test]
fn instructions_only_writes_just_the_markdown_block() {
    let env = Env::new("instr-repo");
    let home = env.data_dir.parent().unwrap().join("home2");
    std::fs::create_dir_all(home.join(".codex")).unwrap();
    let md = home.join("AGENTS.md");
    std::fs::write(&md, "# Team rules\n").unwrap();
    let out = env
        .command()
        .env("HOME", &home)
        .args(["install", "--instructions-only", "--write-instructions"])
        .arg(&md)
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let text = std::fs::read_to_string(&md).unwrap();
    assert!(text.starts_with("# Team rules") && text.contains("agent-mem:begin"));
    assert!(
        !home.join(".codex/config.toml").exists(),
        "client configs must not be touched"
    );
    // `--instructions-only` without a file is a usage error
    let out = env
        .command()
        .args(["install", "--instructions-only"])
        .output()
        .unwrap();
    assert!(!out.status.success());
}

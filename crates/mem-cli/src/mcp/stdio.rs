//! `agent-mem mcp`: newline-delimited JSON-RPC 2.0 over stdio, proxied to the shared daemon.
//! Stdout carries protocol messages only; all logging goes to stderr.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use mem_core::Config;
use parking_lot::Mutex;
use serde_json::{json, Value};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

use crate::lifecycle::client::{agent_from_client_name, default_project, http_client, read_token};
use crate::lifecycle::spawn::Starter;
use crate::mcp::proto::{
    rpc_err, rpc_ok, static_response, ERR_INTERNAL, ERR_INVALID_REQUEST, ERR_PARSE,
};

const READY_WAIT_PER_CALL: Duration = Duration::from_secs(30);
const READY_WAIT_BACKGROUND: Duration = Duration::from_secs(180);

pub struct Proxy {
    cfg: Config,
    http: reqwest::Client,
    starter: Starter,
    ready: AtomicBool,
    token: Mutex<Option<String>>,
    agent: Mutex<String>,
    project: Option<String>,
}

impl Proxy {
    pub fn new(cfg: Config, project: Option<String>) -> Arc<Self> {
        Arc::new(Self {
            http: http_client(Duration::from_secs(15)),
            starter: Starter::new(cfg.clone()),
            cfg,
            ready: AtomicBool::new(false),
            token: Mutex::new(None),
            agent: Mutex::new("unknown".to_string()),
            project,
        })
    }

    /// Start the daemon immediately so the model loads while the client handshakes.
    pub fn warm_up(self: &Arc<Self>) {
        let p = self.clone();
        tokio::spawn(async move {
            match p.starter.ensure_ready(READY_WAIT_BACKGROUND).await {
                Ok(()) => p.ready.store(true, Ordering::Relaxed),
                Err(e) => tracing::warn!("daemon start: {e}"),
            }
        });
    }

    pub async fn handle_line(self: &Arc<Self>, line: &str) -> Option<Value> {
        match serde_json::from_str::<Value>(line) {
            Ok(v) => self.handle_value(v).await,
            Err(e) => Some(rpc_err(
                &Value::Null,
                ERR_PARSE,
                &format!("parse error: {e}"),
            )),
        }
    }

    pub async fn handle_value(self: &Arc<Self>, v: Value) -> Option<Value> {
        match v {
            Value::Array(items) => {
                if items.is_empty() {
                    return Some(rpc_err(&Value::Null, ERR_INVALID_REQUEST, "empty batch"));
                }
                let mut out = Vec::new();
                for item in items {
                    if let Some(r) = self.handle_one(item).await {
                        out.push(r);
                    }
                }
                if out.is_empty() {
                    None
                } else {
                    Some(Value::Array(out))
                }
            }
            other => self.handle_one(other).await,
        }
    }

    async fn handle_one(self: &Arc<Self>, mut v: Value) -> Option<Value> {
        let Some(obj) = v.as_object() else {
            return Some(rpc_err(
                &Value::Null,
                ERR_INVALID_REQUEST,
                "request must be a JSON object",
            ));
        };
        let Some(method) = obj
            .get("method")
            .and_then(Value::as_str)
            .map(str::to_string)
        else {
            if obj.contains_key("result") || obj.contains_key("error") {
                return None;
            }
            let id = obj.get("id").cloned().unwrap_or(Value::Null);
            return Some(rpc_err(&id, ERR_INVALID_REQUEST, "missing method"));
        };
        let id = obj.get("id").cloned();
        let params = obj.get("params").cloned().unwrap_or(Value::Null);

        if method == "initialize" {
            if let Some(name) = params.pointer("/clientInfo/name").and_then(Value::as_str) {
                *self.agent.lock() = agent_from_client_name(name);
            }
        }
        let Some(id) = id else {
            return None; // notifications are consumed locally; the daemon is stateless
        };
        if let Some(r) = static_response(&id, &method, &params) {
            return Some(r);
        }
        if method == "tools/call" {
            self.inject_default_project(&mut v);
        }
        Some(self.forward(&v, &id, &method).await)
    }

    fn inject_default_project(&self, v: &mut Value) {
        let Some(project) = &self.project else { return };
        let name = v
            .pointer("/params/name")
            .and_then(Value::as_str)
            .unwrap_or("");
        if !matches!(name, "remember" | "recall") {
            return;
        }
        let Some(params) = v.get_mut("params").and_then(Value::as_object_mut) else {
            return;
        };
        let args = params.entry("arguments").or_insert_with(|| json!({}));
        if args.is_null() {
            *args = json!({});
        }
        if let Some(a) = args.as_object_mut() {
            let missing = match a.get("project_id") {
                None | Some(Value::Null) => true,
                Some(Value::String(s)) => s.trim().is_empty(),
                _ => false,
            };
            if missing {
                a.insert("project_id".into(), Value::String(project.clone()));
            }
        }
    }

    fn token(&self) -> Option<String> {
        let mut g = self.token.lock();
        if g.is_none() {
            *g = read_token(&self.cfg);
        }
        g.clone()
    }

    fn failure(id: &Value, method: &str, msg: &str) -> Value {
        if method == "tools/call" {
            rpc_ok(
                id,
                json!({"content": [{"type": "text", "text": msg}], "isError": true}),
            )
        } else {
            rpc_err(id, ERR_INTERNAL, msg)
        }
    }

    async fn forward(self: &Arc<Self>, req: &Value, id: &Value, method: &str) -> Value {
        let mut last_err = String::from("unknown error");
        for _attempt in 0..3 {
            if !self.ready.load(Ordering::Relaxed) {
                match self.starter.ensure_ready(READY_WAIT_PER_CALL).await {
                    Ok(()) => self.ready.store(true, Ordering::Relaxed),
                    Err(e) => {
                        return Self::failure(
                            id,
                            method,
                            &format!("agent-mem is not available: {e}"),
                        )
                    }
                }
            }
            let Some(token) = self.token() else {
                last_err = format!("no token at {}", self.cfg.token_path().display());
                self.ready.store(false, Ordering::Relaxed);
                continue;
            };
            let agent = self.agent.lock().clone();
            let resp = self
                .http
                .post(format!("http://{}/mcp", self.cfg.addr))
                .bearer_auth(&token)
                .header("x-agent-source", agent)
                .header("accept", "application/json, text/event-stream")
                .json(req)
                .send()
                .await;
            match resp {
                Err(e) => {
                    last_err = format!("cannot reach daemon: {e}");
                    self.ready.store(false, Ordering::Relaxed);
                }
                Ok(r) => {
                    let status = r.status();
                    if status == reqwest::StatusCode::ACCEPTED
                        || status == reqwest::StatusCode::NO_CONTENT
                    {
                        return rpc_ok(id, json!({}));
                    }
                    if status == reqwest::StatusCode::UNAUTHORIZED {
                        *self.token.lock() = None; // daemon was restarted with a new token
                        last_err = "daemon rejected the token".into();
                        continue;
                    }
                    if status == reqwest::StatusCode::SERVICE_UNAVAILABLE {
                        self.ready.store(false, Ordering::Relaxed);
                        last_err = "daemon is still loading".into();
                        continue;
                    }
                    return match r.json::<Value>().await {
                        Ok(v) => v,
                        Err(e) => Self::failure(
                            id,
                            method,
                            &format!("bad response from daemon ({status}): {e}"),
                        ),
                    };
                }
            }
        }
        Self::failure(id, method, &format!("agent-mem request failed: {last_err}"))
    }
}

pub async fn run(cfg: Config) -> Result<()> {
    let proxy = Proxy::new(cfg, default_project());
    proxy.warm_up();

    let stdout = Arc::new(tokio::sync::Mutex::new(tokio::io::stdout()));
    let mut lines = BufReader::new(tokio::io::stdin()).lines();
    let mut tasks = tokio::task::JoinSet::new();

    while let Some(line) = lines.next_line().await? {
        if line.trim().is_empty() {
            continue;
        }
        let p = proxy.clone();
        let out = stdout.clone();
        tasks.spawn(async move {
            if let Some(resp) = p.handle_line(&line).await {
                let mut text = resp.to_string();
                text.push('\n');
                let mut o = out.lock().await;
                let _ = o.write_all(text.as_bytes()).await;
                let _ = o.flush().await;
            }
        });
    }
    // stdin closed: let in-flight requests finish and flush their responses.
    while tasks.join_next().await.is_some() {}
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn proxy(project: Option<&str>) -> Arc<Proxy> {
        let d = tempfile::tempdir().unwrap();
        let cfg = Config {
            data_dir: d.keep(),
            addr: "127.0.0.1:9".into(),
            ..Config::default()
        };
        Proxy::new(cfg, project.map(String::from))
    }

    #[tokio::test]
    async fn static_methods_answered_without_daemon() {
        let p = proxy(None);
        let r = p
            .handle_line(r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","clientInfo":{"name":"opencode","version":"1"}}}"#)
            .await
            .unwrap();
        assert_eq!(r["result"]["serverInfo"]["name"], "agent-mem");
        assert_eq!(*p.agent.lock(), "opencode");
        let r = p
            .handle_line(r#"{"jsonrpc":"2.0","id":2,"method":"tools/list"}"#)
            .await
            .unwrap();
        assert_eq!(r["result"]["tools"].as_array().unwrap().len(), 4);
        assert!(p
            .handle_line(r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#)
            .await
            .is_none());
    }

    #[tokio::test]
    async fn malformed_json_gets_parse_error() {
        let p = proxy(None);
        let r = p.handle_line("{nope").await.unwrap();
        assert_eq!(r["error"]["code"], ERR_PARSE);
        assert_eq!(r["id"], Value::Null);
    }

    #[test]
    fn default_project_injected_only_when_missing() {
        let p = proxy(Some("my-repo"));
        let mut v = json!({"params":{"name":"recall","arguments":{"query":"x"}}});
        p.inject_default_project(&mut v);
        assert_eq!(v["params"]["arguments"]["project_id"], "my-repo");

        let mut v =
            json!({"params":{"name":"remember","arguments":{"content":"x","project_id":"global"}}});
        p.inject_default_project(&mut v);
        assert_eq!(v["params"]["arguments"]["project_id"], "global");

        let mut v =
            json!({"params":{"name":"remember","arguments":{"content":"x","project_id":"  "}}});
        p.inject_default_project(&mut v);
        assert_eq!(v["params"]["arguments"]["project_id"], "my-repo");

        let mut v = json!({"params":{"name":"forget","arguments":{"memory_id":"a"}}});
        p.inject_default_project(&mut v);
        assert!(v["params"]["arguments"].get("project_id").is_none());

        let mut v = json!({"params":{"name":"recall"}});
        p.inject_default_project(&mut v);
        assert_eq!(v["params"]["arguments"]["project_id"], "my-repo");
    }

    #[test]
    fn failure_shape_depends_on_method() {
        let r = Proxy::failure(&json!(1), "tools/call", "boom");
        assert_eq!(r["result"]["isError"], true);
        let r = Proxy::failure(&json!(1), "resources/read", "boom");
        assert_eq!(r["error"]["code"], ERR_INTERNAL);
    }
}

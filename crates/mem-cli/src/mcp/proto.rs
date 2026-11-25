//! MCP / JSON-RPC 2.0 protocol logic shared by the daemon's `/mcp` endpoint and
//! the stdio proxy (which answers the static methods locally).

use std::sync::Arc;

use mem_core::{Error, RecallRequest, RememberRequest};
use mem_store::Store;
use serde::Deserialize;
use serde_json::{json, Value};

pub const LATEST_PROTOCOL: &str = "2025-06-18";
pub const SUPPORTED_PROTOCOLS: [&str; 3] = ["2025-06-18", "2025-03-26", "2024-11-05"];

pub const ERR_PARSE: i64 = -32700;
pub const ERR_INVALID_REQUEST: i64 = -32600;
pub const ERR_METHOD_NOT_FOUND: i64 = -32601;
pub const ERR_INVALID_PARAMS: i64 = -32602;
pub const ERR_INTERNAL: i64 = -32603;
pub const ERR_RESOURCE_NOT_FOUND: i64 = -32002;

pub const LOADING_MESSAGE: &str =
    "agent-mem is still starting (loading the embedding model; the first run downloads it). Retry in a few seconds.";

type RpcError = (i64, String);

pub fn rpc_ok(id: &Value, result: Value) -> Value {
    json!({"jsonrpc": "2.0", "id": id, "result": result})
}

pub fn rpc_err(id: &Value, code: i64, message: &str) -> Value {
    json!({"jsonrpc": "2.0", "id": id, "error": {"code": code, "message": message}})
}

pub fn initialize_result(requested: Option<&str>) -> Value {
    let version = match requested {
        Some(v) if SUPPORTED_PROTOCOLS.contains(&v) => v,
        _ => LATEST_PROTOCOL,
    };
    json!({
        "protocolVersion": version,
        "capabilities": {
            "tools": {"listChanged": false},
            "resources": {"subscribe": false, "listChanged": false}
        },
        "serverInfo": {"name": "agent-mem", "version": env!("CARGO_PKG_VERSION")},
        "instructions": "Shared local memory for coding agents. Call `recall` at the start of a task and before architectural changes; call `remember` only for durable, non-obvious facts (bug root causes, conventions, decisions with rationale). Never store secrets."
    })
}

pub fn tools_list() -> Value {
    json!([
        {
            "name": "remember",
            "title": "Remember",
            "description": "Store a durable, non-obvious fact: an architectural decision with its rationale, the root cause of a hard bug, a project convention, or a stable user preference. One self-contained fact per call (1-3 sentences, name the files/modules). Do NOT store anything derivable from code or git history, task progress, command output, or secrets. Near-duplicates are merged automatically.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "The fact to remember, self-contained, 1-3 sentences."},
                    "memory_type": {"type": "string", "enum": ["episodic", "semantic", "procedural"], "description": "semantic: facts/decisions (default). procedural: how-to steps/commands. episodic: what happened."},
                    "project_id": {"type": "string", "description": "Repository/project name. Defaults to the current repository. Use \"global\" for preferences that apply to every project."},
                    "tags": {"type": "array", "items": {"type": "string"}, "description": "Up to 16 short lowercase tags."}
                },
                "required": ["content"],
                "additionalProperties": false
            },
            "annotations": {"readOnlyHint": false, "destructiveHint": false, "idempotentHint": true, "openWorldHint": false}
        },
        {
            "name": "recall",
            "title": "Recall",
            "description": "Search shared memory (hybrid semantic + keyword). Call once at the start of a session with the task description, and before architectural changes or work in unfamiliar areas. Returns the most relevant memories for the project plus global ones, each with an id usable with `forget`.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What you want to know. Be specific (module names, error text)."},
                    "project_id": {"type": "string", "description": "Repository/project name. Defaults to the current repository."},
                    "top_k": {"type": "integer", "minimum": 1, "maximum": 50, "description": "Number of memories to return (default 5)."},
                    "memory_type": {"type": "string", "enum": ["episodic", "semantic", "procedural"]},
                    "tags": {"type": "array", "items": {"type": "string"}, "description": "Only memories having all of these tags."},
                    "include_global": {"type": "boolean", "description": "Also search the shared \"global\" namespace (default true)."}
                },
                "required": ["query"],
                "additionalProperties": false
            },
            "annotations": {"readOnlyHint": true, "openWorldHint": false}
        },
        {
            "name": "forget",
            "title": "Forget",
            "description": "Delete an obsolete or incorrect memory by id (ids are returned by `recall`). Store the corrected fact with `remember` afterwards.",
            "inputSchema": {
                "type": "object",
                "properties": {"memory_id": {"type": "string", "description": "Id of the memory to delete."}},
                "required": ["memory_id"],
                "additionalProperties": false
            },
            "annotations": {"readOnlyHint": false, "destructiveHint": true, "idempotentHint": true, "openWorldHint": false}
        },
        {
            "name": "inspect_stats",
            "title": "Inspect stats",
            "description": "Memory statistics: total count, counts by type and project, index size, average and p95 retrieval latency.",
            "inputSchema": {"type": "object", "properties": {}, "additionalProperties": false},
            "annotations": {"readOnlyHint": true, "openWorldHint": false}
        }
    ])
}

fn resources_list() -> Value {
    json!([
        {"uri": "agent-mem://stats", "name": "stats", "description": "Memory statistics (JSON)", "mimeType": "application/json"},
        {"uri": "agent-mem://instructions", "name": "instructions", "description": "AGENTS.md / CLAUDE.md snippet explaining when to recall and remember", "mimeType": "text/markdown"}
    ])
}

fn resource_templates() -> Value {
    json!([
        {"uriTemplate": "agent-mem://project/{project_id}/recent", "name": "recent", "description": "The 20 most recently used memories of a project", "mimeType": "application/json"}
    ])
}

/// Methods the proxy can answer without the daemon. Returns `None` for anything else.
pub fn static_response(id: &Value, method: &str, params: &Value) -> Option<Value> {
    Some(match method {
        "initialize" => rpc_ok(
            id,
            initialize_result(params.get("protocolVersion").and_then(Value::as_str)),
        ),
        "ping" => rpc_ok(id, json!({})),
        "tools/list" => rpc_ok(id, json!({"tools": tools_list()})),
        "resources/list" => rpc_ok(id, json!({"resources": resources_list()})),
        "resources/templates/list" => {
            rpc_ok(id, json!({"resourceTemplates": resource_templates()}))
        }
        "prompts/list" => rpc_ok(id, json!({"prompts": []})),
        _ => return None,
    })
}

/// Handles one JSON value (single request or batch). `None` means "no response" (notifications).
pub async fn handle_payload(
    store: Option<&Arc<Store>>,
    payload: Value,
    agent: &str,
) -> Option<Value> {
    match payload {
        Value::Array(items) => {
            if items.is_empty() {
                return Some(rpc_err(&Value::Null, ERR_INVALID_REQUEST, "empty batch"));
            }
            let mut out = Vec::new();
            for item in items {
                if let Some(r) = handle_single(store, item, agent).await {
                    out.push(r);
                }
            }
            if out.is_empty() {
                None
            } else {
                Some(Value::Array(out))
            }
        }
        other => handle_single(store, other, agent).await,
    }
}

pub async fn handle_single(store: Option<&Arc<Store>>, v: Value, agent: &str) -> Option<Value> {
    let Some(obj) = v.as_object() else {
        return Some(rpc_err(
            &Value::Null,
            ERR_INVALID_REQUEST,
            "request must be a JSON object",
        ));
    };
    let id = obj.get("id").cloned();
    let Some(method) = obj.get("method").and_then(Value::as_str) else {
        // A response to a server-initiated request: nothing to do.
        if obj.contains_key("result") || obj.contains_key("error") {
            return None;
        }
        return Some(rpc_err(
            id.as_ref().unwrap_or(&Value::Null),
            ERR_INVALID_REQUEST,
            "missing method",
        ));
    };
    let Some(id) = id else {
        return None; // notification
    };
    let params = obj.get("params").cloned().unwrap_or(Value::Null);

    if let Some(r) = static_response(&id, method, &params) {
        return Some(r);
    }
    let result: Result<Value, RpcError> = match method {
        "tools/call" => call_tool(store, &params, agent).await,
        "resources/read" => read_resource(store, &params).await,
        _ => Err((ERR_METHOD_NOT_FOUND, format!("method not found: {method}"))),
    };
    Some(match result {
        Ok(r) => rpc_ok(&id, r),
        Err((code, msg)) => rpc_err(&id, code, &msg),
    })
}

fn text_result(text: String, is_error: bool) -> Value {
    json!({"content": [{"type": "text", "text": text}], "isError": is_error})
}

#[derive(Deserialize)]
struct RememberArgs {
    content: String,
    #[serde(default)]
    memory_type: Option<String>,
    #[serde(default)]
    project_id: Option<String>,
    #[serde(default)]
    tags: Option<Vec<String>>,
}

#[derive(Deserialize)]
struct RecallArgs {
    query: String,
    #[serde(default)]
    project_id: Option<String>,
    #[serde(default)]
    top_k: Option<usize>,
    #[serde(default)]
    memory_type: Option<String>,
    #[serde(default)]
    tags: Option<Vec<String>>,
    #[serde(default)]
    include_global: Option<bool>,
}

#[derive(Deserialize)]
struct ForgetArgs {
    memory_id: String,
}

async fn call_tool(
    store: Option<&Arc<Store>>,
    params: &Value,
    agent: &str,
) -> Result<Value, RpcError> {
    let name = params.get("name").and_then(Value::as_str).ok_or((
        ERR_INVALID_PARAMS,
        "tools/call requires a tool name".to_string(),
    ))?;
    if !matches!(name, "remember" | "recall" | "forget" | "inspect_stats") {
        return Err((ERR_INVALID_PARAMS, format!("unknown tool: {name}")));
    }
    let args = params
        .get("arguments")
        .cloned()
        .unwrap_or_else(|| json!({}));
    let Some(store) = store else {
        return Ok(text_result(LOADING_MESSAGE.to_string(), true));
    };

    macro_rules! parse {
        ($t:ty) => {
            match serde_json::from_value::<$t>(args) {
                Ok(a) => a,
                Err(e) => {
                    return Ok(text_result(
                        format!("invalid arguments for {name}: {e}"),
                        true,
                    ))
                }
            }
        };
    }

    let outcome: Result<String, Error> = match name {
        "remember" => {
            let a = parse!(RememberArgs);
            store
                .remember(RememberRequest {
                    content: a.content,
                    memory_type: a.memory_type,
                    project_id: a.project_id,
                    tags: a.tags.unwrap_or_default(),
                    agent_source: Some(agent.to_string()),
                })
                .await
                .map(|r| {
                    if r.deduplicated {
                        format!(
                            "Merged into existing similar memory {} (similarity {:.2}); its weight and recency were refreshed.",
                            r.id,
                            r.similarity.unwrap_or(1.0)
                        )
                    } else {
                        format!("Stored memory {}.", r.id)
                    }
                })
        }
        "recall" => {
            let a = parse!(RecallArgs);
            store
                .recall(RecallRequest {
                    query: a.query,
                    project_id: a.project_id,
                    top_k: a.top_k.unwrap_or(5),
                    memory_type: a.memory_type,
                    tags: a.tags.unwrap_or_default(),
                    include_global: a.include_global.unwrap_or(true),
                })
                .await
                .map(|r| format_recall(&r))
        }
        "forget" => {
            let a = parse!(ForgetArgs);
            let id = a.memory_id;
            store
                .forget(id.clone())
                .await
                .map(|_| format!("Forgot memory {id}."))
        }
        _ => store.stats().await.and_then(|s| {
            serde_json::to_string_pretty(&s).map_err(|e| Error::Internal(e.to_string()))
        }),
    };
    Ok(match outcome {
        Ok(text) => text_result(text, false),
        Err(e) => text_result(e.to_string(), true),
    })
}

pub fn format_recall(r: &mem_core::RecallResponse) -> String {
    if r.hits.is_empty() {
        return "No relevant memories found.".to_string();
    }
    let mut out = format!("Found {} memories ({:.1} ms):\n", r.hits.len(), r.took_ms);
    for (i, h) in r.hits.iter().enumerate() {
        let tags = if h.tags.is_empty() {
            String::new()
        } else {
            format!(" tags={}", h.tags.join(","))
        };
        out.push_str(&format!(
            "\n{}. [{} | {} | score {:.4}] id={}{}\n   {}\n",
            i + 1,
            h.memory_type.as_str(),
            h.project_id,
            h.score,
            h.id,
            tags,
            h.content.replace('\n', "\n   ")
        ));
    }
    out
}

async fn read_resource(store: Option<&Arc<Store>>, params: &Value) -> Result<Value, RpcError> {
    let uri = params.get("uri").and_then(Value::as_str).ok_or((
        ERR_INVALID_PARAMS,
        "resources/read requires a uri".to_string(),
    ))?;
    let contents = |mime: &str, text: String| json!({"contents": [{"uri": uri, "mimeType": mime, "text": text}]});

    if uri == "agent-mem://instructions" {
        return Ok(contents("text/markdown", crate::template::block()));
    }
    let Some(store) = store else {
        return Err((ERR_INTERNAL, LOADING_MESSAGE.to_string()));
    };
    if uri == "agent-mem://stats" {
        let s = store
            .stats()
            .await
            .map_err(|e| (ERR_INTERNAL, e.to_string()))?;
        let text = serde_json::to_string_pretty(&s).map_err(|e| (ERR_INTERNAL, e.to_string()))?;
        return Ok(contents("application/json", text));
    }
    if let Some(rest) = uri.strip_prefix("agent-mem://project/") {
        if let Some(project) = rest.strip_suffix("/recent") {
            let recs = store
                .recent(project.to_string(), 20)
                .await
                .map_err(|e| (ERR_INVALID_PARAMS, e.to_string()))?;
            let text =
                serde_json::to_string_pretty(&recs).map_err(|e| (ERR_INTERNAL, e.to_string()))?;
            return Ok(contents("application/json", text));
        }
    }
    Err((ERR_RESOURCE_NOT_FOUND, format!("resource not found: {uri}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use mem_embed::FakeEmbedder;
    use mem_store::StoreOptions;

    fn store() -> (tempfile::TempDir, Arc<Store>) {
        let d = tempfile::tempdir().unwrap();
        let s = Store::open(StoreOptions::new(d.path()), Arc::new(FakeEmbedder::new())).unwrap();
        (d, s)
    }

    async fn rpc(s: Option<&Arc<Store>>, v: Value) -> Value {
        handle_payload(s, v, "test").await.unwrap()
    }

    fn call(id: i64, name: &str, args: Value) -> Value {
        json!({"jsonrpc":"2.0","id":id,"method":"tools/call","params":{"name":name,"arguments":args}})
    }

    fn text(r: &Value) -> String {
        r["result"]["content"][0]["text"]
            .as_str()
            .unwrap()
            .to_string()
    }

    #[tokio::test]
    async fn initialize_negotiates_version() {
        let r = rpc(None, json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-03-26"}})).await;
        assert_eq!(r["result"]["protocolVersion"], "2025-03-26");
        assert_eq!(r["result"]["serverInfo"]["name"], "agent-mem");
        let r = rpc(None, json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"1999-01-01"}})).await;
        assert_eq!(r["result"]["protocolVersion"], LATEST_PROTOCOL);
        assert_eq!(r["id"], 1);
    }

    #[tokio::test]
    async fn lists_four_tools_with_schemas() {
        let r = rpc(
            None,
            json!({"jsonrpc":"2.0","id":"a","method":"tools/list"}),
        )
        .await;
        assert_eq!(r["id"], "a");
        let names: Vec<&str> = r["result"]["tools"]
            .as_array()
            .unwrap()
            .iter()
            .map(|t| t["name"].as_str().unwrap())
            .collect();
        assert_eq!(names, ["remember", "recall", "forget", "inspect_stats"]);
        for t in r["result"]["tools"].as_array().unwrap() {
            assert_eq!(t["inputSchema"]["type"], "object");
            assert_eq!(t["inputSchema"]["additionalProperties"], false);
        }
    }

    #[tokio::test]
    async fn notifications_get_no_response() {
        assert!(handle_payload(
            None,
            json!({"jsonrpc":"2.0","method":"notifications/initialized"}),
            "t"
        )
        .await
        .is_none());
    }

    #[tokio::test]
    async fn unknown_method_and_bad_requests() {
        let r = rpc(None, json!({"jsonrpc":"2.0","id":2,"method":"nope"})).await;
        assert_eq!(r["error"]["code"], ERR_METHOD_NOT_FOUND);
        let r = rpc(None, json!("string")).await;
        assert_eq!(r["error"]["code"], ERR_INVALID_REQUEST);
        assert_eq!(r["id"], Value::Null);
        let r = rpc(None, json!({"jsonrpc":"2.0","id":3})).await;
        assert_eq!(r["error"]["code"], ERR_INVALID_REQUEST);
        let r = rpc(None, json!([])).await;
        assert_eq!(r["error"]["code"], ERR_INVALID_REQUEST);
    }

    #[tokio::test]
    async fn batch_skips_notifications() {
        let r = rpc(
            None,
            json!([
                {"jsonrpc":"2.0","id":1,"method":"ping"},
                {"jsonrpc":"2.0","method":"notifications/initialized"},
                {"jsonrpc":"2.0","id":2,"method":"tools/list"}
            ]),
        )
        .await;
        assert_eq!(r.as_array().unwrap().len(), 2);
    }

    #[tokio::test]
    async fn tools_roundtrip() {
        let (_d, s) = store();
        let s = Some(&s);
        let r = rpc(s, call(1, "remember", json!({"content":"Auth tokens expire after 15 minutes because of the gateway","project_id":"app","tags":["auth"]}))).await;
        assert_eq!(r["result"]["isError"], false);
        let t = text(&r);
        assert!(t.starts_with("Stored memory "), "{t}");
        let id = t
            .trim_start_matches("Stored memory ")
            .trim_end_matches('.')
            .to_string();

        let r = rpc(
            s,
            call(
                2,
                "recall",
                json!({"query":"auth token expiry","project_id":"app"}),
            ),
        )
        .await;
        let t = text(&r);
        assert!(t.contains(&id) && t.contains("gateway"), "{t}");

        let r = rpc(s, call(3, "inspect_stats", json!({}))).await;
        let stats: Value = serde_json::from_str(&text(&r)).unwrap();
        assert_eq!(stats["total"], 1);

        let r = rpc(s, call(4, "forget", json!({"memory_id": id}))).await;
        assert_eq!(r["result"]["isError"], false);
        let r = rpc(s, call(5, "forget", json!({"memory_id": id}))).await;
        assert_eq!(r["result"]["isError"], true);
        let r = rpc(
            s,
            call(
                6,
                "recall",
                json!({"query":"auth token expiry","project_id":"app"}),
            ),
        )
        .await;
        assert_eq!(text(&r), "No relevant memories found.");
    }

    #[tokio::test]
    async fn tool_errors_are_results_not_rpc_errors() {
        let (_d, s) = store();
        let s = Some(&s);
        let r = rpc(
            s,
            call(1, "remember", json!({"content":"key AKIAIOSFODNN7EXAMPLE"})),
        )
        .await;
        assert_eq!(r["result"]["isError"], true);
        assert!(text(&r).contains("secret"));
        let r = rpc(s, call(2, "remember", json!({"wrong":"field"}))).await;
        assert_eq!(r["result"]["isError"], true);
        let r = rpc(s, call(3, "does_not_exist", json!({}))).await;
        assert_eq!(r["error"]["code"], ERR_INVALID_PARAMS);
        let r = rpc(None, call(4, "recall", json!({"query":"x"}))).await;
        assert_eq!(r["result"]["isError"], true);
        assert!(text(&r).contains("still starting"));
    }

    #[tokio::test]
    async fn resources() {
        let (_d, s) = store();
        let s = Some(&s);
        rpc(
            s,
            call(
                1,
                "remember",
                json!({"content":"resource test fact","project_id":"proj"}),
            ),
        )
        .await;
        let r = rpc(s, json!({"jsonrpc":"2.0","id":1,"method":"resources/list"})).await;
        assert_eq!(r["result"]["resources"].as_array().unwrap().len(), 2);
        let r = rpc(s, json!({"jsonrpc":"2.0","id":2,"method":"resources/read","params":{"uri":"agent-mem://instructions"}})).await;
        assert!(r["result"]["contents"][0]["text"]
            .as_str()
            .unwrap()
            .contains("agent-mem:begin"));
        let r = rpc(s, json!({"jsonrpc":"2.0","id":3,"method":"resources/read","params":{"uri":"agent-mem://project/proj/recent"}})).await;
        assert!(r["result"]["contents"][0]["text"]
            .as_str()
            .unwrap()
            .contains("resource test fact"));
        let r = rpc(s, json!({"jsonrpc":"2.0","id":4,"method":"resources/read","params":{"uri":"agent-mem://stats"}})).await;
        assert!(r["result"]["contents"][0]["text"]
            .as_str()
            .unwrap()
            .contains("\"total\""));
        let r = rpc(s, json!({"jsonrpc":"2.0","id":5,"method":"resources/read","params":{"uri":"agent-mem://nope"}})).await;
        assert_eq!(r["error"]["code"], ERR_RESOURCE_NOT_FOUND);
    }
}

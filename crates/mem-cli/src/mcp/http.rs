//! MCP Streamable HTTP endpoint (`/mcp`). One JSON response per POST; no SSE in v1.

use std::sync::Arc;

use axum::body::Bytes;
use axum::extract::State;
use axum::http::{header, HeaderMap, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::Value;

use crate::mcp::proto::{handle_payload, rpc_err, ERR_PARSE};
use crate::server::rest::{agent_from_headers, AppState};

fn is_initialize(v: &Value) -> bool {
    v.get("method").and_then(Value::as_str) == Some("initialize")
}

pub async fn post(State(st): State<Arc<AppState>>, headers: HeaderMap, body: Bytes) -> Response {
    let payload: Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(rpc_err(
                    &Value::Null,
                    ERR_PARSE,
                    &format!("parse error: {e}"),
                )),
            )
                .into_response()
        }
    };
    let agent = agent_from_headers(&headers).unwrap_or_else(|| "unknown".to_string());
    let init = is_initialize(&payload);
    let store = st.store();
    match handle_payload(store.as_ref(), payload, &agent).await {
        None => StatusCode::ACCEPTED.into_response(),
        Some(resp) => {
            let mut r = Json(resp).into_response();
            if init {
                let sid = uuid::Uuid::new_v4().to_string();
                if let Ok(v) = HeaderValue::from_str(&sid) {
                    r.headers_mut().insert("mcp-session-id", v);
                }
            }
            r
        }
    }
}

/// Server-initiated SSE streams are not offered in v1.
pub async fn method_not_allowed() -> Response {
    let mut r = StatusCode::METHOD_NOT_ALLOWED.into_response();
    r.headers_mut()
        .insert(header::ALLOW, HeaderValue::from_static("POST, DELETE"));
    r
}

/// Sessions are stateless; terminating one is always fine.
pub async fn delete() -> Response {
    StatusCode::OK.into_response()
}

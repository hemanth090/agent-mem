//! HTTP surface of the daemon: REST/JSON endpoints plus the MCP Streamable HTTP endpoint.

use std::sync::Arc;
use std::time::Duration;

use axum::extract::{DefaultBodyLimit, Request, State};
use axum::http::{header, HeaderMap, HeaderValue, StatusCode};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use mem_core::{Error, ForgetRequest, RecallRequest, RememberRequest};
use mem_store::Store;
use parking_lot::{Mutex, RwLock};
use serde_json::{json, Value};

pub const BODY_LIMIT: usize = 64 * 1024;
pub const REQUEST_TIMEOUT: Duration = Duration::from_secs(10);
pub const AGENT_HEADER: &str = "x-agent-source";

pub struct AppState {
    pub token: String,
    pub store: RwLock<Option<Arc<Store>>>,
    pub load_error: Mutex<Option<String>>,
}

impl AppState {
    pub fn new(token: String) -> Arc<Self> {
        Arc::new(Self {
            token,
            store: RwLock::new(None),
            load_error: Mutex::new(None),
        })
    }

    pub fn store(&self) -> Option<Arc<Store>> {
        self.store.read().clone()
    }

    fn require_store(&self) -> Result<Arc<Store>, ApiError> {
        if let Some(s) = self.store() {
            return Ok(s);
        }
        let msg = self
            .load_error
            .lock()
            .clone()
            .unwrap_or_else(|| "still loading the embedding model; retry shortly".to_string());
        Err(ApiError {
            status: StatusCode::SERVICE_UNAVAILABLE,
            code: "loading",
            message: msg,
        })
    }
}

pub struct ApiError {
    status: StatusCode,
    code: &'static str,
    message: String,
}

impl From<Error> for ApiError {
    fn from(e: Error) -> Self {
        let status = match &e {
            Error::Invalid(_) => StatusCode::BAD_REQUEST,
            Error::Rejected(_) => StatusCode::UNPROCESSABLE_ENTITY,
            Error::NotFound(_) => StatusCode::NOT_FOUND,
            Error::Embed(_) | Error::Storage(_) => StatusCode::SERVICE_UNAVAILABLE,
            Error::Index(_) | Error::Io(_) | Error::Internal(_) => {
                StatusCode::INTERNAL_SERVER_ERROR
            }
        };
        ApiError {
            status,
            code: e.code(),
            message: e.to_string(),
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (
            self.status,
            Json(json!({"error": {"code": self.code, "message": self.message}})),
        )
            .into_response()
    }
}

pub fn router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/v1/remember", post(remember))
        .route("/v1/recall", post(recall))
        .route("/v1/forget", post(forget))
        .route("/v1/stats", get(stats))
        .route(
            "/mcp",
            post(crate::mcp::http::post)
                .get(crate::mcp::http::method_not_allowed)
                .delete(crate::mcp::http::delete),
        )
        .layer(DefaultBodyLimit::max(BODY_LIMIT))
        .layer(middleware::from_fn_with_state(state.clone(), guard))
        .with_state(state)
}

// ------------------------------------------------------------------ guard

/// Strips the port from a `Host`-style authority.
fn host_only(authority: &str) -> &str {
    if let Some(rest) = authority.strip_prefix('[') {
        return match rest.find(']') {
            Some(i) => &authority[..i + 2],
            None => authority,
        };
    }
    match authority.rsplit_once(':') {
        Some((h, p)) if p.chars().all(|c| c.is_ascii_digit()) => h,
        _ => authority,
    }
}

fn is_loopback_host(h: &str) -> bool {
    let h = h.trim().to_ascii_lowercase();
    if h == "localhost" || h == "[::1]" {
        return true;
    }
    h.parse::<std::net::Ipv4Addr>()
        .is_ok_and(|ip| ip.is_loopback())
}

fn origin_is_loopback(origin: &str) -> bool {
    let rest = match origin.split_once("://") {
        Some((_, r)) => r,
        None => return false, // includes the literal "null"
    };
    let authority = rest.split('/').next().unwrap_or("");
    is_loopback_host(host_only(authority))
}

fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

fn forbidden(msg: &str) -> Response {
    (
        StatusCode::FORBIDDEN,
        Json(json!({"error": {"code": "forbidden", "message": msg}})),
    )
        .into_response()
}

fn bearer_ok(headers: &HeaderMap, token: &str) -> bool {
    headers
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
        .is_some_and(|t| constant_time_eq(t.trim().as_bytes(), token.as_bytes()))
}

async fn guard(State(st): State<Arc<AppState>>, req: Request, next: Next) -> Response {
    let headers = req.headers();
    if let Some(host) = headers.get(header::HOST).and_then(|v| v.to_str().ok()) {
        if !is_loopback_host(host_only(host)) {
            return forbidden("non-loopback Host header");
        }
    }
    if let Some(origin) = headers.get(header::ORIGIN) {
        let ok = origin.to_str().ok().is_some_and(origin_is_loopback);
        if !ok {
            return forbidden("non-loopback Origin");
        }
    }
    if req.uri().path() != "/health" && !bearer_ok(headers, &st.token) {
        let mut r = (
            StatusCode::UNAUTHORIZED,
            Json(json!({"error": {"code": "unauthorized", "message": "missing or invalid bearer token"}})),
        )
            .into_response();
        r.headers_mut()
            .insert(header::WWW_AUTHENTICATE, HeaderValue::from_static("Bearer"));
        return r;
    }
    match tokio::time::timeout(REQUEST_TIMEOUT, next.run(req)).await {
        Ok(r) => r,
        Err(_) => (
            StatusCode::GATEWAY_TIMEOUT,
            Json(json!({"error": {"code": "timeout", "message": "request timed out"}})),
        )
            .into_response(),
    }
}

// --------------------------------------------------------------- handlers

async fn health(State(st): State<Arc<AppState>>) -> Response {
    let version = env!("CARGO_PKG_VERSION");
    if st.store().is_some() {
        return Json(json!({
            "ok": true, "service": "agent-mem", "version": version,
            "state": "ready", "model_ready": true
        }))
        .into_response();
    }
    let (state, err) = match st.load_error.lock().clone() {
        Some(e) => ("failed", Some(e)),
        None => ("loading", None),
    };
    (
        StatusCode::SERVICE_UNAVAILABLE,
        Json(json!({
            "ok": false, "service": "agent-mem", "version": version,
            "state": state, "model_ready": false, "error": err
        })),
    )
        .into_response()
}

pub fn agent_from_headers(h: &HeaderMap) -> Option<String> {
    h.get(AGENT_HEADER)
        .and_then(|v| v.to_str().ok())
        .map(str::to_string)
}

async fn remember(
    State(st): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(mut req): Json<RememberRequest>,
) -> Result<Json<Value>, ApiError> {
    let store = st.require_store()?;
    if req.agent_source.is_none() {
        req.agent_source = agent_from_headers(&headers);
    }
    let r = store.remember(req).await?;
    Ok(Json(json!(r)))
}

async fn recall(
    State(st): State<Arc<AppState>>,
    Json(req): Json<RecallRequest>,
) -> Result<Json<Value>, ApiError> {
    let store = st.require_store()?;
    Ok(Json(json!(store.recall(req).await?)))
}

async fn forget(
    State(st): State<Arc<AppState>>,
    Json(req): Json<ForgetRequest>,
) -> Result<Json<Value>, ApiError> {
    let store = st.require_store()?;
    store.forget(req.memory_id.clone()).await?;
    Ok(Json(json!({"forgotten": req.memory_id})))
}

async fn stats(State(st): State<Arc<AppState>>) -> Result<Json<Value>, ApiError> {
    let store = st.require_store()?;
    Ok(Json(json!(store.stats().await?)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::{to_bytes, Body};
    use axum::http::Request as HttpRequest;
    use mem_embed::FakeEmbedder;
    use mem_store::StoreOptions;
    use tower::ServiceExt;

    const TOKEN: &str = "test-token-0123456789";

    fn app(with_store: bool) -> (tempfile::TempDir, Router, Arc<AppState>) {
        let d = tempfile::tempdir().unwrap();
        let st = AppState::new(TOKEN.to_string());
        if with_store {
            let s =
                Store::open(StoreOptions::new(d.path()), Arc::new(FakeEmbedder::new())).unwrap();
            *st.store.write() = Some(s);
        }
        (d, router(st.clone()), st)
    }

    fn req(method: &str, uri: &str, auth: bool, body: Option<Value>) -> HttpRequest<Body> {
        let mut b = HttpRequest::builder()
            .method(method)
            .uri(uri)
            .header("host", "127.0.0.1:4321");
        if auth {
            b = b.header("authorization", format!("Bearer {TOKEN}"));
        }
        match body {
            Some(v) => b
                .header("content-type", "application/json")
                .body(Body::from(v.to_string()))
                .unwrap(),
            None => b.body(Body::empty()).unwrap(),
        }
    }

    async fn json_of(r: Response) -> (StatusCode, Value) {
        let status = r.status();
        let bytes = to_bytes(r.into_body(), 1 << 20).await.unwrap();
        (
            status,
            serde_json::from_slice(&bytes).unwrap_or(Value::Null),
        )
    }

    #[test]
    fn host_parsing() {
        assert_eq!(host_only("127.0.0.1:4321"), "127.0.0.1");
        assert_eq!(host_only("localhost"), "localhost");
        assert_eq!(host_only("[::1]:4321"), "[::1]");
        assert!(
            is_loopback_host("127.0.0.1")
                && is_loopback_host("LOCALHOST")
                && is_loopback_host("[::1]")
        );
        assert!(!is_loopback_host("evil.com") && !is_loopback_host("10.0.0.1"));
        assert!(origin_is_loopback("http://localhost:3000"));
        assert!(!origin_is_loopback("https://evil.com"));
        assert!(!origin_is_loopback("null"));
    }

    #[tokio::test]
    async fn health_needs_no_token_and_reports_loading() {
        let (_d, app, _s) = app(false);
        let (code, v) = json_of(
            app.oneshot(req("GET", "/health", false, None))
                .await
                .unwrap(),
        )
        .await;
        assert_eq!(code, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(v["state"], "loading");
        assert_eq!(v["service"], "agent-mem");
        let (_d, app, _s) = app_ready();
        let (code, v) = json_of(
            app.oneshot(req("GET", "/health", false, None))
                .await
                .unwrap(),
        )
        .await;
        assert_eq!(code, StatusCode::OK);
        assert_eq!(v["ok"], true);
    }

    fn app_ready() -> (tempfile::TempDir, Router, Arc<AppState>) {
        app(true)
    }

    #[tokio::test]
    async fn token_required_for_everything_but_health() {
        let (_d, app, _s) = app_ready();
        for (m, u) in [
            ("GET", "/v1/stats"),
            ("POST", "/mcp"),
            ("POST", "/v1/recall"),
        ] {
            let r = app
                .clone()
                .oneshot(req(m, u, false, Some(json!({}))))
                .await
                .unwrap();
            assert_eq!(r.status(), StatusCode::UNAUTHORIZED, "{u}");
        }
        let mut r = req("GET", "/v1/stats", false, None);
        r.headers_mut()
            .insert("authorization", HeaderValue::from_static("Bearer wrong"));
        assert_eq!(
            app.clone().oneshot(r).await.unwrap().status(),
            StatusCode::UNAUTHORIZED
        );
        let r = app
            .oneshot(req("GET", "/v1/stats", true, None))
            .await
            .unwrap();
        assert_eq!(r.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn rejects_foreign_host_and_origin() {
        let (_d, app, _s) = app_ready();
        let mut r = req("GET", "/health", false, None);
        r.headers_mut()
            .insert("host", HeaderValue::from_static("evil.example:4321"));
        assert_eq!(
            app.clone().oneshot(r).await.unwrap().status(),
            StatusCode::FORBIDDEN
        );
        let mut r = req("GET", "/health", false, None);
        r.headers_mut()
            .insert("origin", HeaderValue::from_static("https://evil.example"));
        assert_eq!(
            app.clone().oneshot(r).await.unwrap().status(),
            StatusCode::FORBIDDEN
        );
        let mut r = req("GET", "/health", false, None);
        r.headers_mut()
            .insert("origin", HeaderValue::from_static("http://localhost:5173"));
        assert_eq!(app.oneshot(r).await.unwrap().status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn body_limit_enforced() {
        let (_d, app, _s) = app_ready();
        let big = json!({"content": "x".repeat(BODY_LIMIT + 10)});
        let r = app
            .oneshot(req("POST", "/v1/remember", true, Some(big)))
            .await
            .unwrap();
        assert_eq!(r.status(), StatusCode::PAYLOAD_TOO_LARGE);
    }

    #[tokio::test]
    async fn rest_roundtrip_and_error_mapping() {
        let (_d, app, _s) = app_ready();
        let (c, v) = json_of(
            app.clone()
                .oneshot(req(
                    "POST",
                    "/v1/remember",
                    true,
                    Some(json!({"content":"rest api fact","project_id":"p"})),
                ))
                .await
                .unwrap(),
        )
        .await;
        assert_eq!(c, StatusCode::OK);
        let id = v["id"].as_str().unwrap().to_string();
        let (c, v) = json_of(
            app.clone()
                .oneshot(req(
                    "POST",
                    "/v1/recall",
                    true,
                    Some(json!({"query":"rest api","project_id":"p"})),
                ))
                .await
                .unwrap(),
        )
        .await;
        assert_eq!(c, StatusCode::OK);
        assert_eq!(v["hits"][0]["id"], id);
        let (c, v) = json_of(
            app.clone()
                .oneshot(req(
                    "POST",
                    "/v1/remember",
                    true,
                    Some(json!({"content":"  "})),
                ))
                .await
                .unwrap(),
        )
        .await;
        assert_eq!(
            (c, v["error"]["code"].as_str()),
            (StatusCode::BAD_REQUEST, Some("invalid"))
        );
        let (c, _) = json_of(
            app.clone()
                .oneshot(req(
                    "POST",
                    "/v1/remember",
                    true,
                    Some(json!({"content":"-----BEGIN PRIVATE KEY-----"})),
                ))
                .await
                .unwrap(),
        )
        .await;
        assert_eq!(c, StatusCode::UNPROCESSABLE_ENTITY);
        let (c, _) = json_of(
            app.clone()
                .oneshot(req(
                    "POST",
                    "/v1/forget",
                    true,
                    Some(json!({"memory_id":"nope"})),
                ))
                .await
                .unwrap(),
        )
        .await;
        assert_eq!(c, StatusCode::NOT_FOUND);
        let (c, v) = json_of(
            app.oneshot(req("GET", "/v1/stats", true, None))
                .await
                .unwrap(),
        )
        .await;
        assert_eq!((c, v["total"].as_u64()), (StatusCode::OK, Some(1)));
    }

    #[tokio::test]
    async fn loading_store_returns_503() {
        let (_d, app, _s) = app(false);
        let r = app
            .oneshot(req("GET", "/v1/stats", true, None))
            .await
            .unwrap();
        assert_eq!(r.status(), StatusCode::SERVICE_UNAVAILABLE);
    }

    #[tokio::test]
    async fn mcp_http_endpoint() {
        let (_d, app, _s) = app_ready();
        let r = app
            .clone()
            .oneshot(req(
                "POST",
                "/mcp",
                true,
                Some(json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{}})),
            ))
            .await
            .unwrap();
        assert_eq!(r.status(), StatusCode::OK);
        assert!(r.headers().contains_key("mcp-session-id"));
        let (_, v) = json_of(r).await;
        assert_eq!(v["result"]["serverInfo"]["name"], "agent-mem");

        let r = app
            .clone()
            .oneshot(req(
                "POST",
                "/mcp",
                true,
                Some(json!({"jsonrpc":"2.0","method":"notifications/initialized"})),
            ))
            .await
            .unwrap();
        assert_eq!(r.status(), StatusCode::ACCEPTED);

        let mut bad = req("POST", "/mcp", true, None);
        *bad.body_mut() = Body::from("{not json");
        let (c, v) = json_of(app.clone().oneshot(bad).await.unwrap()).await;
        assert_eq!(c, StatusCode::BAD_REQUEST);
        assert_eq!(v["error"]["code"], -32700);

        assert_eq!(
            app.clone()
                .oneshot(req("GET", "/mcp", true, None))
                .await
                .unwrap()
                .status(),
            StatusCode::METHOD_NOT_ALLOWED
        );
        assert_eq!(
            app.oneshot(req("DELETE", "/mcp", true, None))
                .await
                .unwrap()
                .status(),
            StatusCode::OK
        );
    }
}

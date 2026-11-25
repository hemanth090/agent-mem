//! The long-lived daemon: holds the warm ONNX model, HNSW index and SQLite handles.

use std::io::IsTerminal;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{anyhow, Context, Result};
use mem_core::Config;
use mem_store::{Store, StoreOptions};
use tokio::net::TcpListener;

use crate::lifecycle::client::{health, http_client, Health};
use crate::lifecycle::spawn::Starter;
use crate::server::rest::{router, AppState};

/// Reads the bearer token, creating it (mode 0600) on first use.
pub fn ensure_token(cfg: &Config) -> Result<String> {
    let path = cfg.token_path();
    if let Ok(t) = std::fs::read_to_string(&path) {
        let t = t.trim().to_string();
        if t.len() >= 32 {
            return Ok(t);
        }
    }
    let token = format!(
        "{}{}",
        uuid::Uuid::new_v4().simple(),
        uuid::Uuid::new_v4().simple()
    );
    write_private(&path, &token).with_context(|| format!("writing {}", path.display()))?;
    Ok(token)
}

fn write_private(path: &Path, content: &str) -> std::io::Result<()> {
    use std::io::Write;
    let mut opts = std::fs::OpenOptions::new();
    opts.write(true).create(true).truncate(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        opts.mode(0o600);
    }
    let mut f = opts.open(path)?;
    f.write_all(content.as_bytes())?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))?;
    }
    Ok(())
}

fn create_private_dir(path: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(path)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o700))?;
    }
    Ok(())
}

async fn shutdown_signal(mut stop_rx: tokio::sync::mpsc::Receiver<()>) {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{signal, SignalKind};
        let mut term = match signal(SignalKind::terminate()) {
            Ok(s) => s,
            Err(_) => {
                tokio::select! {
                    _ = tokio::signal::ctrl_c() => {}
                    _ = stop_rx.recv() => {}
                }
                return;
            }
        };
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {}
            _ = term.recv() => {}
            _ = stop_rx.recv() => {}
        }
    }
    #[cfg(not(unix))]
    {
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {}
            _ = stop_rx.recv() => {}
        }
    }
}

/// `agent-mem daemon` without `--foreground`: start a detached daemon and wait until ready.
pub async fn start_background(cfg: Config) -> Result<()> {
    let http = http_client(Duration::from_secs(2));
    if health(&http, &cfg.addr).await == Health::Ready {
        println!("agent-mem daemon already running on {}", cfg.addr);
        return Ok(());
    }
    let starter = Starter::new(cfg.clone());
    eprintln!(
        "starting agent-mem daemon on {} (first run downloads the embedding model)...",
        cfg.addr
    );
    starter
        .ensure_ready(Duration::from_secs(300))
        .await
        .map_err(|e| anyhow!(e))?;
    println!(
        "agent-mem daemon ready on {} (log: {})",
        cfg.addr,
        cfg.log_path().display()
    );
    Ok(())
}

pub async fn run_foreground(cfg: Config) -> Result<()> {
    create_private_dir(&cfg.data_dir)?;
    let addr = cfg.socket_addr().map_err(|e| anyhow!(e))?;

    // Bind first: this is the single-instance guard and lets /health answer "loading".
    let listener = match TcpListener::bind(addr).await {
        Ok(l) => l,
        Err(e) if e.kind() == std::io::ErrorKind::AddrInUse => {
            let http = http_client(Duration::from_secs(2));
            return match health(&http, &cfg.addr).await {
                Health::Ready | Health::Loading => {
                    tracing::info!("agent-mem daemon already running on {addr}");
                    Ok(())
                }
                _ => Err(anyhow!(
                    "address {addr} is already in use by another program"
                )),
            };
        }
        Err(e) => return Err(anyhow!("cannot bind {addr}: {e}")),
    };

    let token = ensure_token(&cfg)?;
    let pid = std::process::id();
    std::fs::write(cfg.pid_path(), pid.to_string())?;
    let state = AppState::new(token);
    let (stop_tx, stop_rx) = tokio::sync::mpsc::channel::<()>(4);

    // Load model + store in the background so the port is answering immediately.
    {
        let state = state.clone();
        let cfg = cfg.clone();
        let stop_tx = stop_tx.clone();
        tokio::spawn(async move {
            let progress = std::io::stderr().is_terminal();
            let cfg2 = cfg.clone();
            let loaded = tokio::task::spawn_blocking(move || -> Result<Arc<Store>> {
                let embedder = crate::embedder::build_embedder(&cfg2, progress, false)
                    .map_err(|e| anyhow!(e))?;
                Store::open(StoreOptions::from_config(&cfg2), embedder).map_err(|e| anyhow!(e))
            })
            .await;
            match loaded {
                Ok(Ok(store)) => {
                    store.spawn_maintenance();
                    *state.store.write() = Some(store);
                    tracing::info!("agent-mem ready");
                }
                Ok(Err(e)) => {
                    tracing::error!("startup failed: {e:#}");
                    *state.load_error.lock() = Some(format!("{e:#}"));
                    let _ = stop_tx.send(()).await;
                }
                Err(e) => {
                    tracing::error!("startup task panicked: {e}");
                    *state.load_error.lock() = Some(e.to_string());
                    let _ = stop_tx.send(()).await;
                }
            }
        });
    }

    tracing::info!(
        "agent-mem {} listening on http://{addr} (data: {})",
        env!("CARGO_PKG_VERSION"),
        cfg.data_dir.display()
    );
    let served = axum::serve(listener, router(state.clone()))
        .with_graceful_shutdown(shutdown_signal(stop_rx))
        .await;

    tracing::info!("shutting down");
    if let Some(store) = state.store() {
        if let Err(e) = store.shutdown().await {
            tracing::warn!("shutdown flush failed: {e}");
        }
    }
    if std::fs::read_to_string(cfg.pid_path())
        .ok()
        .and_then(|s| s.trim().parse::<u32>().ok())
        == Some(pid)
    {
        let _ = std::fs::remove_file(cfg.pid_path());
    }
    served.context("server error")?;
    if let Some(e) = state.load_error.lock().clone() {
        return Err(anyhow!("startup failed: {e}"));
    }
    Ok(())
}

#[cfg(unix)]
fn process_alive(pid: u32) -> bool {
    // SAFETY: signal 0 only checks that the process exists.
    let rc = unsafe { libc::kill(pid as libc::pid_t, 0) };
    rc == 0 || std::io::Error::last_os_error().raw_os_error() != Some(libc::ESRCH)
}

#[cfg(not(unix))]
fn process_alive(_pid: u32) -> bool {
    false
}

/// `agent-mem stop`.
pub async fn stop(cfg: &Config) -> Result<()> {
    let pid: Option<u32> = std::fs::read_to_string(cfg.pid_path())
        .ok()
        .and_then(|s| s.trim().parse().ok());
    let http = http_client(Duration::from_secs(2));
    if health(&http, &cfg.addr).await == Health::Down {
        println!("agent-mem daemon is not running");
        let _ = std::fs::remove_file(cfg.pid_path());
        return Ok(());
    }
    let Some(pid) = pid else {
        return Err(anyhow!(
            "daemon is running but {} is missing; stop it manually",
            cfg.pid_path().display()
        ));
    };
    #[cfg(unix)]
    {
        // SAFETY: plain kill(2) on a pid read from our own pid file.
        let rc = unsafe { libc::kill(pid as libc::pid_t, libc::SIGTERM) };
        if rc != 0 {
            return Err(anyhow!(
                "could not signal pid {pid}: {}",
                std::io::Error::last_os_error()
            ));
        }
    }
    #[cfg(not(unix))]
    {
        return Err(anyhow!("`stop` is only supported on unix; kill pid {pid}"));
    }
    // The port closes before shutdown finishes (access flush, index checkpoint), so wait for
    // the process itself to exit; otherwise a restart could race the old daemon's last writes.
    for _ in 0..100 {
        if health(&http, &cfg.addr).await == Health::Down && !process_alive(pid) {
            println!("agent-mem daemon stopped");
            return Ok(());
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    Err(anyhow!("daemon (pid {pid}) did not stop within 10s"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_created_once_and_private() {
        let d = tempfile::tempdir().unwrap();
        let cfg = Config {
            data_dir: d.path().to_path_buf(),
            ..Config::default()
        };
        let a = ensure_token(&cfg).unwrap();
        assert_eq!(a.len(), 64);
        assert_eq!(a, ensure_token(&cfg).unwrap());
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(cfg.token_path())
                .unwrap()
                .permissions()
                .mode()
                & 0o777;
            assert_eq!(mode, 0o600);
        }
    }
}

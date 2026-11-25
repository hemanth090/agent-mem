//! Detached daemon spawning with a cross-process lock so concurrent proxies start exactly one.

use std::fs::{File, OpenOptions};
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use fs2::FileExt;
use mem_core::Config;

use crate::lifecycle::client::{health, http_client, tail_log, Health};

const LOG_MAX_BYTES: u64 = 5 * 1024 * 1024;
const LOG_KEEP: u32 = 3;

/// `daemon.log` -> `daemon.log.1` -> `.2` -> `.3` once it exceeds 5 MB.
pub fn rotate_log(path: &Path) {
    let too_big = std::fs::metadata(path)
        .map(|m| m.len() > LOG_MAX_BYTES)
        .unwrap_or(false);
    if !too_big {
        return;
    }
    let numbered = |n: u32| {
        let mut s = path.as_os_str().to_os_string();
        s.push(format!(".{n}"));
        std::path::PathBuf::from(s)
    };
    let _ = std::fs::remove_file(numbered(LOG_KEEP));
    for n in (1..LOG_KEEP).rev() {
        let _ = std::fs::rename(numbered(n), numbered(n + 1));
    }
    let _ = std::fs::rename(path, numbered(1));
}

pub fn spawn_daemon(cfg: &Config) -> std::io::Result<Child> {
    std::fs::create_dir_all(&cfg.data_dir)?;
    let log_path = cfg.log_path();
    rotate_log(&log_path);
    let log = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)?;
    let exe = std::env::current_exe()?;
    let mut cmd = Command::new(exe);
    cmd.arg("daemon")
        .arg("--foreground")
        .arg("--data-dir")
        .arg(&cfg.data_dir)
        .arg("--addr")
        .arg(&cfg.addr);
    if let Some(m) = &cfg.model_dir {
        cmd.arg("--model-dir").arg(m);
    }
    cmd.stdin(Stdio::null())
        .stdout(Stdio::from(log.try_clone()?))
        .stderr(Stdio::from(log));

    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        // New session: the daemon must outlive the agent that spawned it.
        unsafe {
            cmd.pre_exec(|| {
                libc::setsid();
                Ok(())
            });
        }
    }
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const DETACHED_PROCESS: u32 = 0x0000_0008;
        const CREATE_NEW_PROCESS_GROUP: u32 = 0x0000_0200;
        cmd.creation_flags(DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP);
    }
    cmd.spawn()
}

/// Ensures a ready daemon exists, starting one if necessary.
pub struct Starter {
    cfg: Config,
    http: reqwest::Client,
    gate: tokio::sync::Mutex<()>,
}

impl Starter {
    pub fn new(cfg: Config) -> Self {
        Self {
            cfg,
            http: http_client(Duration::from_secs(2)),
            gate: tokio::sync::Mutex::new(()),
        }
    }

    pub async fn health(&self) -> Health {
        health(&self.http, &self.cfg.addr).await
    }

    async fn acquire_lock(&self, deadline: Instant) -> Result<File, String> {
        std::fs::create_dir_all(&self.cfg.data_dir).map_err(|e| e.to_string())?;
        let f = OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(self.cfg.lock_path())
            .map_err(|e| format!("cannot open lock file: {e}"))?;
        loop {
            if f.try_lock_exclusive().is_ok() {
                return Ok(f);
            }
            if Instant::now() >= deadline {
                return Err("timed out waiting for the daemon start lock".into());
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    }

    /// Takes the lock, re-checks health, and spawns only if the daemon is still down.
    async fn spawn_if_down(&self, deadline: Instant) -> Result<Option<Child>, String> {
        let lock = self.acquire_lock(deadline).await?;
        if self.health().await != Health::Down {
            return Ok(None); // another proxy won the race
        }
        let mut child = spawn_daemon(&self.cfg).map_err(|e| format!("cannot spawn daemon: {e}"))?;
        // Hold the lock until the port is bound (or the child died), so racers see it up.
        let bound_by = Instant::now() + Duration::from_secs(10);
        loop {
            if self.health().await != Health::Down {
                break;
            }
            if let Ok(Some(status)) = child.try_wait() {
                if !status.success() {
                    return Err(format!(
                        "daemon exited immediately ({status}); log:\n{}",
                        tail_log(&self.cfg, 15)
                    ));
                }
                break;
            }
            if Instant::now() >= bound_by {
                break;
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
        drop(lock);
        Ok(Some(child))
    }

    pub async fn ensure_ready(&self, within: Duration) -> Result<(), String> {
        let deadline = Instant::now() + within;
        let _g = tokio::time::timeout(within, self.gate.lock())
            .await
            .map_err(|_| "timed out waiting for the daemon".to_string())?;
        let mut child: Option<Child> = None;
        let mut spawned = false;
        loop {
            match self.health().await {
                Health::Ready => return Ok(()),
                Health::Failed(e) => {
                    return Err(format!("daemon failed to start: {e}\n{}", tail_log(&self.cfg, 15)))
                }
                Health::Foreign => {
                    return Err(format!(
                        "address {} is in use by something that is not agent-mem; set --addr / AGENT_MEM_ADDR",
                        self.cfg.addr
                    ))
                }
                Health::Loading => {}
                Health::Down => {
                    if !spawned {
                        spawned = true;
                        child = self.spawn_if_down(deadline).await?;
                    } else if let Some(c) = child.as_mut() {
                        if let Ok(Some(status)) = c.try_wait() {
                            if !status.success() {
                                return Err(format!(
                                    "daemon exited ({status}); log:\n{}",
                                    tail_log(&self.cfg, 15)
                                ));
                            }
                            // Clean exit but still down: a lost race whose winner died. Try again.
                            spawned = false;
                            child = None;
                        }
                    }
                }
            }
            if Instant::now() >= deadline {
                return Err(format!(
                    "daemon not ready after {}s (it may still be downloading the embedding model; see {})",
                    within.as_secs(),
                    self.cfg.log_path().display()
                ));
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rotation_shifts_files() {
        let d = tempfile::tempdir().unwrap();
        let p = d.path().join("daemon.log");
        std::fs::write(&p, vec![b'x'; (LOG_MAX_BYTES + 1) as usize]).unwrap();
        std::fs::write(d.path().join("daemon.log.1"), "old1").unwrap();
        rotate_log(&p);
        assert!(!p.exists());
        assert_eq!(
            std::fs::metadata(d.path().join("daemon.log.1"))
                .unwrap()
                .len(),
            LOG_MAX_BYTES + 1
        );
        assert_eq!(
            std::fs::read_to_string(d.path().join("daemon.log.2")).unwrap(),
            "old1"
        );
    }

    #[test]
    fn small_log_not_rotated() {
        let d = tempfile::tempdir().unwrap();
        let p = d.path().join("daemon.log");
        std::fs::write(&p, "small").unwrap();
        rotate_log(&p);
        assert!(p.exists());
    }
}

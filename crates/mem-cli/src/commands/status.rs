//! `agent-mem status`.

use std::time::Duration;

use anyhow::{anyhow, bail, Result};
use mem_core::Config;

use crate::lifecycle::client::{health, http_client, read_token, Health};

pub async fn status(cfg: &Config) -> Result<()> {
    let http = http_client(Duration::from_secs(5));
    match health(&http, &cfg.addr).await {
        Health::Down => {
            println!("agent-mem daemon: not running ({})", cfg.addr);
            Ok(())
        }
        Health::Loading => {
            println!("agent-mem daemon: starting (loading model) on {}", cfg.addr);
            Ok(())
        }
        Health::Failed(e) => bail!("agent-mem daemon failed to start: {e}"),
        Health::Foreign => bail!("{} is in use by another program", cfg.addr),
        Health::Ready => {
            let token = read_token(cfg)
                .ok_or_else(|| anyhow!("cannot read {}", cfg.token_path().display()))?;
            let stats: serde_json::Value = http
                .get(format!("http://{}/v1/stats", cfg.addr))
                .bearer_auth(token)
                .send()
                .await?
                .error_for_status()?
                .json()
                .await?;
            println!("agent-mem daemon: running on {}", cfg.addr);
            println!("{}", serde_json::to_string_pretty(&stats)?);
            Ok(())
        }
    }
}

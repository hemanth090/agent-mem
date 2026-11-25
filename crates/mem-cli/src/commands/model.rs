//! Commands that deal with the embedding model: `fetch-model` and `reindex`.

use std::io::IsTerminal;
use std::time::Duration;

use anyhow::{anyhow, bail, Result};
use mem_core::Config;
use mem_store::{Store, StoreOptions};

use crate::embedder::build_embedder;
use crate::lifecycle::client::{health, http_client, Health};

/// Downloads (if needed) and warms the embedding model.
pub fn fetch_model(cfg: &Config) -> Result<()> {
    let engine =
        build_embedder(cfg, std::io::stderr().is_terminal(), false).map_err(|e| anyhow!(e))?;
    engine.embed_query("warm-up").map_err(|e| anyhow!(e))?;
    println!("model ready: {}", engine.model_name());
    Ok(())
}

/// Re-embeds every memory with the current model. The daemon must be stopped.
pub async fn reindex(cfg: Config) -> Result<()> {
    let http = http_client(Duration::from_secs(2));
    if health(&http, &cfg.addr).await != Health::Down {
        bail!(
            "the daemon is running on {}; run `agent-mem stop` first",
            cfg.addr
        );
    }
    let embedder = build_embedder(&cfg, true, false).map_err(|e| anyhow!(e))?;
    let n = tokio::task::spawn_blocking(move || {
        Store::reindex(&StoreOptions::from_config(&cfg), embedder.as_ref())
    })
    .await??;
    println!("re-embedded {n} memories; the index will be rebuilt on next start");
    Ok(())
}

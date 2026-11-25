//! The `agent-mem install` command: stable binary copy, client registration, model prefetch.

use std::time::Duration;

use anyhow::{anyhow, bail, Context, Result};
use mem_core::Config;

use super::{
    describe, install_binary, run, stable_exe, unix_ts, write_instructions, ClientKind, CopyStatus,
    Options,
};
use crate::cli::InstallArgs;
use crate::commands::model::fetch_model;
use crate::lifecycle::client::{health, http_client, Health};
use crate::server::daemon;
use crate::template;

pub async fn execute(cfg: &Config, args: InstallArgs) -> Result<()> {
    let InstallArgs {
        clients,
        dry_run,
        uninstall,
        write_instructions: instructions_path,
        instructions_only,
        no_copy,
        no_model,
        print_template,
    } = args;

    if print_template {
        print!("{}", template::block());
        return Ok(());
    }
    let home = dirs::home_dir().ok_or_else(|| anyhow!("cannot determine the home directory"))?;
    let current = std::env::current_exe()
        .and_then(std::fs::canonicalize)
        .context("cannot determine the agent-mem executable path")?;
    let force = !clients.is_empty();
    let kinds = if instructions_only {
        Vec::new()
    } else if clients.is_empty() {
        ClientKind::ALL.to_vec()
    } else {
        clients
    };
    let opts = Options {
        dry_run,
        uninstall,
        force,
        timestamp: unix_ts(),
    };
    let mut failed = false;

    // Register a stable copy of the binary, never a build or download directory.
    let mut exe = current.clone();
    if !kinds.is_empty() && !uninstall && !no_copy {
        let dst = stable_exe(&cfg.data_dir);
        match install_binary(&current, &dst, dry_run) {
            Ok(CopyStatus::Same) => exe = dst,
            Ok(CopyStatus::WouldCopy) => {
                println!("+ binary       would copy to {}", dst.display());
                exe = dst;
            }
            Ok(CopyStatus::Copied) => {
                println!("+ binary       installed at {}", dst.display());
                // A running daemon still executes the old binary: stop it so the next use restarts it.
                let http = http_client(Duration::from_secs(2));
                if health(&http, &cfg.addr).await != Health::Down {
                    let _ = daemon::stop(cfg).await;
                }
                exe = dst;
            }
            Err(e) => {
                println!("✗ binary: {e:#}");
                failed = true;
            }
        }
    }

    let outcomes = if failed {
        Vec::new()
    } else {
        run(&home, &exe, &kinds, &opts)
    };
    for o in &outcomes {
        println!("{}", describe(o));
        failed |= o.status.is_failure();
    }
    if let Some(path) = instructions_path {
        match write_instructions(&path, dry_run, opts.timestamp) {
            Ok(msg) => println!("{msg}"),
            Err(e) => {
                println!("✗ instructions: {e:#}");
                failed = true;
            }
        }
    }
    let configured = outcomes
        .iter()
        .any(|o| !o.status.is_failure() && o.path.is_some());
    if !kinds.is_empty() && !uninstall && !outcomes.is_empty() && !configured && !failed {
        println!(
            "\nNo supported agent was detected. Name one explicitly to configure it anyway, e.g. `--clients claude,opencode,codex,antigravity`."
        );
    }
    if configured && !uninstall && !dry_run && !failed {
        if !no_model {
            // Download the model now so the first tool call is instant.
            eprintln!("preparing the embedding model (one-time download, ~35 MB)...");
            if let Err(e) = fetch_model(cfg) {
                println!("! model not downloaded yet ({e:#}); it will be fetched on first use, or run `agent-mem fetch-model`");
            }
        }
        println!(
            "\nDone. Restart your agent(s); agent-mem will then be available in every project."
        );
    }
    if failed {
        bail!("some steps failed (existing config files were left untouched)");
    }
    Ok(())
}

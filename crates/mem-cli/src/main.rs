#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]
//! agent-mem: a single-binary, local, shared memory engine for coding agents.
//!
//! Module map:
//! - `cli`        argument parsing
//! - `server`     the daemon and its HTTP surface (REST + MCP Streamable HTTP)
//! - `mcp`        MCP protocol logic, HTTP endpoint and the stdio proxy
//! - `lifecycle`  client-side daemon health probes and detached spawning
//! - `commands`   one-shot commands: install, status, model management, benchmark

mod cli;
mod commands;
mod embedder;
mod lifecycle;
mod mcp;
mod server;
mod template;

use std::process::ExitCode;

use anyhow::{anyhow, Result};
use clap::Parser;
use cli::{Cli, Cmd};
use mem_core::Config;

fn init_tracing(default_level: &str) {
    let filter = tracing_subscriber::EnvFilter::try_from_env("AGENT_MEM_LOG")
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(default_level));
    // Logs always go to stderr: stdout is reserved for MCP protocol messages.
    let _ = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(std::io::stderr)
        .with_ansi(false)
        .try_init();
}

fn load_config(cli: &Cli) -> Result<Config> {
    let mut cfg = Config::load(cli.data_dir.as_deref()).map_err(|e| anyhow!(e))?;
    if let Some(a) = &cli.addr {
        cfg.addr = a.clone();
    }
    if let Some(m) = &cli.model_dir {
        cfg.model_dir = Some(m.clone());
    }
    cfg.validate().map_err(|e| anyhow!(e))?;
    Ok(cfg)
}

async fn run(cli: Cli) -> Result<()> {
    let cfg = load_config(&cli)?;
    match cli.cmd {
        Cmd::Daemon { foreground } => {
            init_tracing("info");
            if foreground {
                server::daemon::run_foreground(cfg).await
            } else {
                server::daemon::start_background(cfg).await
            }
        }
        Cmd::Mcp => {
            init_tracing("warn");
            mcp::stdio::run(cfg).await
        }
        Cmd::Install(args) => commands::install::command::execute(&cfg, args).await,
        Cmd::FetchModel => {
            init_tracing("warn");
            commands::model::fetch_model(&cfg)
        }
        Cmd::Reindex => {
            init_tracing("warn");
            commands::model::reindex(cfg).await
        }
        Cmd::Status => commands::status::status(&cfg).await,
        Cmd::Stop => server::daemon::stop(&cfg).await,
        Cmd::Template { .. } => {
            print!("{}", template::block());
            Ok(())
        }
        Cmd::BenchRecall {
            records,
            queries,
            fake,
        } => {
            init_tracing("warn");
            commands::bench::run(&cfg, records, queries, fake).await
        }
    }
}

#[tokio::main]
async fn main() -> ExitCode {
    let cli = Cli::parse();
    match run(cli).await {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e:#}");
            ExitCode::FAILURE
        }
    }
}

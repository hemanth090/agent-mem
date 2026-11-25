use std::path::PathBuf;

use clap::{Args, Parser, Subcommand, ValueEnum};

use crate::commands::install::ClientKind;

#[derive(Parser, Debug)]
#[command(
    name = "agent-mem",
    version,
    about = "Local, shared, hybrid-search memory for coding agents (Claude Code, OpenCode, Codex CLI, Antigravity)"
)]
pub struct Cli {
    /// Data directory (default: ~/.agent-mem, env AGENT_MEM_DATA_DIR).
    #[arg(long, global = true, value_name = "DIR")]
    pub data_dir: Option<PathBuf>,

    /// Daemon address; must be loopback (default 127.0.0.1:4321, env AGENT_MEM_ADDR).
    #[arg(long, global = true, value_name = "HOST:PORT")]
    pub addr: Option<String>,

    /// Where the embedding model is cached (default: <data-dir>/models, env AGENT_MEM_MODEL_DIR).
    #[arg(long, global = true, value_name = "DIR")]
    pub model_dir: Option<PathBuf>,

    #[command(subcommand)]
    pub cmd: Cmd,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum TemplateFormat {
    Agents,
    Claude,
}

#[derive(Args, Debug)]
pub struct InstallArgs {
    /// Limit to these clients (comma separated). Default: all detected.
    #[arg(long, value_enum, value_delimiter = ',')]
    pub clients: Vec<ClientKind>,
    /// Show what would change without writing anything.
    #[arg(long)]
    pub dry_run: bool,
    /// Remove the agent-mem entry instead of adding it.
    #[arg(long)]
    pub uninstall: bool,
    /// Also add/update the usage instructions block in this AGENTS.md / CLAUDE.md.
    #[arg(long, value_name = "FILE")]
    pub write_instructions: Option<PathBuf>,
    /// With --write-instructions: only write that file, do not touch any client config.
    #[arg(long, requires = "write_instructions")]
    pub instructions_only: bool,
    /// Register the binary where it is now instead of copying it to <data-dir>/bin first.
    #[arg(long)]
    pub no_copy: bool,
    /// Do not pre-download the embedding model during install.
    #[arg(long)]
    pub no_model: bool,
    /// Print the instruction snippet and exit.
    #[arg(long)]
    pub print_template: bool,
}

#[derive(Subcommand, Debug)]
pub enum Cmd {
    /// Start the daemon that holds the warm model and index (detached unless --foreground).
    Daemon {
        /// Run in this process (for launchd/systemd or debugging).
        #[arg(long)]
        foreground: bool,
    },
    /// MCP server over stdio; starts the daemon if needed and proxies to it.
    Mcp,
    /// Register the MCP server in Claude Code, OpenCode, Codex CLI and Antigravity.
    Install(InstallArgs),
    /// Download the embedding model now (useful before going offline).
    FetchModel,
    /// Re-embed all memories (after a model change). The daemon must be stopped.
    Reindex,
    /// Show daemon state and memory statistics.
    Status,
    /// Stop the running daemon.
    Stop,
    /// Print the AGENTS.md / CLAUDE.md instruction snippet.
    Template {
        #[arg(long, value_enum, default_value = "agents")]
        format: TemplateFormat,
    },
    /// Measure retrieval latency on a throwaway database.
    BenchRecall {
        #[arg(long, default_value_t = 2000)]
        records: usize,
        #[arg(long, default_value_t = 200)]
        queries: usize,
        /// Use the hashing fake embedder instead of the real model.
        #[arg(long)]
        fake: bool,
    },
}

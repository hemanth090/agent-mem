#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]
//! Core types, ranking math and configuration for agent-mem.

pub mod config;
pub mod error;
pub mod rank;
pub mod schema;

pub use config::Config;
pub use error::{Error, Result};
pub use schema::*;

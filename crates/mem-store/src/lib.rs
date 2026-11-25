#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]
//! SQLite (WAL + FTS5) + usearch HNSW persistence layer with cosine dedup.

pub mod hnsw;
pub mod secrets;
pub mod sqlite;
pub mod store;

pub use store::{fts_query, now_ms, Store, StoreOptions};

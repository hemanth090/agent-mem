use std::path::Path;
use std::time::Duration;

use mem_core::{Error, MemoryRecord, MemoryType, Result};
use rusqlite::{Connection, OpenFlags, Row};

pub const SCHEMA_VERSION: i64 = 1;

pub fn sq(e: rusqlite::Error) -> Error {
    Error::Storage(e.to_string())
}

const SCHEMA_V1: &str = r#"
CREATE TABLE IF NOT EXISTS meta(
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS memories(
  key           INTEGER PRIMARY KEY AUTOINCREMENT,
  id            TEXT NOT NULL UNIQUE,
  content       TEXT NOT NULL,
  memory_type   TEXT NOT NULL CHECK(memory_type IN ('episodic','semantic','procedural')),
  project_id    TEXT NOT NULL,
  agent_source  TEXT NOT NULL,
  tags          TEXT NOT NULL DEFAULT '[]',
  created_at    INTEGER NOT NULL,
  last_accessed INTEGER NOT NULL,
  access_count  INTEGER NOT NULL DEFAULT 1,
  weight        REAL NOT NULL DEFAULT 1.0,
  embedding     BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_mem_project ON memories(project_id, memory_type);

CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
  content, tags,
  content='memories', content_rowid='key',
  tokenize='porter unicode61 remove_diacritics 2'
);

CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
  INSERT INTO memories_fts(rowid, content, tags) VALUES (new.key, new.content, new.tags);
END;
CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
  INSERT INTO memories_fts(memories_fts, rowid, content, tags)
  VALUES ('delete', old.key, old.content, old.tags);
END;
CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE OF content, tags ON memories BEGIN
  INSERT INTO memories_fts(memories_fts, rowid, content, tags)
  VALUES ('delete', old.key, old.content, old.tags);
  INSERT INTO memories_fts(rowid, content, tags) VALUES (new.key, new.content, new.tags);
END;
"#;

fn apply_pragmas(conn: &Connection) -> Result<()> {
    conn.busy_timeout(Duration::from_millis(5000)).map_err(sq)?;
    conn.execute_batch(
        "PRAGMA synchronous=NORMAL; PRAGMA foreign_keys=ON; PRAGMA temp_store=MEMORY;",
    )
    .map_err(sq)?;
    Ok(())
}

/// Opens the single read-write connection (WAL, busy_timeout=5000) and migrates.
pub fn open_writer(path: &Path) -> Result<Connection> {
    let conn = Connection::open(path).map_err(sq)?;
    conn.busy_timeout(Duration::from_millis(5000)).map_err(sq)?;
    let mode: String = conn
        .query_row("PRAGMA journal_mode=WAL", [], |r| r.get(0))
        .map_err(sq)?;
    if !mode.eq_ignore_ascii_case("wal") {
        return Err(Error::Storage(format!(
            "could not enable WAL (journal_mode={mode})"
        )));
    }
    apply_pragmas(&conn)?;
    migrate(&conn)?;
    Ok(conn)
}

pub fn open_reader(path: &Path) -> Result<Connection> {
    let conn = Connection::open_with_flags(
        path,
        OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )
    .map_err(sq)?;
    apply_pragmas(&conn)?;
    Ok(conn)
}

fn migrate(conn: &Connection) -> Result<()> {
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);",
    )
    .map_err(sq)?;
    let current: i64 = match get_meta(conn, "schema_version")? {
        Some(v) => v.parse().unwrap_or(0),
        None => 0,
    };
    if current > SCHEMA_VERSION {
        return Err(Error::Storage(format!(
            "database schema v{current} is newer than this binary supports (v{SCHEMA_VERSION})"
        )));
    }
    if current < 1 {
        conn.execute_batch("BEGIN IMMEDIATE;").map_err(sq)?;
        let r = conn.execute_batch(SCHEMA_V1).and_then(|_| {
            conn.execute(
                "INSERT OR REPLACE INTO meta(key,value) VALUES('schema_version','1')",
                [],
            )
            .map(|_| ())
        });
        match r {
            Ok(()) => conn.execute_batch("COMMIT;").map_err(sq)?,
            Err(e) => {
                let _ = conn.execute_batch("ROLLBACK;");
                return Err(sq(e));
            }
        }
    }
    Ok(())
}

pub fn get_meta(conn: &Connection, key: &str) -> Result<Option<String>> {
    match conn.query_row("SELECT value FROM meta WHERE key=?1", [key], |r| {
        r.get::<_, String>(0)
    }) {
        Ok(v) => Ok(Some(v)),
        Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
        Err(e) => Err(sq(e)),
    }
}

pub fn set_meta(conn: &Connection, key: &str, value: &str) -> Result<()> {
    conn.execute(
        "INSERT INTO meta(key,value) VALUES(?1,?2) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        [key, value],
    )
    .map_err(sq)?;
    Ok(())
}

pub const SELECT_COLS: &str =
    "key, id, content, memory_type, project_id, agent_source, tags, created_at, last_accessed, access_count, weight";

/// Maps a row selected with `SELECT_COLS` to `(key, record)`.
pub fn row_to_record(r: &Row<'_>) -> rusqlite::Result<(u64, MemoryRecord)> {
    let key: i64 = r.get(0)?;
    let mt: String = r.get(3)?;
    let tags_json: String = r.get(6)?;
    let ac: i64 = r.get(9)?;
    let rec = MemoryRecord {
        id: r.get(1)?,
        content: r.get(2)?,
        memory_type: MemoryType::parse(&mt).unwrap_or_default(),
        project_id: r.get(4)?,
        agent_source: r.get(5)?,
        tags: serde_json::from_str(&tags_json).unwrap_or_default(),
        created_at: r.get(7)?,
        last_accessed: r.get(8)?,
        access_count: ac.clamp(0, u32::MAX as i64) as u32,
        weight: r.get::<_, f64>(10)? as f32,
    };
    Ok((key as u64, rec))
}

pub fn embedding_to_blob(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

// `as_chunks` needs Rust 1.88; keep the declared MSRV (1.80).
#[allow(clippy::chunks_exact_to_as_chunks)]
pub fn blob_to_embedding(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blob_roundtrip() {
        let v = vec![0.5f32, -1.25, 3.0];
        assert_eq!(blob_to_embedding(&embedding_to_blob(&v)), v);
    }

    #[test]
    fn open_enables_wal_and_migrates_idempotently() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("m.db");
        {
            let c = open_writer(&p).unwrap();
            assert_eq!(
                get_meta(&c, "schema_version").unwrap().as_deref(),
                Some("1")
            );
            let mode: String = c
                .query_row("PRAGMA journal_mode", [], |r| r.get(0))
                .unwrap();
            assert_eq!(mode, "wal");
            let bt: i64 = c
                .query_row("PRAGMA busy_timeout", [], |r| r.get(0))
                .unwrap();
            assert_eq!(bt, 5000);
        }
        let c = open_writer(&p).unwrap();
        assert_eq!(
            get_meta(&c, "schema_version").unwrap().as_deref(),
            Some("1")
        );
        let r = open_reader(&p).unwrap();
        let n: i64 = r
            .query_row("SELECT COUNT(*) FROM memories", [], |r| r.get(0))
            .unwrap();
        assert_eq!(n, 0);
    }

    #[test]
    fn newer_schema_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("m.db");
        let c = open_writer(&p).unwrap();
        set_meta(&c, "schema_version", "99").unwrap();
        drop(c);
        assert!(open_writer(&p).is_err());
    }
}

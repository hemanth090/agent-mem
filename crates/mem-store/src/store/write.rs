//! Write path: `remember` (with dedup) and `forget`.

use super::*;

impl Store {
    // ----------------------------------------------------------------- remember

    pub fn remember_sync(&self, req: RememberRequest) -> Result<RememberResponse> {
        let content = validate_content(&req.content, self.opts.max_content_bytes)?;
        let mtype = match req.memory_type.as_deref() {
            Some(s) => MemoryType::parse(s)?,
            None => MemoryType::default(),
        };
        let project = normalize_project(req.project_id.as_deref())?;
        let tags = normalize_tags(&req.tags)?;
        let agent = normalize_agent(req.agent_source.as_deref());
        if !self.opts.allow_secrets {
            if let Some(what) = find_secret(&content) {
                return Err(Error::Rejected(format!(
                    "content looks like it contains a secret ({what}); secrets are never stored"
                )));
            }
        }
        // Embedding happens outside every lock.
        let emb = self
            .embedder
            .embed_passages(std::slice::from_ref(&content))?
            .pop()
            .ok_or_else(|| Error::Embed("empty embedding result".into()))?;
        self.insert_embedded(content, mtype, project, agent, tags, emb)
    }

    fn insert_embedded(
        &self,
        content: String,
        mtype: MemoryType,
        project: String,
        agent: String,
        tags: Vec<String>,
        emb: Vec<f32>,
    ) -> Result<RememberResponse> {
        // The writer mutex serializes dedup-check + insert, so concurrent identical
        // writes produce exactly one record.
        let mut conn = self.writer.lock();
        let now = now_ms();

        let nearest = {
            let meta = self.meta.read();
            self.index.search(&emb, 1, |k| {
                meta.get(&k)
                    .is_some_and(|m| &*m.project == project.as_str() && m.mtype == mtype)
            })?
        };
        if let Some(&(key, sim)) = nearest.first() {
            if sim > self.opts.dedup_threshold {
                let tx = conn.transaction().map_err(sq)?;
                let (id, old_tags): (String, String) = tx
                    .query_row(
                        "SELECT id, tags FROM memories WHERE key=?1",
                        [key as i64],
                        |r| Ok((r.get(0)?, r.get(1)?)),
                    )
                    .map_err(sq)?;
                let mut merged: Vec<String> = serde_json::from_str(&old_tags).unwrap_or_default();
                for t in &tags {
                    if !merged.contains(t) {
                        merged.push(t.clone());
                    }
                }
                merged.truncate(MAX_TAGS);
                let merged_json =
                    serde_json::to_string(&merged).map_err(|e| Error::Internal(e.to_string()))?;
                tx.execute(
                    "UPDATE memories SET last_accessed=?1, weight=MIN(?2, weight+?3), \
                     access_count=access_count+1, tags=?4 WHERE key=?5",
                    params![now, MAX_WEIGHT, DEDUP_WEIGHT_BUMP, merged_json, key as i64],
                )
                .map_err(sq)?;
                tx.commit().map_err(sq)?;
                if let Some(m) = self.meta.write().get_mut(&key) {
                    m.tags = Arc::new(merged);
                }
                return Ok(RememberResponse {
                    id,
                    deduplicated: true,
                    similarity: Some(sim),
                });
            }
        }

        let id = uuid::Uuid::now_v7().to_string();
        let tags_json = serde_json::to_string(&tags).map_err(|e| Error::Internal(e.to_string()))?;
        let tx = conn.transaction().map_err(sq)?;
        tx.execute(
            "INSERT INTO memories(id, content, memory_type, project_id, agent_source, tags, \
             created_at, last_accessed, access_count, weight, embedding) \
             VALUES (?1,?2,?3,?4,?5,?6,?7,?7,1,1.0,?8)",
            params![
                id,
                content,
                mtype.as_str(),
                project,
                agent,
                tags_json,
                now,
                embedding_to_blob(&emb)
            ],
        )
        .map_err(sq)?;
        let key = tx.last_insert_rowid() as u64;
        // If the index add fails, `tx` is dropped and rolled back.
        self.index.add(key, &emb)?;
        if let Err(e) = tx.commit() {
            let _ = self.index.remove(key);
            return Err(sq(e));
        }
        self.meta.write().insert(
            key,
            KeyMeta {
                project: Arc::from(project.as_str()),
                mtype,
                tags: Arc::new(tags),
            },
        );
        self.note_write();
        Ok(RememberResponse {
            id,
            deduplicated: false,
            similarity: None,
        })
    }

    pub async fn remember(self: &Arc<Self>, req: RememberRequest) -> Result<RememberResponse> {
        let s = self.clone();
        tokio::task::spawn_blocking(move || s.remember_sync(req))
            .await
            .map_err(join_err)?
    }

    // ------------------------------------------------------------------- forget

    pub fn forget_sync(&self, id: &str) -> Result<()> {
        let mut conn = self.writer.lock();
        let tx = conn.transaction().map_err(sq)?;
        let key: i64 =
            match tx.query_row("SELECT key FROM memories WHERE id=?1", [id], |r| r.get(0)) {
                Ok(k) => k,
                Err(rusqlite::Error::QueryReturnedNoRows) => {
                    return Err(Error::NotFound(format!("memory '{id}'")))
                }
                Err(e) => return Err(sq(e)),
            };
        tx.execute("DELETE FROM memories WHERE key=?1", [key])
            .map_err(sq)?;
        tx.commit().map_err(sq)?;
        // If we crash before the index is updated, the size check at startup rebuilds it.
        self.index.remove(key as u64)?;
        self.meta.write().remove(&(key as u64));
        self.note_write();
        Ok(())
    }

    pub async fn forget(self: &Arc<Self>, id: String) -> Result<()> {
        let s = self.clone();
        tokio::task::spawn_blocking(move || s.forget_sync(&id))
            .await
            .map_err(join_err)?
    }
}

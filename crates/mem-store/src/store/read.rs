//! Read path: hybrid `recall`, point reads and stats.

use super::*;

impl Store {
    // ------------------------------------------------------------------- recall

    pub(super) fn build_scope(&self, req: &RecallRequest) -> Result<(Scope, String, usize)> {
        let query = req.query.trim().to_string();
        if query.is_empty() {
            return Err(Error::Invalid("query is empty".into()));
        }
        let project = normalize_project(req.project_id.as_deref())?;
        let mut projects = vec![project.clone()];
        if req.include_global && project != GLOBAL_PROJECT {
            projects.push(GLOBAL_PROJECT.to_string());
        }
        let mtype = match req.memory_type.as_deref() {
            Some(s) => Some(MemoryType::parse(s)?),
            None => None,
        };
        let tags = normalize_tags(&req.tags)?;
        let top_k = req.top_k.clamp(1, MAX_TOP_K);
        Ok((
            Scope {
                projects,
                mtype,
                tags,
            },
            query,
            top_k,
        ))
    }

    pub(super) fn pool_size(&self, top_k: usize) -> usize {
        self.opts.candidate_pool.max(top_k * 2)
    }

    pub(super) fn ann_candidates(
        &self,
        q: &[f32],
        scope: &Scope,
        n: usize,
    ) -> Result<Vec<(u64, f32)>> {
        let meta = self.meta.read();
        let mut hits = self
            .index
            .search(q, n, |k| meta.get(&k).is_some_and(|m| scope.matches(m)))?;
        hits.retain(|(_, sim)| *sim >= self.opts.min_similarity);
        Ok(hits)
    }

    pub(super) fn fts_candidates(&self, query: &str, scope: &Scope, n: usize) -> Result<Vec<u64>> {
        let Some(fq) = fts_query(query) else {
            return Ok(Vec::new());
        };
        let mut sql = String::from(
            "SELECT memories_fts.rowid FROM memories_fts \
             JOIN memories m ON m.key = memories_fts.rowid \
             WHERE memories_fts MATCH ?1 AND m.project_id IN (",
        );
        let mut args: Vec<Value> = vec![Value::Text(fq)];
        for (i, p) in scope.projects.iter().enumerate() {
            if i > 0 {
                sql.push(',');
            }
            args.push(Value::Text(p.clone()));
            sql.push_str(&format!("?{}", args.len()));
        }
        sql.push(')');
        if let Some(t) = scope.mtype {
            args.push(Value::Text(t.as_str().to_string()));
            sql.push_str(&format!(" AND m.memory_type = ?{}", args.len()));
        }
        let fetch = if scope.tags.is_empty() { n } else { n * 4 };
        args.push(Value::Integer(fetch as i64));
        sql.push_str(&format!(
            " ORDER BY bm25(memories_fts) LIMIT ?{}",
            args.len()
        ));

        let keys: Vec<u64> = self.with_reader(|c| {
            let mut st = c.prepare_cached(&sql).map_err(sq)?;
            let rows = st
                .query_map(params_from_iter(args.iter()), |r| r.get::<_, i64>(0))
                .map_err(sq)?;
            let mut out = Vec::new();
            for r in rows {
                out.push(r.map_err(sq)? as u64);
            }
            Ok(out)
        })?;
        if scope.tags.is_empty() {
            return Ok(keys);
        }
        let meta = self.meta.read();
        Ok(keys
            .into_iter()
            .filter(|k| meta.get(k).is_some_and(|m| scope.matches(m)))
            .take(n)
            .collect())
    }

    pub(super) fn load_rows(&self, keys: &[u64]) -> Result<Vec<(u64, MemoryRecord)>> {
        if keys.is_empty() {
            return Ok(Vec::new());
        }
        let ph = vec!["?"; keys.len()].join(",");
        let sql = format!("SELECT {SELECT_COLS} FROM memories WHERE key IN ({ph})");
        self.with_reader(|c| {
            let mut st = c.prepare(&sql).map_err(sq)?;
            let rows = st
                .query_map(
                    params_from_iter(keys.iter().map(|k| *k as i64)),
                    row_to_record,
                )
                .map_err(sq)?;
            let mut out = Vec::new();
            for r in rows {
                out.push(r.map_err(sq)?);
            }
            Ok(out)
        })
    }

    pub(super) fn assemble(
        &self,
        ann: Vec<(u64, f32)>,
        fts: Vec<u64>,
        top_k: usize,
        started: Instant,
    ) -> Result<RecallResponse> {
        let ann_keys: Vec<u64> = ann.iter().map(|(k, _)| *k).collect();
        let ann_set: HashSet<u64> = ann_keys.iter().copied().collect();
        let fts_set: HashSet<u64> = fts.iter().copied().collect();
        let fused = rrf_fuse(&[ann_keys, fts], self.opts.rrf_k);
        let keys: Vec<u64> = fused.keys().copied().collect();
        let rows = self.load_rows(&keys)?;

        let now = now_ms();
        let mut scored: Vec<(f32, u64, MemoryRecord)> = rows
            .into_iter()
            .filter_map(|(k, rec)| {
                let rrf = *fused.get(&k)?;
                let hl = self.opts.half_lives.for_type(rec.memory_type);
                let s = final_score(
                    rrf,
                    rec.weight,
                    now,
                    rec.last_accessed,
                    hl,
                    rec.access_count,
                );
                Some((s, k, rec))
            })
            .collect();
        scored.sort_by(|a, b| {
            cmp_scored(
                (a.0, a.2.last_accessed, &a.2.id),
                (b.0, b.2.last_accessed, &b.2.id),
            )
        });
        scored.truncate(top_k);

        {
            let mut pending = self.pending_access.lock();
            for (_, k, _) in &scored {
                pending.push((*k, now));
            }
        }

        let hits = scored
            .into_iter()
            .map(|(score, k, rec)| {
                let mut sources = Vec::new();
                if ann_set.contains(&k) {
                    sources.push("vector".to_string());
                }
                if fts_set.contains(&k) {
                    sources.push("bm25".to_string());
                }
                RecallHit {
                    id: rec.id,
                    content: rec.content,
                    memory_type: rec.memory_type,
                    project_id: rec.project_id,
                    tags: rec.tags,
                    score,
                    created_at: rec.created_at,
                    last_accessed: rec.last_accessed,
                    access_count: rec.access_count,
                    sources,
                }
            })
            .collect();

        let took = started.elapsed().as_secs_f64() * 1000.0;
        {
            let mut l = self.latencies.lock();
            if l.len() >= LATENCY_WINDOW {
                l.pop_front();
            }
            l.push_back(took);
        }
        Ok(RecallResponse {
            hits,
            took_ms: took,
        })
    }

    /// Fully synchronous recall (ANN then BM25 sequentially). Used by tests and benchmarks.
    pub fn recall_sync(&self, req: &RecallRequest) -> Result<RecallResponse> {
        let started = Instant::now();
        let (scope, query, top_k) = self.build_scope(req)?;
        let q = self.embedder.embed_query(&query)?;
        let pool = self.pool_size(top_k);
        let ann = self.ann_candidates(&q, &scope, pool)?;
        let fts = self.fts_candidates(&query, &scope, pool)?;
        self.assemble(ann, fts, top_k, started)
    }

    /// Concurrent ANN + BM25 via `tokio::join!`, fused with RRF and decay.
    pub async fn recall(self: &Arc<Self>, req: RecallRequest) -> Result<RecallResponse> {
        let started = Instant::now();
        let (scope, query, top_k) = self.build_scope(&req)?;
        let pool = self.pool_size(top_k);
        let scope = Arc::new(scope);

        let s = self.clone();
        let q2 = query.clone();
        let qvec = tokio::task::spawn_blocking(move || s.embedder.embed_query(&q2))
            .await
            .map_err(join_err)??;

        let (s1, sc1) = (self.clone(), scope.clone());
        let ann_task = tokio::task::spawn_blocking(move || s1.ann_candidates(&qvec, &sc1, pool));
        let (s2, sc2) = (self.clone(), scope.clone());
        let fts_task = tokio::task::spawn_blocking(move || s2.fts_candidates(&query, &sc2, pool));
        let (ann, fts) = tokio::join!(ann_task, fts_task);
        let ann = ann.map_err(join_err)??;
        let fts = fts.map_err(join_err)??;

        let s3 = self.clone();
        tokio::task::spawn_blocking(move || s3.assemble(ann, fts, top_k, started))
            .await
            .map_err(join_err)?
    }

    // -------------------------------------------------------------------- reads

    pub fn get_sync(&self, id: &str) -> Result<Option<MemoryRecord>> {
        self.with_reader(|c| {
            match c.query_row(
                &format!("SELECT {SELECT_COLS} FROM memories WHERE id=?1"),
                [id],
                row_to_record,
            ) {
                Ok((_, r)) => Ok(Some(r)),
                Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
                Err(e) => Err(sq(e)),
            }
        })
    }

    pub fn recent_sync(&self, project: &str, n: usize) -> Result<Vec<MemoryRecord>> {
        let project = normalize_project(Some(project))?;
        self.with_reader(|c| {
            let mut st = c
                .prepare(&format!(
                    "SELECT {SELECT_COLS} FROM memories WHERE project_id=?1 \
                     ORDER BY last_accessed DESC LIMIT ?2"
                ))
                .map_err(sq)?;
            let rows = st
                .query_map(params![project, n.min(200) as i64], row_to_record)
                .map_err(sq)?;
            let mut out = Vec::new();
            for r in rows {
                out.push(r.map_err(sq)?.1);
            }
            Ok(out)
        })
    }

    pub async fn recent(self: &Arc<Self>, project: String, n: usize) -> Result<Vec<MemoryRecord>> {
        let s = self.clone();
        tokio::task::spawn_blocking(move || s.recent_sync(&project, n))
            .await
            .map_err(join_err)?
    }

    pub fn stats_sync(&self) -> Result<Stats> {
        let mut st = Stats::default();
        self.with_reader(|c| {
            st.total = c
                .query_row("SELECT COUNT(*) FROM memories", [], |r| r.get::<_, i64>(0))
                .map_err(sq)? as u64;
            let mut q = c
                .prepare("SELECT memory_type, COUNT(*) FROM memories GROUP BY memory_type")
                .map_err(sq)?;
            let rows = q
                .query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)?)))
                .map_err(sq)?;
            for r in rows {
                let (t, n) = r.map_err(sq)?;
                st.by_type.insert(t, n as u64);
            }
            let mut q = c
                .prepare(
                    "SELECT project_id, COUNT(*) AS n FROM memories GROUP BY project_id \
                     ORDER BY n DESC, project_id LIMIT 20",
                )
                .map_err(sq)?;
            let rows = q
                .query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)?)))
                .map_err(sq)?;
            for r in rows {
                let (p, n) = r.map_err(sq)?;
                st.by_project.push((p, n as u64));
            }
            Ok(())
        })?;
        st.index_vectors = self.index.len() as u64;
        st.index_bytes = self.index.serialized_bytes() as u64;
        let file_len = |p: &Path| std::fs::metadata(p).map(|m| m.len()).unwrap_or(0);
        let mut wal = self.opts.db_path.clone().into_os_string();
        wal.push("-wal");
        st.db_bytes = file_len(&self.opts.db_path) + file_len(Path::new(&wal));
        {
            let l = self.latencies.lock();
            st.recall_samples = l.len() as u64;
            if !l.is_empty() {
                st.avg_recall_ms = l.iter().sum::<f64>() / l.len() as f64;
                let mut v: Vec<f64> = l.iter().copied().collect();
                v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let idx = ((v.len() as f64 * 0.95).ceil() as usize).clamp(1, v.len()) - 1;
                st.p95_recall_ms = v[idx];
            }
        }
        let (h, m) = self.embedder.cache_counters();
        st.query_cache_hit_rate = if h + m == 0 {
            0.0
        } else {
            h as f64 / (h + m) as f64
        };
        st.model = self.embedder.model_name().to_string();
        st.uptime_s = self.started.elapsed().as_secs();
        st.version = env!("CARGO_PKG_VERSION").to_string();
        Ok(st)
    }

    pub async fn stats(self: &Arc<Self>) -> Result<Stats> {
        let s = self.clone();
        tokio::task::spawn_blocking(move || s.stats_sync())
            .await
            .map_err(join_err)?
    }
}

//! Store behaviour tests (fake embedder, temp directories).
use super::*;
use mem_embed::{CachedEmbedder, FakeEmbedder};

fn open(dir: &Path) -> Arc<Store> {
    let mut o = StoreOptions::new(dir);
    o.checkpoint_every_writes = 1_000_000;
    Store::open(o, Arc::new(CachedEmbedder::new(FakeEmbedder::new(), 64))).unwrap()
}

fn rem(
    s: &Store,
    content: &str,
    project: Option<&str>,
    mt: Option<&str>,
    tags: &[&str],
) -> RememberResponse {
    s.remember_sync(RememberRequest {
        content: content.into(),
        memory_type: mt.map(String::from),
        project_id: project.map(String::from),
        tags: tags.iter().map(|t| t.to_string()).collect(),
        agent_source: Some("test".into()),
    })
    .unwrap()
}

fn rec(q: &str, project: Option<&str>) -> RecallRequest {
    RecallRequest {
        query: q.into(),
        project_id: project.map(String::from),
        top_k: 5,
        memory_type: None,
        tags: vec![],
        include_global: true,
    }
}

#[test]
fn fts_query_sanitizes() {
    assert_eq!(fts_query("hello world").unwrap(), "\"hello\" OR \"world\"");
    assert_eq!(
        fts_query("a AND b* NEAR(c) \"d\"").unwrap(),
        "\"b\" OR \"near\" OR \"c\" OR \"d\""
    );
    assert_eq!(
        fts_query("How do we handle the retry policy?").unwrap(),
        "\"handle\" OR \"retry\" OR \"policy\""
    );
    // only stopwords: keep them rather than return nothing
    assert_eq!(
        fts_query("who are we").unwrap(),
        "\"who\" OR \"are\" OR \"we\""
    );
    assert!(fts_query("*** ()").is_none());
    assert!(fts_query("").is_none());
}

#[test]
fn remember_and_recall_roundtrip() {
    let d = tempfile::tempdir().unwrap();
    let s = open(d.path());
    rem(
        &s,
        "We use SQLite in WAL mode for persistence",
        Some("proj"),
        None,
        &["db"],
    );
    rem(
        &s,
        "Frontend uses React with Vite bundler",
        Some("proj"),
        None,
        &[],
    );
    let r = s
        .recall_sync(&rec("sqlite wal persistence", Some("proj")))
        .unwrap();
    assert!(!r.hits.is_empty());
    assert!(r.hits[0].content.contains("SQLite"));
    assert!(r.hits[0].sources.contains(&"vector".to_string()));
    assert!(r.hits[0].sources.contains(&"bm25".to_string()));
    assert_eq!(r.hits[0].tags, vec!["db"]);
}

#[test]
fn dedup_threshold_boundary_and_scope() {
    let d = tempfile::tempdir().unwrap();
    let s = open(d.path());
    let a = rem(&s, "database uses sqlite wal mode", Some("p"), None, &["a"]);
    assert!(!a.deduplicated);
    // identical text -> similarity 1.0 > 0.95 -> dedup
    let b = rem(&s, "database uses sqlite wal mode", Some("p"), None, &["b"]);
    assert!(b.deduplicated);
    assert_eq!(b.id, a.id);
    assert!(b.similarity.unwrap() > 0.99);
    let rec1 = s.get_sync(&a.id).unwrap().unwrap();
    assert_eq!(rec1.access_count, 2);
    assert!(rec1.weight > 1.1);
    assert_eq!(rec1.tags, vec!["a", "b"]);
    assert_eq!(s.stats_sync().unwrap().total, 1);
    // same text, different project -> separate record
    let c = rem(&s, "database uses sqlite wal mode", Some("q"), None, &[]);
    assert!(!c.deduplicated);
    // same text, different type -> separate record
    let e = rem(
        &s,
        "database uses sqlite wal mode",
        Some("p"),
        Some("procedural"),
        &[],
    );
    assert!(!e.deduplicated);
    // clearly different text -> new record
    let f = rem(
        &s,
        "completely unrelated banana smoothie",
        Some("p"),
        None,
        &[],
    );
    assert!(!f.deduplicated);
    assert_eq!(s.stats_sync().unwrap().total, 4);
}

#[test]
fn dedup_below_threshold_is_not_merged() {
    let d = tempfile::tempdir().unwrap();
    let mut o = StoreOptions::new(d.path());
    // 3 of 4 words shared => cosine 0.75 with the fake embedder.
    o.dedup_threshold = 0.8;
    let s = Store::open(o, Arc::new(FakeEmbedder::new())).unwrap();
    rem(&s, "alpha bravo charlie delta", Some("p"), None, &[]);
    let r = rem(&s, "alpha bravo charlie echo", Some("p"), None, &[]);
    assert!(!r.deduplicated);
    let d2 = tempfile::tempdir().unwrap();
    let mut o = StoreOptions::new(d2.path());
    o.dedup_threshold = 0.7;
    let s = Store::open(o, Arc::new(FakeEmbedder::new())).unwrap();
    rem(&s, "alpha bravo charlie delta", Some("p"), None, &[]);
    let r = rem(&s, "alpha bravo charlie echo", Some("p"), None, &[]);
    assert!(r.deduplicated);
}

#[test]
fn min_similarity_drops_unrelated_vector_hits() {
    let d = tempfile::tempdir().unwrap();
    let s = open(d.path());
    rem(
        &s,
        "billing webhooks retry with backoff",
        Some("p"),
        None,
        &[],
    );
    // No shared words: cosine 0 -> below the floor, and BM25 has nothing either.
    let r = s
        .recall_sync(&rec("kubernetes ingress", Some("p")))
        .unwrap();
    assert!(r.hits.is_empty());
    let r = s.recall_sync(&rec("billing webhooks", Some("p"))).unwrap();
    assert_eq!(r.hits.len(), 1);

    let d2 = tempfile::tempdir().unwrap();
    let mut o = StoreOptions::new(d2.path());
    o.min_similarity = 0.0;
    let s = Store::open(o, Arc::new(FakeEmbedder::new())).unwrap();
    rem(
        &s,
        "billing webhooks retry with backoff",
        Some("p"),
        None,
        &[],
    );
    let r = s
        .recall_sync(&rec("kubernetes ingress", Some("p")))
        .unwrap();
    assert_eq!(
        r.hits.len(),
        1,
        "floor disabled -> nearest neighbour returned"
    );
}

#[test]
fn project_and_global_scoping() {
    let d = tempfile::tempdir().unwrap();
    let s = open(d.path());
    rem(
        &s,
        "redis cache decision for project a",
        Some("a"),
        None,
        &[],
    );
    rem(
        &s,
        "redis cache decision for project b",
        Some("b"),
        None,
        &[],
    );
    rem(&s, "redis cache global preference", None, None, &[]);

    let r = s.recall_sync(&rec("redis cache", Some("a"))).unwrap();
    let projects: HashSet<_> = r.hits.iter().map(|h| h.project_id.clone()).collect();
    assert_eq!(
        projects,
        HashSet::from(["a".to_string(), "global".to_string()])
    );

    let mut req = rec("redis cache", Some("a"));
    req.include_global = false;
    let r = s.recall_sync(&req).unwrap();
    assert_eq!(r.hits.len(), 1);
    assert_eq!(r.hits[0].project_id, "a");

    let r = s.recall_sync(&rec("redis cache", None)).unwrap();
    assert!(r.hits.iter().all(|h| h.project_id == "global"));
}

#[test]
fn type_and_tag_filters() {
    let d = tempfile::tempdir().unwrap();
    let s = open(d.path());
    rem(
        &s,
        "deploy with make release",
        Some("p"),
        Some("procedural"),
        &["deploy"],
    );
    rem(
        &s,
        "deploy failed last tuesday",
        Some("p"),
        Some("episodic"),
        &["incident"],
    );
    let mut req = rec("deploy", Some("p"));
    req.memory_type = Some("procedural".into());
    let r = s.recall_sync(&req).unwrap();
    assert_eq!(r.hits.len(), 1);
    assert_eq!(r.hits[0].memory_type, MemoryType::Procedural);
    let mut req = rec("deploy", Some("p"));
    req.tags = vec!["incident".into()];
    let r = s.recall_sync(&req).unwrap();
    assert_eq!(r.hits.len(), 1);
    assert!(r.hits[0].content.contains("failed"));
}

#[test]
fn forget_removes_from_everything() {
    let d = tempfile::tempdir().unwrap();
    let s = open(d.path());
    let a = rem(&s, "obsolete fact about zebra", Some("p"), None, &[]);
    rem(&s, "keep this other fact", Some("p"), None, &[]);
    s.forget_sync(&a.id).unwrap();
    assert!(s.get_sync(&a.id).unwrap().is_none());
    let r = s
        .recall_sync(&rec("zebra obsolete fact", Some("p")))
        .unwrap();
    assert!(r.hits.iter().all(|h| h.id != a.id));
    // FTS must not match it either
    let fts = s
        .fts_candidates(
            "zebra",
            &Scope {
                projects: vec!["p".into()],
                mtype: None,
                tags: vec![],
            },
            10,
        )
        .unwrap();
    assert!(fts.is_empty());
    assert_eq!(s.index.len(), 1);
    assert!(matches!(s.forget_sync(&a.id), Err(Error::NotFound(_))));
}

#[test]
fn rejects_secrets_and_bad_input() {
    let d = tempfile::tempdir().unwrap();
    let s = open(d.path());
    let r = s.remember_sync(RememberRequest {
        content: "my key is AKIAIOSFODNN7EXAMPLE".into(),
        ..Default::default()
    });
    assert!(matches!(r, Err(Error::Rejected(_))));
    let r = s.remember_sync(RememberRequest {
        content: "  ".into(),
        ..Default::default()
    });
    assert!(matches!(r, Err(Error::Invalid(_))));
    let r = s.remember_sync(RememberRequest {
        content: "x".into(),
        memory_type: Some("bogus".into()),
        ..Default::default()
    });
    assert!(matches!(r, Err(Error::Invalid(_))));
    assert!(s.recall_sync(&rec("   ", None)).is_err());
}

#[test]
fn index_deleted_is_rebuilt() {
    let d = tempfile::tempdir().unwrap();
    let ids: Vec<String>;
    {
        let s = open(d.path());
        ids = (0..5)
            .map(|i| {
                rem(
                    &s,
                    &format!("unique fact number{i} about topic{i}"),
                    Some("p"),
                    None,
                    &[],
                )
                .id
            })
            .collect();
        s.shutdown_sync().unwrap();
    }
    std::fs::remove_file(StoreOptions::new(d.path()).index_path).unwrap();
    let s = open(d.path());
    assert_eq!(s.index.len(), 5);
    let r = s.recall_sync(&rec("number3 topic3", Some("p"))).unwrap();
    assert_eq!(r.hits[0].id, ids[3]);
}

#[test]
fn index_size_mismatch_is_rebuilt() {
    let d = tempfile::tempdir().unwrap();
    {
        let s = open(d.path());
        rem(&s, "first fact alpha", Some("p"), None, &[]);
        s.shutdown_sync().unwrap(); // index file now has 1 vector
        rem(&s, "second fact beta", Some("p"), None, &[]);
        // crash: no checkpoint, so the file on disk is stale (1 vs 2 rows)
    }
    let s = open(d.path());
    assert_eq!(s.index.len(), 2);
}

#[test]
fn corrupt_index_is_rebuilt() {
    let d = tempfile::tempdir().unwrap();
    {
        let s = open(d.path());
        rem(&s, "some fact", Some("p"), None, &[]);
        s.shutdown_sync().unwrap();
    }
    std::fs::write(StoreOptions::new(d.path()).index_path, b"garbage garbage").unwrap();
    let s = open(d.path());
    assert_eq!(s.index.len(), 1);
}

#[test]
fn model_mismatch_refused_then_reindex_fixes() {
    struct Other;
    impl Embedder for Other {
        fn embed_passages(&self, t: &[String]) -> Result<Vec<Vec<f32>>> {
            FakeEmbedder::new().embed_passages(t)
        }
        fn embed_query(&self, t: &str) -> Result<Vec<f32>> {
            FakeEmbedder::new().embed_query(t)
        }
        fn dim(&self) -> usize {
            384
        }
        fn model_name(&self) -> &str {
            "other-model"
        }
    }
    let d = tempfile::tempdir().unwrap();
    {
        let s = open(d.path());
        rem(&s, "a fact", Some("p"), None, &[]);
        s.shutdown_sync().unwrap();
    }
    let o = StoreOptions::new(d.path());
    assert!(Store::open(o.clone(), Arc::new(Other)).is_err());
    assert_eq!(Store::reindex(&o, &Other).unwrap(), 1);
    let s = Store::open(o, Arc::new(Other)).unwrap();
    assert_eq!(s.index.len(), 1);
}

#[test]
fn access_flush_batches_updates() {
    let d = tempfile::tempdir().unwrap();
    let s = open(d.path());
    let a = rem(&s, "frequently used fact", Some("p"), None, &[]);
    for _ in 0..3 {
        s.recall_sync(&rec("frequently used fact", Some("p")))
            .unwrap();
    }
    assert_eq!(s.get_sync(&a.id).unwrap().unwrap().access_count, 1);
    assert_eq!(s.flush_access().unwrap(), 1);
    assert_eq!(s.get_sync(&a.id).unwrap().unwrap().access_count, 4);
    assert_eq!(s.flush_access().unwrap(), 0);
}

#[test]
fn decay_and_usage_affect_ranking() {
    let d = tempfile::tempdir().unwrap();
    let s = open(d.path());
    let old = rem(
        &s,
        "cache policy decision alpha",
        Some("p"),
        Some("episodic"),
        &[],
    );
    let fresh = rem(
        &s,
        "cache policy decision beta",
        Some("p"),
        Some("episodic"),
        &[],
    );
    {
        // age "old" by 60 days (> 4 episodic half-lives)
        let w = s.writer_for_test();
        w.execute(
            "UPDATE memories SET last_accessed = last_accessed - ?1 WHERE id = ?2",
            params![60i64 * 86_400_000, old.id],
        )
        .unwrap();
    }
    let r = s
        .recall_sync(&rec("cache policy decision", Some("p")))
        .unwrap();
    assert_eq!(r.hits[0].id, fresh.id);
    assert_eq!(r.hits[1].id, old.id);
    assert!(r.hits[0].score > r.hits[1].score * 4.0);
}

#[test]
fn stats_report() {
    let d = tempfile::tempdir().unwrap();
    let s = open(d.path());
    rem(&s, "one fact", Some("p"), Some("semantic"), &[]);
    rem(&s, "two fact", Some("q"), Some("procedural"), &[]);
    s.recall_sync(&rec("fact", Some("p"))).unwrap();
    let st = s.stats_sync().unwrap();
    assert_eq!(st.total, 2);
    assert_eq!(st.by_type["semantic"], 1);
    assert_eq!(st.by_type["procedural"], 1);
    assert_eq!(st.index_vectors, 2);
    assert!(st.db_bytes > 0);
    assert_eq!(st.recall_samples, 1);
    assert!(st.avg_recall_ms >= 0.0);
    assert_eq!(st.model, "fake-hash-384");
}

#[test]
fn recent_orders_by_last_accessed() {
    let d = tempfile::tempdir().unwrap();
    let s = open(d.path());
    rem(&s, "older fact one", Some("p"), None, &[]);
    std::thread::sleep(Duration::from_millis(5));
    rem(&s, "newer fact two", Some("p"), None, &[]);
    let r = s.recent_sync("p", 10).unwrap();
    assert_eq!(r.len(), 2);
    assert!(r[0].content.contains("newer"));
}

#[test]
fn concurrent_identical_writes_yield_one_record() {
    let d = tempfile::tempdir().unwrap();
    let s = open(d.path());
    let handles: Vec<_> = (0..16)
        .map(|_| {
            let s = s.clone();
            std::thread::spawn(move || {
                s.remember_sync(RememberRequest {
                    content: "the very same architectural decision".into(),
                    project_id: Some("p".into()),
                    ..Default::default()
                })
                .unwrap()
            })
        })
        .collect();
    let res: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    assert_eq!(res.iter().filter(|r| !r.deduplicated).count(), 1);
    assert_eq!(s.stats_sync().unwrap().total, 1);
}

#[test]
fn concurrent_writers_and_readers() {
    let d = tempfile::tempdir().unwrap();
    let s = open(d.path());
    let mut hs = Vec::new();
    for w in 0..8 {
        let s = s.clone();
        hs.push(std::thread::spawn(move || {
            for i in 0..10 {
                s.remember_sync(RememberRequest {
                    content: format!("writer{w} item{i} token{w}x{i}"),
                    project_id: Some("p".into()),
                    ..Default::default()
                })
                .unwrap();
            }
        }));
    }
    for r in 0..8 {
        let s = s.clone();
        hs.push(std::thread::spawn(move || {
            for _ in 0..20 {
                s.recall_sync(&rec(&format!("writer{r} item"), Some("p")))
                    .unwrap();
            }
        }));
    }
    for h in hs {
        h.join().unwrap();
    }
    let st = s.stats_sync().unwrap();
    assert_eq!(st.total, 80);
    assert_eq!(st.index_vectors, 80);
}

#[test]
fn survives_external_read_transaction() {
    let d = tempfile::tempdir().unwrap();
    let s = open(d.path());
    rem(&s, "initial fact", Some("p"), None, &[]);
    let ext = Connection::open(StoreOptions::new(d.path()).db_path).unwrap();
    ext.execute_batch("BEGIN; SELECT COUNT(*) FROM memories;")
        .unwrap();
    rem(
        &s,
        "written while a reader holds a snapshot",
        Some("p"),
        None,
        &[],
    );
    ext.execute_batch("COMMIT").unwrap();
    assert_eq!(s.stats_sync().unwrap().total, 2);
}

#[tokio::test]
async fn async_api_end_to_end() {
    let d = tempfile::tempdir().unwrap();
    let s = open(d.path());
    let r = s
        .remember(RememberRequest {
            content: "async path works with tokio join".into(),
            project_id: Some("p".into()),
            ..Default::default()
        })
        .await
        .unwrap();
    let out = s.recall(rec("tokio join async", Some("p"))).await.unwrap();
    assert_eq!(out.hits[0].id, r.id);
    assert_eq!(s.stats().await.unwrap().total, 1);
    s.forget(r.id.clone()).await.unwrap();
    assert!(s
        .recall(rec("tokio join async", Some("p")))
        .await
        .unwrap()
        .hits
        .is_empty());
    s.shutdown().await.unwrap();
}

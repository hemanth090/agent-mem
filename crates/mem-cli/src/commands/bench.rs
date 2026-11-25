//! `agent-mem bench-recall`: measures real retrieval latency on a throwaway database.

use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;
use mem_core::{Config, RecallRequest, RememberRequest};
use mem_store::{Store, StoreOptions};

const TOPICS: [&str; 12] = [
    "authentication",
    "database",
    "caching",
    "deployment",
    "logging",
    "testing",
    "serialization",
    "networking",
    "scheduling",
    "permissions",
    "migrations",
    "observability",
];
const VERBS: [&str; 8] = [
    "uses",
    "avoids",
    "requires",
    "retries",
    "validates",
    "batches",
    "debounces",
    "isolates",
];

fn synth(i: usize) -> String {
    let t = TOPICS[i % TOPICS.len()];
    let v = VERBS[(i / TOPICS.len()) % VERBS.len()];
    format!(
        "Module m{} {v} {t} with strategy s{} because incident {} showed that component c{} breaks under load.",
        i % 997,
        i % 211,
        i,
        i % 53
    )
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((sorted.len() as f64 * p).ceil() as usize).clamp(1, sorted.len()) - 1;
    sorted[idx]
}

fn report(label: &str, mut v: Vec<f64>) {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    println!(
        "{label:<28} p50 {:>7.2} ms   p95 {:>7.2} ms   p99 {:>7.2} ms   max {:>7.2} ms",
        percentile(&v, 0.50),
        percentile(&v, 0.95),
        percentile(&v, 0.99),
        v.last().copied().unwrap_or(0.0)
    );
}

pub async fn run(cfg: &Config, n: usize, queries: usize, fake: bool) -> Result<()> {
    let dir = tempfile::tempdir()?;
    let bench_cfg = Config {
        data_dir: dir.path().to_path_buf(),
        ..cfg.clone()
    };
    let embedder =
        crate::embedder::build_embedder(&bench_cfg, false, fake).map_err(|e| anyhow::anyhow!(e))?;
    println!(
        "embedder: {}   records: {n}   queries: {queries}",
        embedder.model_name()
    );
    let store = Store::open(StoreOptions::from_config(&bench_cfg), Arc::clone(&embedder))
        .map_err(|e| anyhow::anyhow!(e))?;

    let seed_start = Instant::now();
    let mut remember_ms = Vec::new();
    for i in 0..n {
        let t = Instant::now();
        store
            .remember(RememberRequest {
                content: synth(i),
                project_id: Some(if i % 5 == 0 {
                    "global".into()
                } else {
                    "bench".into()
                }),
                ..Default::default()
            })
            .await
            .map_err(|e| anyhow::anyhow!(e))?;
        remember_ms.push(t.elapsed().as_secs_f64() * 1000.0);
    }
    println!("seeded in {:.1}s", seed_start.elapsed().as_secs_f64());
    report("remember (incl. embedding)", remember_ms);

    let mut cold = Vec::new();
    let mut warm = Vec::new();
    for q in 0..queries {
        let req = RecallRequest {
            // The unique token makes every query text new, so the first call is a true cold embed.
            query: format!(
                "{} {} strategy q{q}",
                TOPICS[q % TOPICS.len()],
                VERBS[q % VERBS.len()]
            ),
            project_id: Some("bench".into()),
            top_k: 5,
            memory_type: None,
            tags: vec![],
            include_global: true,
        };
        let t = Instant::now();
        store
            .recall(req.clone())
            .await
            .map_err(|e| anyhow::anyhow!(e))?;
        let c = t.elapsed().as_secs_f64() * 1000.0;
        let t = Instant::now();
        store.recall(req).await.map_err(|e| anyhow::anyhow!(e))?;
        let w = t.elapsed().as_secs_f64() * 1000.0;
        cold.push(c);
        warm.push(w);
    }
    report("recall (cold query embed)", cold);
    report("recall (query cached)", warm);
    let st = store.stats().await.map_err(|e| anyhow::anyhow!(e))?;
    println!(
        "index vectors: {}   index size: {:.1} MB   db: {:.1} MB",
        st.index_vectors,
        st.index_bytes as f64 / 1e6,
        st.db_bytes as f64 / 1e6
    );
    Ok(())
}

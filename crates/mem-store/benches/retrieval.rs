#![allow(clippy::unwrap_used, clippy::expect_used)]
//! Retrieval micro-benchmarks with the hashing fake embedder (so they measure the store,
//! not ONNX). Use `agent-mem bench-recall` for end-to-end numbers with the real model.

use std::hint::black_box;
use std::sync::Arc;

use criterion::{criterion_group, criterion_main, Criterion};
use mem_core::{RecallRequest, RememberRequest};
use mem_embed::FakeEmbedder;
use mem_store::{Store, StoreOptions};

const N: usize = 20_000;

fn seed() -> (tempfile::TempDir, Arc<Store>) {
    let dir = tempfile::tempdir().unwrap();
    let mut o = StoreOptions::new(dir.path());
    o.checkpoint_every_writes = u64::MAX;
    let store = Store::open(o, Arc::new(FakeEmbedder::new())).unwrap();
    for i in 0..N {
        store
            .remember_sync(RememberRequest {
                content: format!(
                    "module m{} handles topic{} with strategy s{} after incident {}",
                    i % 997,
                    i % 61,
                    i % 211,
                    i
                ),
                project_id: Some("bench".into()),
                ..Default::default()
            })
            .unwrap();
    }
    (dir, store)
}

fn bench(c: &mut Criterion) {
    let (_dir, store) = seed();
    let req = RecallRequest {
        query: "module m42 topic7 strategy".into(),
        project_id: Some("bench".into()),
        top_k: 5,
        memory_type: None,
        tags: vec![],
        include_global: true,
    };
    c.bench_function("recall_e2e_fake_embedder_20k", |b| {
        b.iter(|| store.recall_sync(black_box(&req)).unwrap())
    });
    let selective = RecallRequest {
        query: "m42 topic7 s5".into(),
        ..req.clone()
    };
    c.bench_function("recall_e2e_selective_terms_20k", |b| {
        b.iter(|| store.recall_sync(black_box(&selective)).unwrap())
    });
    let mut g = c.benchmark_group("remember");
    g.sample_size(30);
    let mut n = 0u64;
    g.bench_function("insert_into_20k", |b| {
        b.iter(|| {
            n += 1;
            store
                .remember_sync(RememberRequest {
                    content: format!("fresh unique bench fact {n} zz{n}"),
                    project_id: Some("bench".into()),
                    ..Default::default()
                })
                .unwrap()
        })
    });
    g.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);

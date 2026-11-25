#![allow(clippy::unwrap_used, clippy::expect_used)]
use std::hint::black_box;

use criterion::{criterion_group, criterion_main, Criterion};
use mem_core::rank::{final_score, rrf_fuse};

fn bench_fuse(c: &mut Criterion) {
    let ann: Vec<u64> = (0..50).collect();
    let fts: Vec<u64> = (25..75).collect();
    c.bench_function("rrf_fuse_2x50", |b| {
        b.iter(|| rrf_fuse(black_box(&[ann.clone(), fts.clone()]), 60.0))
    });
    c.bench_function("final_score", |b| {
        b.iter(|| {
            final_score(
                black_box(0.03),
                1.0,
                10_000_000_000,
                9_000_000_000,
                180.0,
                7,
            )
        })
    });
}

criterion_group!(benches, bench_fuse);
criterion_main!(benches);

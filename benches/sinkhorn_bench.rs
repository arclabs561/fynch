use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

fn bench_sinkhorn(c: &mut Criterion) {
    let mut group = c.benchmark_group("sinkhorn");

    let values = vec![3.0, 1.0, 2.0, 5.0, 4.0, 0.5, 7.0, 6.0];

    for &epsilon in &[1.0, 0.1] {
        group.bench_with_input(
            BenchmarkId::new("sinkhorn_rank", epsilon),
            &epsilon,
            |b, &eps| b.iter(|| fynch::sinkhorn::sinkhorn_rank(black_box(&values), eps)),
        );
    }

    group.finish();
}

criterion_group!(benches, bench_sinkhorn);
criterion_main!(benches);

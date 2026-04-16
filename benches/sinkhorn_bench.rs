use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use fynch::sorting_network::{bitonic_sort, odd_even_sort};

fn bench_sinkhorn(c: &mut Criterion) {
    let mut group = c.benchmark_group("sinkhorn");

    let values = vec![3.0, 1.0, 2.0, 5.0, 4.0, 0.5, 7.0, 6.0];

    for &epsilon in &[1.0, 0.1] {
        group.bench_with_input(
            BenchmarkId::new("sinkhorn_rank/n=8", epsilon),
            &epsilon,
            |b, &eps| b.iter(|| fynch::sinkhorn::sinkhorn_rank(black_box(&values), eps)),
        );
    }

    // Larger sizes
    for &n in &[32usize, 64, 128] {
        let big: Vec<f64> = (0..n).map(|i| (i as f64 * 1.3 + 0.7) % (n as f64)).collect();
        group.bench_with_input(
            BenchmarkId::new("sinkhorn_rank/eps=0.5", n),
            &n,
            |b, _| b.iter(|| fynch::sinkhorn::sinkhorn_rank(black_box(&big), 0.5)),
        );
    }

    group.finish();
}

fn bench_sorting_networks(c: &mut Criterion) {
    let mut group = c.benchmark_group("sorting_network");

    for &n in &[4usize, 8, 16, 32] {
        let x: Vec<f64> = (0..n).map(|i| (n - i) as f64).collect();
        group.bench_with_input(BenchmarkId::new("bitonic", n), &n, |b, _| {
            b.iter(|| bitonic_sort(black_box(&x), 10.0))
        });
        group.bench_with_input(BenchmarkId::new("odd_even", n), &n, |b, _| {
            b.iter(|| odd_even_sort(black_box(&x), 10.0))
        });
    }

    group.finish();
}

criterion_group!(benches, bench_sinkhorn, bench_sorting_networks);
criterion_main!(benches);

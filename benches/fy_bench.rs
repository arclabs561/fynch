use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

fn bench_fenchel_predictions(c: &mut Criterion) {
    let mut group = c.benchmark_group("fenchel_predictions");

    let theta = vec![2.0, 1.0, 0.1, -0.3, 3.2, 0.0, 0.7, -1.1, 0.9, 1.4];

    for &temperature in &[1.0, 0.5, 0.1] {
        group.bench_with_input(
            BenchmarkId::new("softmax_with_temperature", temperature),
            &temperature,
            |b, &temp| b.iter(|| fynch::fenchel::softmax_with_temperature(black_box(&theta), temp)),
        );
    }

    group.bench_function("sparsemax", |b| {
        b.iter(|| fynch::fenchel::sparsemax(black_box(&theta)))
    });

    group.finish();
}

criterion_group!(benches, bench_fenchel_predictions);
criterion_main!(benches);

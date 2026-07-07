# Changelog

## [Unreleased]

## [0.3.2] - 2026-07-07

### Changed
- Delegated the optional Gumbel noise and Gumbel-Softmax helpers to `drawset`.

## [0.3.1] - 2026-07-07

### Changed
- Switched the optional `gumbel` feature from `kuji` to `drawset`.

## [0.3.0] - 2026-07-03

### Fixed
- `fast_soft_sort` now implements Blondel et al. (2020) soft sort as the
  reference implementation (google-research/fast-soft-sort) defines it,
  ascending with L2 regularization. The previous construction applied PAVA to
  `sorted - rho` with the operand roles inverted, so its outputs matched the
  reference only on inputs whose sorted values happen to equal their ranks,
  and its small-regularization limit was `rho * strength` (tending to zero)
  instead of the hard-sorted values. Outputs change for effectively all
  inputs; the parameter is now named `regularization_strength` to match its
  role (smaller = closer to hard sort).

### Added
- `fast_soft_rank`: the $O(n \log n)$ Blondel soft rank via the permutahedron
  projection, ascending with L2 regularization, returning soft ranks in input
  order. Distinct from the $O(n^2)$ pairwise-sigmoid `soft_rank` relaxation.
- `tests/blondel_reference.rs` with a fixture generated from the reference
  implementation, pinning both functions to externally produced values
  (inputs with ties and negatives, strengths 0.1 / 1.0 / 10.0), plus a
  hard-limit rank test independent of the fixture.

## [0.2.2] - 2026-06-26

### Added
- Optional `logp` feature implementing `logp::BregmanGenerator` for
  `Shannon` and `SquaredL2`.

## [0.2.1] - 2026-06-26

### Added
- `sparsemap_explicit` and `sparsemap_loss_explicit` for SparseMAP over a
  small explicit vertex set.

## [0.1.4] - 2026-06-10

### Changed
- Bumped `innr` to 0.4.
- README and CONTRIBUTING polish; publish gated on cargo-semver-checks.

Earlier releases predate this changelog; see git history.

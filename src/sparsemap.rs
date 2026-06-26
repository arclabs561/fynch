//! SparseMAP over an explicit finite structured domain.
//!
//! SparseMAP is the Fenchel-Young prediction map induced by squared L2
//! regularization over a structured marginal polytope. This module starts with
//! the smallest useful domain representation: callers provide the vertices of
//! the polytope explicitly, and the solver returns the projected marginal plus
//! the sparse convex combination that produced it.
//!
//! This is intentionally not wired into [`crate::fenchel::Regularizer`].
//! `Regularizer` is the simplex-valued API; SparseMAP needs a structured domain
//! boundary.

use crate::{Error, Result};

const MAX_EXPLICIT_VERTICES: usize = 16;
const TOL: f64 = 1e-10;

/// One active vertex in a SparseMAP prediction.
#[derive(Debug, Clone, PartialEq)]
pub struct SparseMapWeight {
    /// Index of the input vertex.
    pub vertex: usize,
    /// Convex-combination weight assigned to the vertex.
    pub weight: f64,
}

/// SparseMAP prediction over an explicit finite vertex set.
#[derive(Debug, Clone, PartialEq)]
pub struct SparseMapPrediction {
    /// The predicted marginal vector.
    pub marginal: Vec<f64>,
    /// Sparse convex combination of input vertices that forms `marginal`.
    pub active: Vec<SparseMapWeight>,
    /// Fenchel conjugate value at the input scores.
    pub conjugate: f64,
}

/// Compute SparseMAP over an explicit list of vertices.
///
/// Given scores `theta` and vertices `v_j`, this solves:
///
/// ```text
/// argmax_mu <theta, mu> - 0.5 ||mu||^2
///   where mu is in conv({v_j})
/// ```
///
/// Equivalently, it projects `theta` onto the convex hull of the supplied
/// vertices. For standard-basis vertices this recovers sparsemax.
///
/// The explicit implementation is bounded to small domains
/// (`MAX_EXPLICIT_VERTICES`) because it is the seed prototype for the
/// structured-oracle API, not a graph or sequence decoder.
///
/// # Errors
///
/// Returns [`Error::EmptyInput`] when `theta` or `vertices` is empty,
/// [`Error::LengthMismatch`] when a vertex has the wrong dimension, and
/// [`Error::InvalidWeights`] when values are non-finite or the explicit vertex
/// limit is exceeded.
pub fn sparsemap_explicit(theta: &[f64], vertices: &[Vec<f64>]) -> Result<SparseMapPrediction> {
    let domain = ExplicitDomain::new(theta, vertices)?;
    let (first, first_vertex) = domain.best_vertex(theta);

    let mut active = vec![first];
    let mut weights = vec![1.0];
    let mut marginal = first_vertex.to_vec();

    for _ in 0..(vertices.len() * 4 + 4) {
        let gradient: Vec<f64> = theta
            .iter()
            .zip(&marginal)
            .map(|(&score, &mu)| score - mu)
            .collect();
        let (candidate, vertex) = domain.best_vertex(&gradient);
        let gap = dot(&gradient, vertex) - dot(&gradient, &marginal);

        if gap <= TOL {
            return Ok(prediction(theta, marginal, &active, &weights));
        }

        if !active.contains(&candidate) {
            active.push(candidate);
        }

        let projected = project_active(theta, vertices, &active);
        active = projected.active;
        weights = projected.weights;
        marginal = projected.marginal;
    }

    Ok(prediction(theta, marginal, &active, &weights))
}

/// Compute the SparseMAP Fenchel-Young loss for an explicit-domain target.
///
/// This computes:
///
/// ```text
/// L(theta; y) = Omega*(theta) - <theta, y> + 0.5 ||y||^2
/// ```
///
/// where `Omega*` is supplied by [`sparsemap_explicit`].
///
/// `target` is expected to be a marginal vector for the same explicit domain.
/// This function validates shape and finiteness, but does not project or
/// otherwise repair targets outside the convex hull.
pub fn sparsemap_loss_explicit(
    theta: &[f64],
    target: &[f64],
    vertices: &[Vec<f64>],
) -> Result<f64> {
    if theta.len() != target.len() {
        return Err(Error::LengthMismatch(theta.len(), target.len()));
    }
    if !target.iter().all(|v| v.is_finite()) {
        return Err(Error::InvalidWeights);
    }

    let prediction = sparsemap_explicit(theta, vertices)?;
    Ok(prediction.conjugate - dot(theta, target) + 0.5 * dot(target, target))
}

struct ExplicitDomain<'a> {
    dim: usize,
    vertices: &'a [Vec<f64>],
}

impl<'a> ExplicitDomain<'a> {
    fn new(theta: &[f64], vertices: &'a [Vec<f64>]) -> Result<Self> {
        if theta.is_empty() || vertices.is_empty() {
            return Err(Error::EmptyInput);
        }
        if vertices.len() > MAX_EXPLICIT_VERTICES {
            return Err(Error::InvalidWeights);
        }
        if !theta.iter().all(|v| v.is_finite()) {
            return Err(Error::InvalidWeights);
        }

        for vertex in vertices {
            if vertex.len() != theta.len() {
                return Err(Error::LengthMismatch(vertex.len(), theta.len()));
            }
            if !vertex.iter().all(|v| v.is_finite()) {
                return Err(Error::InvalidWeights);
            }
        }

        Ok(Self {
            dim: theta.len(),
            vertices,
        })
    }

    fn best_vertex(&self, scores: &[f64]) -> (usize, &[f64]) {
        debug_assert_eq!(scores.len(), self.dim);
        let mut best = 0;
        let mut best_score = dot(scores, &self.vertices[0]);

        for (index, vertex) in self.vertices.iter().enumerate().skip(1) {
            let score = dot(scores, vertex);
            if score > best_score {
                best = index;
                best_score = score;
            }
        }

        (best, &self.vertices[best])
    }
}

struct ActiveProjection {
    active: Vec<usize>,
    weights: Vec<f64>,
    marginal: Vec<f64>,
    dist_sq: f64,
}

fn project_active(theta: &[f64], vertices: &[Vec<f64>], active: &[usize]) -> ActiveProjection {
    let mut best: Option<ActiveProjection> = None;
    let subset_count = 1usize << active.len();

    for mask in 1..subset_count {
        let subset: Vec<usize> = active
            .iter()
            .enumerate()
            .filter_map(|(offset, &index)| ((mask & (1usize << offset)) != 0).then_some(index))
            .collect();

        if let Some(candidate) = project_subset(theta, vertices, &subset) {
            if best
                .as_ref()
                .is_none_or(|current| candidate.dist_sq < current.dist_sq)
            {
                best = Some(candidate);
            }
        }
    }

    best.expect("single active vertex projection is always feasible")
}

fn project_subset(
    theta: &[f64],
    vertices: &[Vec<f64>],
    subset: &[usize],
) -> Option<ActiveProjection> {
    let k = subset.len();
    let mut system = vec![vec![0.0; k + 1]; k + 1];
    let mut rhs = vec![0.0; k + 1];

    for (row, &i) in subset.iter().enumerate() {
        for (col, &j) in subset.iter().enumerate() {
            system[row][col] = dot(&vertices[i], &vertices[j]);
        }
        system[row][k] = 1.0;
        system[k][row] = 1.0;
        rhs[row] = dot(&vertices[i], theta);
    }
    rhs[k] = 1.0;

    let solution = solve_linear(system, rhs)?;
    let mut weights = solution[..k].to_vec();
    if weights.iter().any(|&w| w < -1e-9 || !w.is_finite()) {
        return None;
    }

    for weight in &mut weights {
        if weight.abs() < TOL {
            *weight = 0.0;
        }
    }
    let sum: f64 = weights.iter().sum();
    if sum <= 0.0 {
        return None;
    }
    for weight in &mut weights {
        *weight /= sum;
    }

    let marginal = combine(vertices, subset, &weights, theta.len());
    let dist_sq = theta
        .iter()
        .zip(&marginal)
        .map(|(&score, &mu)| {
            let diff = score - mu;
            diff * diff
        })
        .sum();

    let mut compact_active = Vec::new();
    let mut compact_weights = Vec::new();
    for (&index, &weight) in subset.iter().zip(&weights) {
        if weight > TOL {
            compact_active.push(index);
            compact_weights.push(weight);
        }
    }

    Some(ActiveProjection {
        active: compact_active,
        weights: compact_weights,
        marginal,
        dist_sq,
    })
}

fn solve_linear(mut system: Vec<Vec<f64>>, mut rhs: Vec<f64>) -> Option<Vec<f64>> {
    let n = rhs.len();

    for col in 0..n {
        let pivot =
            (col..n).max_by(|&a, &b| system[a][col].abs().total_cmp(&system[b][col].abs()))?;
        if system[pivot][col].abs() < 1e-12 {
            return None;
        }
        system.swap(col, pivot);
        rhs.swap(col, pivot);

        let pivot_value = system[col][col];
        for value in &mut system[col][col..n] {
            *value /= pivot_value;
        }
        rhs[col] /= pivot_value;

        let pivot_tail = system[col][col..n].to_vec();
        for row in 0..n {
            if row == col {
                continue;
            }
            let factor = system[row][col];
            if factor == 0.0 {
                continue;
            }
            for (value, pivot_value) in system[row][col..n].iter_mut().zip(&pivot_tail) {
                *value -= factor * pivot_value;
            }
            rhs[row] -= factor * rhs[col];
        }
    }

    Some(rhs)
}

fn prediction(
    theta: &[f64],
    marginal: Vec<f64>,
    active: &[usize],
    weights: &[f64],
) -> SparseMapPrediction {
    let conjugate = dot(theta, &marginal) - 0.5 * dot(&marginal, &marginal);
    SparseMapPrediction {
        marginal,
        active: active
            .iter()
            .zip(weights)
            .map(|(&vertex, &weight)| SparseMapWeight { vertex, weight })
            .collect(),
        conjugate,
    }
}

fn combine(vertices: &[Vec<f64>], active: &[usize], weights: &[f64], dim: usize) -> Vec<f64> {
    let mut out = vec![0.0; dim];
    for (&index, &weight) in active.iter().zip(weights) {
        for (value, &coordinate) in out.iter_mut().zip(&vertices[index]) {
            *value += weight * coordinate;
        }
    }
    out
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(&x, &y)| x * y).sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fenchel::{sparsemax, Regularizer, SquaredL2};

    fn standard_basis(n: usize) -> Vec<Vec<f64>> {
        (0..n)
            .map(|i| {
                let mut v = vec![0.0; n];
                v[i] = 1.0;
                v
            })
            .collect()
    }

    #[test]
    fn explicit_standard_basis_matches_sparsemax() {
        let theta = [2.0, 1.0, 0.1, -1.0];
        let vertices = standard_basis(theta.len());

        let pred = sparsemap_explicit(&theta, &vertices).unwrap();
        let expected = sparsemax(&theta);

        for (a, b) in pred.marginal.iter().zip(expected) {
            assert!((a - b).abs() < 1e-10, "{a} != {b}");
        }
        assert!(pred.active.len() < theta.len());
    }

    #[test]
    fn loss_matches_squared_l2_on_simplex_vertices() {
        let theta = [2.0, 1.0, 0.1];
        let target = [1.0, 0.0, 0.0];
        let vertices = standard_basis(theta.len());

        let sparsemap_loss = sparsemap_loss_explicit(&theta, &target, &vertices).unwrap();
        let sparsemax_loss = SquaredL2.loss(&theta, &target);

        assert!((sparsemap_loss - sparsemax_loss).abs() < 1e-10);
    }

    #[test]
    fn segment_projection_uses_sparse_active_combination() {
        let theta = [0.75];
        let vertices = vec![vec![0.0], vec![2.0]];

        let pred = sparsemap_explicit(&theta, &vertices).unwrap();

        assert!((pred.marginal[0] - 0.75).abs() < 1e-10);
        assert_eq!(pred.active.len(), 2);
        assert!((pred.active.iter().map(|w| w.weight).sum::<f64>() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn loss_is_zero_at_prediction() {
        let theta = [0.25, 0.75];
        let vertices = vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0]];
        let pred = sparsemap_explicit(&theta, &vertices).unwrap();

        let loss = sparsemap_loss_explicit(&theta, &pred.marginal, &vertices).unwrap();

        assert!(loss.abs() < 1e-10, "{loss}");
    }

    #[test]
    fn validates_vertex_dimensions() {
        let err = sparsemap_explicit(&[1.0, 2.0], &[vec![1.0]]).unwrap_err();
        assert!(matches!(err, Error::LengthMismatch(1, 2)));
    }
}

use crate::safetensors::{DType, Tensor2D};

/// Result of a thin Singular Value Decomposition `DeltaW = U * diag(sigma) * V^T`.
#[derive(Debug, Clone)]
pub struct SvdDecomposition {
    /// Orthonormal left singular vectors `U` of shape `(d_out, k)`
    pub u: Tensor2D,
    /// Singular values sorted in non-increasing order `sigma_1 >= sigma_2 >= ... >= sigma_k >= 0`
    pub sigma: Vec<f64>,
    /// Orthonormal right singular vectors `V` of shape `(d_in, k)` (note: `V`, not `V^T`)
    pub v: Tensor2D,
}

#[inline]
pub fn dot_slice(a: &[f64], b: &[f64]) -> f64 {
    let mut sum = 0.0;
    for (&x, &y) in a.iter().zip(b.iter()) {
        sum += x * y;
    }
    sum
}

#[inline]
pub fn axpy_slice(y: &mut [f64], a: f64, x: &[f64]) {
    for (yi, &xi) in y.iter_mut().zip(x.iter()) {
        *yi += a * xi;
    }
}

#[inline]
pub fn scale_slice(x: &mut [f64], alpha: f64) {
    for xi in x.iter_mut() {
        *xi *= alpha;
    }
}

/// Transpose a `Tensor2D` `(rows, cols)` into `(cols, rows)`.
pub fn transpose(m: &Tensor2D) -> Tensor2D {
    let mut out = Tensor2D::zeros(m.cols, m.rows, m.dtype);
    for r in 0..m.rows {
        let row_offset = r * m.cols;
        for c in 0..m.cols {
            out.data[c * m.rows + r] = m.data[row_offset + c];
        }
    }
    out
}

/// Multiply two `Tensor2D` matrices `A (m x k)` and `B (k x n)` -> `C (m x n)`.
pub fn matmul(a: &Tensor2D, b: &Tensor2D) -> Tensor2D {
    assert_eq!(
        a.cols, b.rows,
        "matmul dimension mismatch: ({}, {}) x ({}, {})",
        a.rows, a.cols, b.rows, b.cols
    );
    let m = a.rows;
    let k = a.cols;
    let n = b.cols;
    let mut out = Tensor2D::zeros(m, n, a.dtype);

    for i in 0..m {
        let a_row = &a.data[i * k..(i + 1) * k];
        let out_row = &mut out.data[i * n..(i + 1) * n];
        for p in 0..k {
            let a_ip = a_row[p];
            if a_ip == 0.0 {
                continue;
            }
            let b_row = &b.data[p * n..(p + 1) * n];
            axpy_slice(out_row, a_ip, b_row);
        }
    }
    out
}

/// Compute thin QR factorization `M = Q * R` using Modified Gram-Schmidt with
/// DGKS (Daniel-Gragg-Kaufman-Stewart) re-orthogonalization.
///
/// Input `m` has shape `(rows, cols)` with `rows >= cols`.
/// Returns `(q_cols, r_mat)` where `q_cols` is a list of `cols` orthonormal column vectors
/// each of length `rows`, and `r_mat` is `(cols, cols)` upper triangular.
pub fn thin_qr_columns(m: &Tensor2D) -> (Vec<Vec<f64>>, Tensor2D) {
    let rows = m.rows;
    let cols = m.cols;
    let mut q_cols: Vec<Vec<f64>> = vec![vec![0.0; rows]; cols];
    for r in 0..rows {
        let row_slice = &m.data[r * cols..(r + 1) * cols];
        for c in 0..cols {
            q_cols[c][r] = row_slice[c];
        }
    }

    let mut r_mat = Tensor2D::zeros(cols, cols, DType::F32);

    for j in 0..cols {
        // First Modified Gram-Schmidt pass
        for i in 0..j {
            let proj = dot_slice(&q_cols[i], &q_cols[j]);
            r_mat.set(i, j, proj);
            let qi = q_cols[i].clone();
            axpy_slice(&mut q_cols[j], -proj, &qi);
        }
        // Second DGKS re-orthogonalization pass for numerical stability near machine epsilon
        for i in 0..j {
            let corr = dot_slice(&q_cols[i], &q_cols[j]);
            if corr.abs() > 1e-16 {
                r_mat.set(i, j, r_mat.get(i, j) + corr);
                let qi = q_cols[i].clone();
                axpy_slice(&mut q_cols[j], -corr, &qi);
            }
        }

        let norm = dot_slice(&q_cols[j], &q_cols[j]).sqrt();
        if norm > 1e-14 {
            r_mat.set(j, j, norm);
            scale_slice(&mut q_cols[j], 1.0 / norm);
        } else {
            r_mat.set(j, j, 0.0);
            q_cols[j].fill(0.0);
            // Pick a deterministic unit vector orthogonal to previous columns
            for basis_idx in 0..rows {
                q_cols[j].fill(0.0);
                q_cols[j][basis_idx] = 1.0;
                for i in 0..j {
                    let proj = dot_slice(&q_cols[i], &q_cols[j]);
                    let qi = q_cols[i].clone();
                    axpy_slice(&mut q_cols[j], -proj, &qi);
                }
                let bnorm = dot_slice(&q_cols[j], &q_cols[j]).sqrt();
                if bnorm > 1e-7 {
                    scale_slice(&mut q_cols[j], 1.0 / bnorm);
                    break;
                }
            }
        }
    }

    (q_cols, r_mat)
}

/// Multiply column-stored orthonormal matrix `Q` (`rows x r`) by a small matrix `W` (`r x k`)
/// producing a row-major `Tensor2D` of shape `(rows, k)`.
pub fn multiply_q_cols_by_small(
    q_cols: &[Vec<f64>],
    rows: usize,
    w: &Tensor2D,
    k: usize,
    dtype: DType,
) -> Tensor2D {
    let r = q_cols.len();
    assert_eq!(w.rows, r);
    assert!(k <= w.cols);

    // Compute in column-major first for contiguous vector AXPY, then write row-major
    let mut out_cols: Vec<Vec<f64>> = vec![vec![0.0; rows]; k];
    for j in 0..k {
        for p in 0..r {
            let weight = w.get(p, j);
            if weight != 0.0 {
                axpy_slice(&mut out_cols[j], weight, &q_cols[p]);
            }
        }
    }

    let mut out = Tensor2D::zeros(rows, k, dtype);
    for row in 0..rows {
        let base = row * k;
        for col in 0..k {
            out.data[base + col] = out_cols[col][row];
        }
    }
    out
}

/// High-accuracy One-Sided Jacobi SVD on a matrix `M` of shape `(m, n)`.
/// If `m < n`, internally decomposes `M^T` and swaps `U` and `V`.
pub fn jacobi_svd(mat: &Tensor2D) -> SvdDecomposition {
    if mat.rows < mat.cols {
        let mt = transpose(mat);
        let svd_t = jacobi_svd_tall(&mt);
        return SvdDecomposition {
            u: svd_t.v,
            sigma: svd_t.sigma,
            v: svd_t.u,
        };
    }
    jacobi_svd_tall(mat)
}

fn jacobi_svd_tall(mat: &Tensor2D) -> SvdDecomposition {
    let m = mat.rows;
    let n = mat.cols;
    assert!(m >= n);

    // Work on columns of W (initially copy of mat) and V (initially n x n identity)
    let mut w_cols: Vec<Vec<f64>> = vec![vec![0.0; m]; n];
    for r in 0..m {
        for c in 0..n {
            w_cols[c][r] = mat.get(r, c);
        }
    }

    let mut v_cols: Vec<Vec<f64>> = vec![vec![0.0; n]; n];
    for i in 0..n {
        v_cols[i][i] = 1.0;
    }

    let max_sweeps = 60;
    let eps = 1e-15;

    for _sweep in 0..max_sweeps {
        let mut max_off = 0.0_f64;

        for p in 0..n {
            for q in (p + 1)..n {
                let alpha = dot_slice(&w_cols[p], &w_cols[p]);
                let beta = dot_slice(&w_cols[q], &w_cols[q]);
                let gamma = dot_slice(&w_cols[p], &w_cols[q]);

                let denom = (alpha * beta).sqrt();
                if denom > 0.0 {
                    let rel = gamma.abs() / denom;
                    if rel > max_off {
                        max_off = rel;
                    }
                }

                if gamma.abs() <= eps * denom {
                    continue;
                }

                // Compute Jacobi rotation angle that orthogonalizes columns p and q
                let zeta = (beta - alpha) / (2.0 * gamma);
                let t = if zeta >= 0.0 {
                    1.0 / (zeta + (1.0 + zeta * zeta).sqrt())
                } else {
                    -1.0 / (-zeta + (1.0 + zeta * zeta).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = c * t;

                // Rotate w_cols[p] and w_cols[q]
                for i in 0..m {
                    let wp = w_cols[p][i];
                    let wq = w_cols[q][i];
                    w_cols[p][i] = c * wp - s * wq;
                    w_cols[q][i] = s * wp + c * wq;
                }

                // Rotate v_cols[p] and v_cols[q]
                for i in 0..n {
                    let vp = v_cols[p][i];
                    let vq = v_cols[q][i];
                    v_cols[p][i] = c * vp - s * vq;
                    v_cols[q][i] = s * vp + c * vq;
                }
            }
        }

        if max_off < eps {
            break;
        }
    }

    // Extract norms (singular values) and sort descending
    let mut norms: Vec<(usize, f64)> = (0..n)
        .map(|j| (j, dot_slice(&w_cols[j], &w_cols[j]).sqrt()))
        .collect();
    norms.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut sigma = Vec::with_capacity(n);
    let mut u_cols: Vec<Vec<f64>> = Vec::with_capacity(n);
    let mut v_sorted_cols: Vec<Vec<f64>> = Vec::with_capacity(n);

    for (orig_idx, s_val) in norms {
        sigma.push(s_val);
        v_sorted_cols.push(v_cols[orig_idx].clone());
        if s_val > 1e-14 {
            let mut u_col = w_cols[orig_idx].clone();
            scale_slice(&mut u_col, 1.0 / s_val);
            // Orthogonalize against previously accepted u_cols for strict numerical cleanliness
            for prev in &u_cols {
                let proj = dot_slice(prev, &u_col);
                if proj.abs() > 1e-15 {
                    axpy_slice(&mut u_col, -proj, prev);
                }
            }
            let renorm = dot_slice(&u_col, &u_col).sqrt();
            if renorm > 1e-14 {
                scale_slice(&mut u_col, 1.0 / renorm);
            }
            u_cols.push(u_col);
        } else {
            // Find an orthogonal completion vector
            let mut u_col = vec![0.0; m];
            for basis in 0..m {
                u_col.fill(0.0);
                u_col[basis] = 1.0;
                for prev in &u_cols {
                    let proj = dot_slice(prev, &u_col);
                    axpy_slice(&mut u_col, -proj, prev);
                }
                let renorm = dot_slice(&u_col, &u_col).sqrt();
                if renorm > 1e-7 {
                    scale_slice(&mut u_col, 1.0 / renorm);
                    break;
                }
            }
            u_cols.push(u_col);
        }
    }

    let mut u = Tensor2D::zeros(m, n, mat.dtype);
    for r in 0..m {
        for c in 0..n {
            u.set(r, c, u_cols[c][r]);
        }
    }

    let mut v = Tensor2D::zeros(n, n, mat.dtype);
    for r in 0..n {
        for c in 0..n {
            v.set(r, c, v_sorted_cols[c][r]);
        }
    }

    SvdDecomposition { u, sigma, v }
}

/// Fast QR-Core Singular Value Spectrum for a low-rank LoRA product `DeltaW = B * A`.
///
/// Because `Q_B` (`d_out x r`) and `Q_A` (`d_in x r`) have orthonormal columns,
/// the non-zero singular values of `B * A` (`d_out x d_in`) are identically the
/// singular values of the tiny `r x r` core matrix `M = R_B * R_A^T`.
pub fn qr_core_singular_values(b: &Tensor2D, a: &Tensor2D) -> Vec<f64> {
    let (_qb_cols, rb) = thin_qr_columns(b);
    let at = transpose(a);
    let (_qa_cols, ra) = thin_qr_columns(&at);
    let rat = transpose(&ra);
    let core = matmul(&rb, &rat);
    let core_svd = jacobi_svd(&core);
    core_svd.sigma
}

/// Full QR-Core SVD returning the top `k` left/right singular vectors and all `r` singular values.
///
/// Complexity: `O(r^2 * (d_out + d_in) + r^3 + r * k * (d_out + d_in))` instead of `O(d_out * d_in * min(d_out, d_in))`.
pub fn qr_core_svd_truncated(b: &Tensor2D, a: &Tensor2D, k: usize) -> SvdDecomposition {
    let r = a.rows;
    assert_eq!(b.cols, r);
    let keep = k.min(r).max(1);

    let (qb_cols, rb) = thin_qr_columns(b);
    let at = transpose(a);
    let (qa_cols, ra) = thin_qr_columns(&at);
    let rat = transpose(&ra);
    let core = matmul(&rb, &rat);
    let core_svd = jacobi_svd(&core);

    let u_trunc = multiply_q_cols_by_small(&qb_cols, b.rows, &core_svd.u, keep, b.dtype);
    let v_trunc = multiply_q_cols_by_small(&qa_cols, a.cols, &core_svd.v, keep, a.dtype);

    SvdDecomposition {
        u: u_trunc,
        sigma: core_svd.sigma,
        v: v_trunc,
    }
}

/// Convert a truncated `SvdDecomposition` (with `keep = u.cols`) into balanced LoRA factors
/// `B_k = U_k * sqrt(Sigma_k)` (`d_out x keep`) and `A_k = sqrt(Sigma_k) * V_k^T` (`keep x d_in`).
pub fn svd_to_lora_factors(svd: &SvdDecomposition, keep: usize, dtype: DType) -> (Tensor2D, Tensor2D) {
    let k = keep.min(svd.u.cols);
    let d_out = svd.u.rows;
    let d_in = svd.v.rows;

    let mut b_new = Tensor2D::zeros(d_out, k, dtype);
    let mut a_new = Tensor2D::zeros(k, d_in, dtype);

    let sqrt_s: Vec<f64> = (0..k).map(|i| svd.sigma[i].max(0.0).sqrt()).collect();

    for row in 0..d_out {
        for j in 0..k {
            b_new.set(row, j, svd.u.get(row, j) * sqrt_s[j]);
        }
    }

    for j in 0..k {
        let s = sqrt_s[j];
        for col in 0..d_in {
            a_new.set(j, col, s * svd.v.get(col, j));
        }
    }

    (b_new, a_new)
}

use ndarray::{Array1, s};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArrayDyn};
use pyo3::prelude::*;
use rayon::prelude::*;

/// Result type for functions returning a pair of f32 arrays
type ArrayPairResult<'py> = PyResult<(Bound<'py, PyArray1<f32>>, Bound<'py, PyArray1<f32>>)>;

#[pyfunction]
pub fn accumulate_router_stats_cpu<'py>(
    py: Python<'py>,
    idx: PyReadonlyArrayDyn<'py, i64>,
    gates: PyReadonlyArrayDyn<'py, f32>,
    num_experts: usize,
) -> ArrayPairResult<'py> {
    let idx_arr = idx.as_array();
    let gates_arr = gates.as_array();

    // idx and gates are (B, T, k)
    let shape = idx_arr.shape();
    if shape.len() != 3 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "idx must be 3D (B, T, k)",
        ));
    }
    if gates_arr.shape() != shape {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "gates must have the same shape as idx {shape:?}, got {:?}",
            gates_arr.shape()
        )));
    }
    if num_experts == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "num_experts must be positive",
        ));
    }

    // Parallel reduction over batches
    let (counts, prob_sums) = accumulate_router_stats_3d(idx_arr, gates_arr, num_experts);

    let counts_arr = Array1::from(counts);
    let probs_arr = Array1::from(prob_sums);

    Ok((counts_arr.into_pyarray(py), probs_arr.into_pyarray(py)))
}

pub(crate) fn accumulate_router_stats_3d(
    idx_arr: ndarray::ArrayViewD<'_, i64>,
    gates_arr: ndarray::ArrayViewD<'_, f32>,
    num_experts: usize,
) -> (Vec<f32>, Vec<f32>) {
    let shape = idx_arr.shape();
    let b_dim = shape[0];
    let t_dim = shape[1];
    let k_dim = shape[2];

    (0..b_dim)
        .into_par_iter()
        .map(|b| {
            let mut local_counts = vec![0.0f32; num_experts];
            let mut local_probs = vec![0.0f32; num_experts];

            for t in 0..t_dim {
                let idx_bt = idx_arr.slice(s![b, t, ..]);
                let gates_bt = gates_arr.slice(s![b, t, ..]);

                for k in 0..k_dim {
                    let raw = idx_bt[k];
                    if raw < 0 {
                        continue;
                    }
                    let e = raw as usize;
                    let g = gates_bt[k];

                    if e < num_experts {
                        local_probs[e] += g;

                        let mut seen = false;
                        for prev_k in 0..k {
                            if idx_bt[prev_k] == raw {
                                seen = true;
                                break;
                            }
                        }

                        if !seen {
                            local_counts[e] += 1.0;
                        }
                    }
                }
            }
            (local_counts, local_probs)
        })
        .reduce(
            || (vec![0.0; num_experts], vec![0.0; num_experts]),
            |mut a, b| {
                for i in 0..num_experts {
                    a.0[i] += b.0[i];
                    a.1[i] += b.1[i];
                }
                a
            },
        )
}

pub(crate) fn update_metabolism_slices(
    f: &[f32],
    e: &[f32],
    af: &[f32],
    ae: &[f32],
    u: &[f32],
    f_new: &mut [f32],
    e_new: &mut [f32],
) {
    f_new
        .par_iter_mut()
        .zip(e_new.par_iter_mut())
        .enumerate()
        .for_each(|(i, (fn_val, en_val))| {
            let fv = f[i];
            let ev = e[i];
            let afv = af[i];
            let aev = ae[i];
            let uv = u[i];

            *fn_val = fv * (1.0 - afv) + afv * uv;
            *en_val = ev * (1.0 - aev) + aev * (1.0 - uv);
        });
}

#[pyfunction]
pub fn update_metabolism_cpu<'py>(
    py: Python<'py>,
    fatigue: PyReadonlyArrayDyn<'py, f32>,
    energy: PyReadonlyArrayDyn<'py, f32>,
    alpha_fatigue: PyReadonlyArrayDyn<'py, f32>,
    alpha_energy: PyReadonlyArrayDyn<'py, f32>,
    util: PyReadonlyArrayDyn<'py, f32>,
) -> ArrayPairResult<'py> {
    // Ensure 1D
    let f = fatigue
        .as_array()
        .into_dimensionality::<ndarray::Ix1>()
        .map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("fatigue must be 1D: {}", e))
        })?;
    let e = energy
        .as_array()
        .into_dimensionality::<ndarray::Ix1>()
        .map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("energy must be 1D: {}", e))
        })?;
    let af = alpha_fatigue
        .as_array()
        .into_dimensionality::<ndarray::Ix1>()
        .map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("alpha_fatigue must be 1D: {}", e))
        })?;
    let ae = alpha_energy
        .as_array()
        .into_dimensionality::<ndarray::Ix1>()
        .map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("alpha_energy must be 1D: {}", e))
        })?;
    let u = util
        .as_array()
        .into_dimensionality::<ndarray::Ix1>()
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("util must be 1D: {}", e)))?;

    let size = f.len();
    for (name, actual) in [
        ("energy", e.len()),
        ("alpha_fatigue", af.len()),
        ("alpha_energy", ae.len()),
        ("util", u.len()),
    ] {
        if actual != size {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "{name} length must match fatigue length {size}, got {actual}"
            )));
        }
    }
    let mut f_new = Array1::<f32>::zeros(size);
    let mut e_new = Array1::<f32>::zeros(size);

    let f_slice = f_new.as_slice_mut().ok_or_else(|| {
        pyo3::exceptions::PyRuntimeError::new_err(
            "internal fatigue output allocation was not contiguous",
        )
    })?;
    let e_slice = e_new.as_slice_mut().ok_or_else(|| {
        pyo3::exceptions::PyRuntimeError::new_err(
            "internal energy output allocation was not contiguous",
        )
    })?;

    let f_cow = f.as_standard_layout();
    let e_cow = e.as_standard_layout();
    let af_cow = af.as_standard_layout();
    let ae_cow = ae.as_standard_layout();
    let u_cow = u.as_standard_layout();
    update_metabolism_slices(
        f_cow.as_slice().expect("standard layout"),
        e_cow.as_slice().expect("standard layout"),
        af_cow.as_slice().expect("standard layout"),
        ae_cow.as_slice().expect("standard layout"),
        u_cow.as_slice().expect("standard layout"),
        f_slice,
        e_slice,
    );

    Ok((f_new.into_pyarray(py), e_new.into_pyarray(py)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    #[test]
    fn accumulate_router_stats_deduplicates_counts_and_sums_gates() {
        // Shape (B=1, T=2, K=3), 4 experts
        // Token 0 selects [1, 1, 2] with gates [0.3, 0.2, 0.5] -> expert 1 count=1, prob=0.5; expert 2 count=1, prob=0.5
        // Token 1 selects [0, 2, 99] with gates [0.4, 0.6, 1.0] -> expert 0 count=1, prob=0.4; expert 2 count=1, prob=0.6; 99 ignored
        let idx = Array3::from_shape_vec((1, 2, 3), vec![1i64, 1, 2, 0, 2, 99])
            .unwrap()
            .into_dyn();
        let gates = Array3::from_shape_vec((1, 2, 3), vec![0.3f32, 0.2, 0.5, 0.4, 0.6, 1.0])
            .unwrap()
            .into_dyn();

        let (counts, probs) = accumulate_router_stats_3d(idx.view(), gates.view(), 4);
        assert_eq!(counts, vec![1.0, 1.0, 2.0, 0.0]);
        assert!((probs[0] - 0.4).abs() < 1e-6);
        assert!((probs[1] - 0.5).abs() < 1e-6);
        assert!((probs[2] - 1.1).abs() < 1e-6);
        assert_eq!(probs[3], 0.0);
    }

    #[test]
    fn update_metabolism_computes_expected_ema() {
        let f = [0.2f32, 0.8];
        let e = [0.9f32, 0.4];
        let af = [0.1f32, 0.25];
        let ae = [0.05f32, 0.5];
        let u = [0.6f32, 0.2];
        let mut f_new = [0.0f32; 2];
        let mut e_new = [0.0f32; 2];

        update_metabolism_slices(&f, &e, &af, &ae, &u, &mut f_new, &mut e_new);
        assert!((f_new[0] - (0.2 * 0.9 + 0.1 * 0.6)).abs() < 1e-6);
        assert!((f_new[1] - (0.8 * 0.75 + 0.25 * 0.2)).abs() < 1e-6);
        assert!((e_new[0] - (0.9 * 0.95 + 0.05 * 0.4)).abs() < 1e-6);
        assert!((e_new[1] - (0.4 * 0.5 + 0.5 * 0.8)).abs() < 1e-6);
    }
}

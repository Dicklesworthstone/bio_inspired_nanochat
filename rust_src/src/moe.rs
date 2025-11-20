use ndarray::{Array1, s};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArrayDyn};
use pyo3::prelude::*;
use rayon::prelude::*;

#[pyfunction]
pub fn accumulate_router_stats_cpu<'py>(
    py: Python<'py>,
    idx: PyReadonlyArrayDyn<'py, i64>,
    gates: PyReadonlyArrayDyn<'py, f32>,
    num_experts: usize,
) -> PyResult<(Bound<'py, PyArray1<f32>>, Bound<'py, PyArray1<f32>>)> {
    let idx_arr = idx.as_array();
    let gates_arr = gates.as_array();
    
    // idx and gates are (B, T, k)
    let shape = idx_arr.shape();
    if shape.len() != 3 {
        return Err(pyo3::exceptions::PyValueError::new_err("idx must be 3D (B, T, k)"));
    }
    let b_dim = shape[0];
    let t_dim = shape[1];
    let k_dim = shape[2];
    
    // Parallel reduction over batches
    let (counts, prob_sums) = (0..b_dim).into_par_iter().map(|b| {
        let mut local_counts = vec![0.0f32; num_experts];
        let mut local_probs = vec![0.0f32; num_experts];
        
        for t in 0..t_dim {
            // Use slice to avoid temporary view drop issues
            let idx_bt = idx_arr.slice(s![b, t, ..]);
            let gates_bt = gates_arr.slice(s![b, t, ..]);
            
            // To match Python "any(dim=-1)" logic for counts, we must not double count experts
            // if they appear multiple times in the same top-k (unlikely in real top-k, but possible in random test)
            // Since k is small, a simple check is fine.
            // We can use a stack-allocated small array or bitmask if experts are small, but here we just loop.
            
            for k in 0..k_dim {
                let e = idx_bt[k] as usize;
                let g = gates_bt[k];
                
                if e < num_experts {
                    local_probs[e] += g;
                    
                    // Check if we already counted this expert for this token
                    let mut seen = false;
                    for prev_k in 0..k {
                        if (idx_bt[prev_k] as usize) == e {
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
    }).reduce(
        || (vec![0.0; num_experts], vec![0.0; num_experts]),
        |mut a, b| {
            for i in 0..num_experts {
                a.0[i] += b.0[i];
                a.1[i] += b.1[i];
            }
            a
        }
    );
    
    let counts_arr = Array1::from(counts);
    let probs_arr = Array1::from(prob_sums);
    
    Ok((counts_arr.into_pyarray(py), probs_arr.into_pyarray(py)))
}

#[pyfunction]
pub fn update_metabolism_cpu<'py>(
    py: Python<'py>,
    fatigue: PyReadonlyArrayDyn<'py, f32>,
    energy: PyReadonlyArrayDyn<'py, f32>,
    alpha_fatigue: PyReadonlyArrayDyn<'py, f32>,
    alpha_energy: PyReadonlyArrayDyn<'py, f32>,
    util: PyReadonlyArrayDyn<'py, f32>,
) -> PyResult<(Bound<'py, PyArray1<f32>>, Bound<'py, PyArray1<f32>>)> {
    // Ensure 1D
    let f = fatigue.as_array().into_dimensionality::<ndarray::Ix1>()
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("fatigue must be 1D: {}", e)))?;
    let e = energy.as_array().into_dimensionality::<ndarray::Ix1>()
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("energy must be 1D: {}", e)))?;
    let af = alpha_fatigue.as_array().into_dimensionality::<ndarray::Ix1>()
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("alpha_fatigue must be 1D: {}", e)))?;
    let ae = alpha_energy.as_array().into_dimensionality::<ndarray::Ix1>()
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("alpha_energy must be 1D: {}", e)))?;
    let u = util.as_array().into_dimensionality::<ndarray::Ix1>()
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("util must be 1D: {}", e)))?;
    
    let size = f.len();
    let mut f_new = Array1::<f32>::zeros(size);
    let mut e_new = Array1::<f32>::zeros(size);
    
    // Use slice iteration for parallel processing to avoid Zip limit (max 6)
    let f_slice = f_new.as_slice_mut().unwrap();
    let e_slice = e_new.as_slice_mut().unwrap();
    
    f_slice.par_iter_mut().zip(e_slice.par_iter_mut()).enumerate().for_each(|(i, (fn_val, en_val))| {
        let fv = f[i];
        let ev = e[i];
        let afv = af[i];
        let aev = ae[i];
        let uv = u[i];
        
        *fn_val = fv * (1.0 - afv) + afv * uv;
        *en_val = ev * (1.0 - aev) + aev * (1.0 - uv);
    });
        
    Ok((f_new.into_pyarray(py), e_new.into_pyarray(py)))
}

use ndarray::{Array3, Array4, s};
use numpy::{IntoPyArray, PyArrayDyn, PyReadonlyArrayDyn};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use rayon::prelude::*;

#[derive(Clone, Copy)]
struct Config {
    tau_c: f32,
    tau_buf: f32,
    tau_prime: f32,
    tau_rrp: f32,
    tau_energy: f32,
    alpha_ca: f32,
    alpha_buf_on: f32,
    alpha_buf_off: f32,
    alpha_prime: f32,
    alpha_unprime: f32,
    alpha_refill: f32,
    energy_in: f32,
    energy_cost_rel: f32,
    energy_cost_pump: f32,
    syt_fast_kd: f32,
    syt_slow_kd: f32,
    complexin_bias: f32,
    qmax: f32,
    q_beta: f32,
    barrier_strength: f32,
    epsilon: f32,
}

impl Config {
    fn from_py(cfg: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self {
            tau_c: cfg.getattr("tau_c")?.extract()?,
            tau_buf: cfg.getattr("tau_buf")?.extract()?,
            tau_prime: cfg.getattr("tau_prime")?.extract()?,
            tau_rrp: cfg.getattr("tau_rrp")?.extract()?,
            tau_energy: cfg.getattr("tau_energy")?.extract()?,
            alpha_ca: cfg.getattr("alpha_ca")?.extract()?,
            alpha_buf_on: cfg.getattr("alpha_buf_on")?.extract()?,
            alpha_buf_off: cfg.getattr("alpha_buf_off")?.extract()?,
            alpha_prime: cfg.getattr("alpha_prime")?.extract()?,
            alpha_unprime: cfg.getattr("alpha_unprime")?.extract()?,
            alpha_refill: cfg.getattr("alpha_refill")?.extract()?,
            energy_in: cfg.getattr("energy_in")?.extract()?,
            energy_cost_rel: cfg.getattr("energy_cost_rel")?.extract()?,
            energy_cost_pump: cfg.getattr("energy_cost_pump")?.extract()?,
            syt_fast_kd: cfg.getattr("syt_fast_kd")?.extract()?,
            syt_slow_kd: cfg.getattr("syt_slow_kd")?.extract()?,
            complexin_bias: cfg.getattr("complexin_bias")?.extract()?,
            qmax: cfg.getattr("qmax")?.extract()?,
            q_beta: cfg.getattr("q_beta")?.extract()?,
            barrier_strength: cfg.getattr("barrier_strength")?.extract()?,
            epsilon: cfg.getattr("epsilon")?.extract()?,
        })
    }
}

#[inline]
fn softplus(x: f32) -> f32 {
    if x > 20.0 { x } else { x.exp().ln_1p() }
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[pyfunction]
pub fn presyn_step_cpu<'py>(
    py: Python<'py>,
    q: PyReadonlyArrayDyn<'py, f32>,
    k: PyReadonlyArrayDyn<'py, f32>,
    logits: PyReadonlyArrayDyn<'py, f32>,
    state: Bound<'py, PyDict>,
    cfg_obj: Bound<'py, PyAny>,
) -> PyResult<(Bound<'py, PyArrayDyn<f32>>, Bound<'py, PyDict>)> {
    let c = Config::from_py(&cfg_obj)?;

    let q_arr = q
        .as_array()
        .into_dimensionality::<ndarray::Ix4>()
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("q must be 4D: {}", e)))?;
    let k_arr = k
        .as_array()
        .into_dimensionality::<ndarray::Ix4>()
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("k must be 4D: {}", e)))?;
    let logits_arr = logits
        .as_array()
        .into_dimensionality::<ndarray::Ix4>()
        .map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("logits must be 4D: {}", e))
        })?;

    let shape = q_arr.shape();
    let b_dim = shape[0];
    let h_dim = shape[1];
    let t_dim = shape[2];
    let d_dim = shape[3];

    // Extract state tensors
    let c_tensor = state
        .get_item("C")?
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Missing state key 'C'"))?
        .extract::<PyReadonlyArrayDyn<f32>>()?;
    let buf_tensor = state
        .get_item("BUF")?
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Missing state key 'BUF'"))?
        .extract::<PyReadonlyArrayDyn<f32>>()?;
    let rrp_tensor = state
        .get_item("RRP")?
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Missing state key 'RRP'"))?
        .extract::<PyReadonlyArrayDyn<f32>>()?;
    let res_tensor = state
        .get_item("RES")?
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Missing state key 'RES'"))?
        .extract::<PyReadonlyArrayDyn<f32>>()?;
    let pr_tensor = state
        .get_item("PR")?
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Missing state key 'PR'"))?
        .extract::<PyReadonlyArrayDyn<f32>>()?;
    let cl_tensor = state
        .get_item("CL")?
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Missing state key 'CL'"))?
        .extract::<PyReadonlyArrayDyn<f32>>()?;
    let e_tensor = state
        .get_item("E")?
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Missing state key 'E'"))?
        .extract::<PyReadonlyArrayDyn<f32>>()?;

    let c_arr = c_tensor
        .as_array()
        .into_dimensionality::<ndarray::Ix3>()
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("C must be 3D: {}", e)))?;
    let buf_arr = buf_tensor
        .as_array()
        .into_dimensionality::<ndarray::Ix3>()
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("BUF must be 3D: {}", e)))?;
    let rrp_arr = rrp_tensor
        .as_array()
        .into_dimensionality::<ndarray::Ix3>()
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("RRP must be 3D: {}", e)))?;
    let res_arr = res_tensor
        .as_array()
        .into_dimensionality::<ndarray::Ix3>()
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("RES must be 3D: {}", e)))?;
    let pr_arr = pr_tensor
        .as_array()
        .into_dimensionality::<ndarray::Ix3>()
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("PR must be 3D: {}", e)))?;
    let cl_arr = cl_tensor
        .as_array()
        .into_dimensionality::<ndarray::Ix3>()
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("CL must be 3D: {}", e)))?;
    let e_arr = e_tensor
        .as_array()
        .into_dimensionality::<ndarray::Ix3>()
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("E must be 3D: {}", e)))?;

    // Initialize outputs
    let mut syn_logit = Array4::<f32>::zeros((b_dim, h_dim, t_dim, t_dim));
    let mut c_new = Array3::<f32>::zeros((b_dim, h_dim, t_dim));
    let mut buf_new = Array3::<f32>::zeros((b_dim, h_dim, t_dim));
    let mut rrp_new = Array3::<f32>::zeros((b_dim, h_dim, t_dim));
    let mut res_new = Array3::<f32>::zeros((b_dim, h_dim, t_dim));
    let mut pr_new = Array3::<f32>::zeros((b_dim, h_dim, t_dim));
    let mut e_new = Array3::<f32>::zeros((b_dim, h_dim, t_dim));

    // Precompute constants
    let rho_c = (-1.0 / c.tau_c).exp();
    let rho_b = (-1.0 / c.tau_buf).exp();
    let rho_p = (-1.0 / c.tau_prime).exp();
    let rho_r = (-1.0 / c.tau_rrp).exp();
    let rho_e = (-1.0 / c.tau_energy).exp();
    let sqrt_d = (d_dim as f32).sqrt();

    // Unsafe pointers and strides for output mutation in parallel
    let syn_logit_ptr = syn_logit.as_mut_ptr() as usize;
    let c_new_ptr = c_new.as_mut_ptr() as usize;
    let buf_new_ptr = buf_new.as_mut_ptr() as usize;
    let rrp_new_ptr = rrp_new.as_mut_ptr() as usize;
    let res_new_ptr = res_new.as_mut_ptr() as usize;
    let pr_new_ptr = pr_new.as_mut_ptr() as usize;
    let e_new_ptr = e_new.as_mut_ptr() as usize;

    let stride_logits = t_dim * t_dim;
    let stride_state = t_dim;

    (0..b_dim).into_par_iter().for_each(move |b| {
        (0..h_dim).into_par_iter().for_each(move |h| {
            let offset_idx = b * h_dim + h;

            // Reconstruct mutable slices via unsafe pointers
            // Safety: unique (b,h) index implies disjoint memory regions
            let syn_logit_slice = unsafe {
                std::slice::from_raw_parts_mut(
                    (syn_logit_ptr as *mut f32).add(offset_idx * stride_logits),
                    stride_logits,
                )
            };
            let c_new_slice = unsafe {
                std::slice::from_raw_parts_mut(
                    (c_new_ptr as *mut f32).add(offset_idx * stride_state),
                    stride_state,
                )
            };
            let buf_new_slice = unsafe {
                std::slice::from_raw_parts_mut(
                    (buf_new_ptr as *mut f32).add(offset_idx * stride_state),
                    stride_state,
                )
            };
            let rrp_new_slice = unsafe {
                std::slice::from_raw_parts_mut(
                    (rrp_new_ptr as *mut f32).add(offset_idx * stride_state),
                    stride_state,
                )
            };
            let res_new_slice = unsafe {
                std::slice::from_raw_parts_mut(
                    (res_new_ptr as *mut f32).add(offset_idx * stride_state),
                    stride_state,
                )
            };
            let pr_new_slice = unsafe {
                std::slice::from_raw_parts_mut(
                    (pr_new_ptr as *mut f32).add(offset_idx * stride_state),
                    stride_state,
                )
            };
            let e_new_slice = unsafe {
                std::slice::from_raw_parts_mut(
                    (e_new_ptr as *mut f32).add(offset_idx * stride_state),
                    stride_state,
                )
            };

            // Read-only views using slicing (ArrayView is Copy)
            let q_bh = q_arr.slice(s![b, h, .., ..]);
            let k_bh = k_arr.slice(s![b, h, .., ..]);
            let logits_bh = logits_arr.slice(s![b, h, .., ..]);

            let c_bh = c_arr.slice(s![b, h, ..]);
            let buf_bh = buf_arr.slice(s![b, h, ..]);
            let rrp_bh = rrp_arr.slice(s![b, h, ..]);
            let res_bh = res_arr.slice(s![b, h, ..]);
            let pr_bh = pr_arr.slice(s![b, h, ..]);
            let cl_bh = cl_arr.slice(s![b, h, ..]);
            let e_bh = e_arr.slice(s![b, h, ..]);

            // Local buffers for temporal integration
            let mut influx = Vec::with_capacity(t_dim);
            for t in 0..t_dim {
                let mut sum_drive = 0.0;
                for j in 0..=t {
                    let val = logits_bh[[t, j]];
                    let clamped = val.clamp(-20.0, 20.0);
                    let drive = softplus(clamped);
                    sum_drive += drive;
                }
                influx.push(sum_drive / ((t + 1) as f32));
            }

            let mut c_new_vec = Vec::with_capacity(t_dim);
            let mut buf_new_vec = Vec::with_capacity(t_dim);
            let mut rrp_refill_vec = Vec::with_capacity(t_dim);
            let mut pr_mid_vec = Vec::with_capacity(t_dim);
            let mut res_mid_vec = Vec::with_capacity(t_dim);
            let mut e_mid_vec = Vec::with_capacity(t_dim);

            for t in 0..t_dim {
                let c_val = c_bh[t];
                let buf_val = buf_bh[t];
                let inf = influx[t];

                let c_next = rho_c * c_val + c.alpha_ca * inf
                    - c.alpha_buf_on * c_val * (1.0 - buf_val)
                    + c.alpha_buf_off * buf_val;
                let buf_next = rho_b * buf_val + c.alpha_buf_on * c_val * (1.0 - buf_val)
                    - c.alpha_buf_off * buf_val;

                c_new_vec.push(c_next.max(0.0));
                buf_new_vec.push(buf_next.clamp(0.0, 1.0));

                let pr_val = pr_bh[t];
                let rrp_val = rrp_bh[t];
                let res_val = res_bh[t];
                let e_val = e_bh[t];

                let pr_mid = (rho_p * pr_val + c.alpha_prime * (1.0 - pr_val)).clamp(0.0, 1.0);
                let rrp_refill = (rho_r * rrp_val + c.alpha_refill * res_val).clamp(0.0, 1.0);
                let res_mid = (res_val - c.alpha_refill * res_val).clamp(0.0, 1.0);
                let e_mid = (rho_e * e_val + c.energy_in).clamp(0.0, 1.6);

                pr_mid_vec.push(pr_mid);
                rrp_refill_vec.push(rrp_refill);
                res_mid_vec.push(res_mid);
                e_mid_vec.push(e_mid);
            }

            let mut release_frac_bh = vec![vec![0.0; t_dim]; t_dim];
            let mut used_rrp = vec![0.0; t_dim];

            for t in 0..t_dim {
                let q_t = q_bh.slice(s![t, ..]);

                let c_val = c_new_vec[t];
                let fast = c_val / (c_val + c.syt_fast_kd);
                let slow = c_val / (c_val + c.syt_slow_kd);
                let syt = 0.7 * fast + 0.3 * slow;

                let pr_m = pr_mid_vec[t];
                let cl_val = cl_bh[t];

                let fuse_logit_base = 3.0 * syt + 2.0 * pr_m - 2.0 * (cl_val + c.complexin_bias);
                let fuse_base = sigmoid(fuse_logit_base);

                let mut raw_release_row = Vec::with_capacity(t + 1);
                let mut row_sum = 0.0;

                for j in 0..=t {
                    let k_j = k_bh.slice(s![j, ..]);
                    // Manually compute dot product
                    let dot: f32 = q_t.iter().zip(k_j.iter()).map(|(a, b)| a * b).sum();
                    let d_bilin = sigmoid(dot / sqrt_d);

                    let fuse_p = fuse_base * d_bilin;
                    let avail = rrp_refill_vec[t];

                    let rr = (fuse_p * avail).clamp(0.0, 1.0);
                    raw_release_row.push(rr);
                    row_sum += rr;
                }

                let avail = rrp_refill_vec[t];
                let scale = if row_sum > c.epsilon {
                    (avail / row_sum).min(1.0)
                } else {
                    1.0
                };

                let mut used = 0.0;
                for j in 0..=t {
                    let rel = raw_release_row[j] * scale;
                    release_frac_bh[t][j] = rel;
                    used += rel;
                }
                used_rrp[t] = used;
            }

            for t in 0..t_dim {
                let used = used_rrp[t];
                let rrp_n = (rrp_refill_vec[t] - used).clamp(0.0, 1.0);
                let res_n = (res_mid_vec[t] + used).clamp(0.0, 1.0);
                let pr_n = (pr_mid_vec[t] - c.alpha_unprime * used).clamp(0.0, 1.0);
                let e_n =
                    (e_mid_vec[t] - c.energy_cost_rel * used - c.energy_cost_pump * (1.0 - res_n))
                        .clamp(0.0, 1.6);

                let qamp = sigmoid(c.q_beta * (e_n - 0.5)) * c.qmax;

                // Write back state
                c_new_slice[t] = c_new_vec[t];
                buf_new_slice[t] = buf_new_vec[t];
                rrp_new_slice[t] = rrp_n;
                res_new_slice[t] = res_n;
                pr_new_slice[t] = pr_n;
                e_new_slice[t] = e_n;

                // Syn Logit
                for j in 0..=t {
                    let rel = release_frac_bh[t][j];
                    let dist = ((t as f32) - (j as f32)).abs() / (t_dim.max(1) as f32);
                    let val = (rel * qamp).max(c.epsilon).ln() - c.barrier_strength * dist;

                    syn_logit_slice[t * t_dim + j] = val;
                }
                for j in (t + 1)..t_dim {
                    syn_logit_slice[t * t_dim + j] = c.epsilon.ln();
                }
            }
        });
    });

    let out_dict = PyDict::new(py);
    out_dict.set_item("C", c_new.into_pyarray(py))?;
    out_dict.set_item("BUF", buf_new.into_pyarray(py))?;
    out_dict.set_item("RRP", rrp_new.into_pyarray(py))?;
    out_dict.set_item("RES", res_new.into_pyarray(py))?;
    out_dict.set_item("PR", pr_new.into_pyarray(py))?;
    out_dict.set_item("CL", cl_tensor.as_array().to_owned().into_pyarray(py))?;
    out_dict.set_item("E", e_new.into_pyarray(py))?;

    Ok((
        syn_logit.into_dyn().into_pyarray(py).to_owned(),
        out_dict.to_owned(),
    ))
}

/// Parameters for the deterministic, sparse canonical decode equation.
///
/// This is intentionally separate from `Config`: `presyn_step_cpu` above is the historical dense
/// landing kernel, whereas this contract mirrors `SynapticPresyn.release_canonical` for Tq == 1.
#[derive(Clone, Copy)]
struct CanonicalConfig {
    rho_c: f32,
    rho_b: f32,
    alpha_ca: f32,
    alpha_buf_on: f32,
    alpha_buf_off: f32,
    syt_fast_kd: f32,
    syt_slow_kd: f32,
    doc2_gain: f32,
    complexin_bias: f32,
    q_beta: f32,
    qmax: f32,
    prime_rate: f32,
    unprime_per_release: f32,
    nsf_recover: f32,
    rec_rate: f32,
    energy_fill: f32,
    energy_max: f32,
    energy_use: f32,
    endo_delay: usize,
}

impl CanonicalConfig {
    fn from_py(cfg: &Bound<'_, PyAny>) -> PyResult<Self> {
        let tau_c: f32 = cfg.getattr("tau_c")?.extract()?;
        let tau_buf: f32 = cfg.getattr("tau_buf")?.extract()?;
        let stochastic_train_frac: f32 = cfg.getattr("stochastic_train_frac")?.extract()?;
        let metriplectic_integrator: bool = cfg.getattr("metriplectic_integrator")?.extract()?;
        let learnable_kinetics: bool = cfg.getattr("learnable_kinetics")?.extract()?;
        if stochastic_train_frac != 0.0 || metriplectic_integrator || learnable_kinetics {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "canonical Rust decode supports deterministic fixed-kinetics mode only",
            ));
        }
        Ok(Self {
            rho_c: (-1.0 / tau_c).exp(),
            rho_b: (-1.0 / tau_buf).exp(),
            alpha_ca: cfg.getattr("alpha_ca")?.extract()?,
            alpha_buf_on: cfg.getattr("alpha_buf_on")?.extract()?,
            alpha_buf_off: cfg.getattr("alpha_buf_off")?.extract()?,
            syt_fast_kd: cfg.getattr("syt_fast_kd")?.extract()?,
            syt_slow_kd: cfg.getattr("syt_slow_kd")?.extract()?,
            doc2_gain: cfg.getattr("doc2_gain")?.extract()?,
            complexin_bias: cfg.getattr("complexin_bias")?.extract()?,
            q_beta: cfg.getattr("q_beta")?.extract()?,
            qmax: cfg.getattr("qmax")?.extract()?,
            prime_rate: cfg.getattr("prime_rate")?.extract()?,
            unprime_per_release: cfg.getattr("unprime_per_release")?.extract()?,
            nsf_recover: cfg.getattr("nsf_recover")?.extract()?,
            rec_rate: cfg.getattr("rec_rate")?.extract()?,
            energy_fill: cfg.getattr("energy_fill")?.extract()?,
            energy_max: cfg.getattr("energy_max")?.extract()?,
            energy_use: cfg.getattr("energy_use")?.extract()?,
            endo_delay: cfg.getattr("endo_delay")?.extract()?,
        })
    }
}

fn state_array3(state: &Bound<'_, PyDict>, name: &str) -> PyResult<Array3<f32>> {
    let tensor = state
        .get_item(name)?
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err(format!("missing state[{name:?}]")))?
        .extract::<PyReadonlyArrayDyn<f32>>()?;
    tensor
        .as_array()
        .into_dimensionality::<ndarray::Ix3>()
        .map(|array| array.to_owned())
        .map_err(|error| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "state[{name:?}] must be a float32 rank-3 array: {error}"
            ))
        })
}

fn ensure_shape(name: &str, actual: &[usize], expected: &[usize]) -> PyResult<()> {
    if actual == expected {
        Ok(())
    } else {
        Err(pyo3::exceptions::PyValueError::new_err(format!(
            "{name} must have shape {expected:?}, got {actual:?}"
        )))
    }
}

/// Canonical sparse presynaptic CPU step for deterministic one-query decode.
///
/// The function consumes NumPy float32/int64/bool arrays so the fixture is language agnostic.  It
/// implements the same gather -> per-edge release -> duplicate-safe scatter -> whole-key state
/// advancement order as Python and the live Triton kernel.  Eval keeps `ema_e` frozen; callers
/// receive the release tensor already normalized by that scalar.
#[pyfunction]
pub fn presyn_release_canonical_cpu<'py>(
    py: Python<'py>,
    drive: PyReadonlyArrayDyn<'py, f32>,
    idx: PyReadonlyArrayDyn<'py, i64>,
    valid: PyReadonlyArrayDyn<'py, bool>,
    state: Bound<'py, PyDict>,
    cfg_obj: Bound<'py, PyAny>,
    ema_e: f32,
) -> PyResult<(Bound<'py, PyArrayDyn<f32>>, Bound<'py, PyDict>)> {
    if !ema_e.is_finite() || ema_e < 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "ema_e must be finite and non-negative",
        ));
    }
    let cfg = CanonicalConfig::from_py(&cfg_obj)?;
    let drive = drive
        .as_array()
        .into_dimensionality::<ndarray::Ix4>()
        .map_err(|error| {
            pyo3::exceptions::PyValueError::new_err(format!("drive must be rank 4: {error}"))
        })?;
    let idx = idx
        .as_array()
        .into_dimensionality::<ndarray::Ix4>()
        .map_err(|error| {
            pyo3::exceptions::PyValueError::new_err(format!("idx must be rank 4: {error}"))
        })?;
    let valid = valid
        .as_array()
        .into_dimensionality::<ndarray::Ix4>()
        .map_err(|error| {
            pyo3::exceptions::PyValueError::new_err(format!("valid must be rank 4: {error}"))
        })?;
    let drive_shape = drive.shape();
    if drive_shape[2] != 1 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "canonical Rust decode requires Tq == 1, got {}",
            drive_shape[2]
        )));
    }
    ensure_shape("idx", idx.shape(), drive_shape)?;
    ensure_shape("valid", valid.shape(), drive_shape)?;

    let batch = drive_shape[0];
    let heads = drive_shape[1];
    let topk = drive_shape[3];
    let mut calcium = state_array3(&state, "C")?;
    let mut buffer = state_array3(&state, "BUF")?;
    let mut rrp = state_array3(&state, "RRP")?;
    let mut reserve = state_array3(&state, "RES")?;
    let mut priming = state_array3(&state, "PR")?;
    let mut complexin = state_array3(&state, "CL")?;
    let amplitude = state_array3(&state, "AMP")?;
    let mut energy = state_array3(&state, "E")?;
    let state_shape = calcium.shape().to_vec();
    if state_shape[0] != batch || state_shape[1] != heads {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "state batch/head dimensions must be ({batch}, {heads}), got {:?}",
            &state_shape[..2]
        )));
    }
    let key_count = state_shape[2];
    for (name, array) in [
        ("BUF", &buffer),
        ("RRP", &rrp),
        ("RES", &reserve),
        ("PR", &priming),
        ("CL", &complexin),
        ("AMP", &amplitude),
        ("E", &energy),
    ] {
        ensure_shape(name, array.shape(), &state_shape)?;
    }

    let delay_object = state
        .get_item("DELAY")?
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("missing state[\"DELAY\"]"))?;
    let delay_list = delay_object.cast::<PyList>()?;
    if delay_list.len() != cfg.endo_delay {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "DELAY must contain {} arrays, got {}",
            cfg.endo_delay,
            delay_list.len()
        )));
    }
    let mut delay = Vec::with_capacity(cfg.endo_delay);
    for (position, item) in delay_list.iter().enumerate() {
        let array = item.extract::<PyReadonlyArrayDyn<f32>>()?;
        let array = array
            .as_array()
            .into_dimensionality::<ndarray::Ix3>()
            .map_err(|error| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "DELAY[{position}] must be rank 3: {error}"
                ))
            })?
            .to_owned();
        ensure_shape(&format!("DELAY[{position}]"), array.shape(), &state_shape)?;
        delay.push(array);
    }

    for &selected in &idx {
        if selected < 0 || selected as usize >= key_count {
            return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "idx value {selected} is outside [0, {key_count})"
            )));
        }
    }

    let previous_calcium = calcium.clone();
    let previous_buffer = buffer.clone();
    let previous_rrp = rrp.clone();
    let previous_reserve = reserve.clone();
    let previous_priming = priming.clone();
    let previous_complexin = complexin.clone();
    let previous_energy = energy.clone();
    let mut released_by_key = Array3::<f32>::zeros((batch, heads, key_count));
    let mut drive_by_key = Array3::<f32>::zeros((batch, heads, key_count));
    let mut accessed = Array3::<bool>::from_elem((batch, heads, key_count), false);
    let mut release = Array4::<f32>::zeros((batch, heads, 1, topk));

    for b in 0..batch {
        for h in 0..heads {
            for edge in 0..topk {
                let key = idx[[b, h, 0, edge]] as usize;
                let edge_valid = valid[[b, h, 0, edge]];
                let edge_drive = drive[[b, h, 0, edge]];
                let c_prev = previous_calcium[[b, h, key]];
                let buf_prev = previous_buffer[[b, h, key]];
                let c_edge = (cfg.rho_c * c_prev + cfg.alpha_ca * softplus(edge_drive)
                    - cfg.alpha_buf_on * c_prev * (1.0 - buf_prev)
                    + cfg.alpha_buf_off * buf_prev)
                    .max(0.0);
                let fast = c_edge / (c_edge + cfg.syt_fast_kd);
                let slow = c_edge / (c_edge + cfg.syt_slow_kd);
                let syt = 0.7 * fast + 0.3 * slow + cfg.doc2_gain * sigmoid(4.0 * (c_edge - 0.12));
                let fuse_base = sigmoid(
                    3.0 * syt + 2.0 * previous_priming[[b, h, key]]
                        - 2.0 * (previous_complexin[[b, h, key]] + cfg.complexin_bias),
                );
                let probability = (fuse_base * sigmoid(edge_drive)).clamp(0.0, 1.0);
                let released = if edge_valid {
                    probability * previous_rrp[[b, h, key]]
                } else {
                    0.0
                };
                let qamp = sigmoid(cfg.q_beta * (previous_energy[[b, h, key]] - 0.5)) * cfg.qmax;
                release[[b, h, 0, edge]] = released * qamp / (ema_e + 1e-6);
                if edge_valid {
                    released_by_key[[b, h, key]] += released;
                    drive_by_key[[b, h, key]] += edge_drive;
                    accessed[[b, h, key]] = true;
                }
            }
        }
    }

    let mut next_delay = if cfg.endo_delay > 0 {
        delay[1..].to_vec()
    } else {
        Vec::new()
    };
    let mut delay_tail = Array3::<f32>::zeros((batch, heads, key_count));
    for b in 0..batch {
        for h in 0..heads {
            for key in 0..key_count {
                let c_prev = previous_calcium[[b, h, key]];
                let buf_prev = previous_buffer[[b, h, key]];
                let release_sum = released_by_key[[b, h, key]];
                let influx = if accessed[[b, h, key]] {
                    cfg.alpha_ca * softplus(drive_by_key[[b, h, key]])
                } else {
                    0.0
                };
                calcium[[b, h, key]] = (cfg.rho_c * c_prev + influx
                    - cfg.alpha_buf_on * c_prev * (1.0 - buf_prev)
                    + cfg.alpha_buf_off * buf_prev)
                    .max(0.0);
                buffer[[b, h, key]] = (cfg.rho_b * buf_prev
                    + cfg.alpha_buf_on * c_prev * (1.0 - buf_prev)
                    - cfg.alpha_buf_off * buf_prev)
                    .clamp(0.0, 1.0);
                let rrp_depleted = (previous_rrp[[b, h, key]] - release_sum).max(0.0);
                let reserve_refilled = if cfg.endo_delay > 0 {
                    previous_reserve[[b, h, key]] + delay[0][[b, h, key]]
                } else {
                    previous_reserve[[b, h, key]]
                };
                let take = reserve_refilled.min(1.0);
                reserve[[b, h, key]] = (reserve_refilled - cfg.prime_rate * take).max(0.0);
                rrp[[b, h, key]] = (rrp_depleted + cfg.prime_rate * take).clamp(0.0, 30.0);
                priming[[b, h, key]] = (previous_priming[[b, h, key]]
                    * (1.0 - cfg.unprime_per_release * release_sum)
                    + cfg.nsf_recover * (1.0 - previous_priming[[b, h, key]]))
                .clamp(0.0, 1.0);
                complexin[[b, h, key]] = (previous_complexin[[b, h, key]] * 0.995 + 0.005
                    - cfg.unprime_per_release * release_sum)
                    .clamp(0.0, 1.0);
                energy[[b, h, key]] = (previous_energy[[b, h, key]]
                    + cfg.energy_fill * (cfg.energy_max - previous_energy[[b, h, key]])
                    - cfg.energy_use * release_sum)
                    .clamp(0.0, cfg.energy_max);
                if cfg.endo_delay > 0 {
                    delay_tail[[b, h, key]] = release_sum * cfg.rec_rate;
                }
            }
        }
    }
    if cfg.endo_delay > 0 {
        next_delay.push(delay_tail);
    }

    let out = PyDict::new(py);
    out.set_item("C", calcium.into_pyarray(py))?;
    out.set_item("BUF", buffer.into_pyarray(py))?;
    out.set_item("RRP", rrp.into_pyarray(py))?;
    out.set_item("RES", reserve.into_pyarray(py))?;
    out.set_item("PR", priming.into_pyarray(py))?;
    out.set_item("CL", complexin.into_pyarray(py))?;
    out.set_item("AMP", amplitude.into_pyarray(py))?;
    out.set_item("E", energy.into_pyarray(py))?;
    let delay_output = PyList::empty(py);
    for entry in next_delay {
        delay_output.append(entry.into_pyarray(py))?;
    }
    out.set_item("DELAY", delay_output)?;

    Ok((
        release.into_dyn().into_pyarray(py).to_owned(),
        out.to_owned(),
    ))
}

use ndarray::{ArrayD, ArrayViewD, Axis};
use numpy::{IntoPyArray, PyArrayDyn, PyReadonlyArrayDyn};
use pyo3::prelude::*;
use pyo3::types::PyDict;
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
    (1.0 + x.exp()).ln()
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
    
    let q_arr = q.as_array();
    let k_arr = k.as_array();
    let logits_arr = logits.as_array();
    
    let shape = q_arr.shape();
    let b_dim = shape[0];
    let h_dim = shape[1];
    let t_dim = shape[2];
    let d_dim = shape[3];

    // Extract state tensors
    let c_tensor = state.get_item("C")?.unwrap().extract::<PyReadonlyArrayDyn<f32>>()?;
    let buf_tensor = state.get_item("BUF")?.unwrap().extract::<PyReadonlyArrayDyn<f32>>()?;
    let rrp_tensor = state.get_item("RRP")?.unwrap().extract::<PyReadonlyArrayDyn<f32>>()?;
    let res_tensor = state.get_item("RES")?.unwrap().extract::<PyReadonlyArrayDyn<f32>>()?;
    let pr_tensor = state.get_item("PR")?.unwrap().extract::<PyReadonlyArrayDyn<f32>>()?;
    let cl_tensor = state.get_item("CL")?.unwrap().extract::<PyReadonlyArrayDyn<f32>>()?;
    let e_tensor = state.get_item("E")?.unwrap().extract::<PyReadonlyArrayDyn<f32>>()?;

    let c_arr = c_tensor.as_array();
    let buf_arr = buf_tensor.as_array();
    let rrp_arr = rrp_tensor.as_array();
    let res_arr = res_tensor.as_array();
    let pr_arr = pr_tensor.as_array();
    let cl_arr = cl_tensor.as_array();
    let e_arr = e_tensor.as_array();

    // Initialize outputs with zeros
    let mut syn_logit = ArrayD::<f32>::zeros(logits_arr.shape());
    let mut c_new = ArrayD::<f32>::zeros(c_arr.shape());
    let mut buf_new = ArrayD::<f32>::zeros(buf_arr.shape());
    let mut rrp_new = ArrayD::<f32>::zeros(rrp_arr.shape());
    let mut res_new = ArrayD::<f32>::zeros(res_arr.shape());
    let mut pr_new = ArrayD::<f32>::zeros(pr_arr.shape());
    let mut e_new = ArrayD::<f32>::zeros(e_arr.shape());

    // Precompute constants
    let rho_c = (-1.0 / c.tau_c).exp();
    let rho_b = (-1.0 / c.tau_buf).exp();
    let rho_p = (-1.0 / c.tau_prime).exp();
    let rho_r = (-1.0 / c.tau_rrp).exp();
    let rho_e = (-1.0 / c.tau_energy).exp();
    let sqrt_d = (d_dim as f32).sqrt();

    // Wrapper for raw pointers to allow passing to rayon threads
    #[derive(Clone, Copy)]
    struct UnsafeArray<T>(*mut ArrayD<T>);
    unsafe impl<T> Sync for UnsafeArray<T> {}
    unsafe impl<T> Send for UnsafeArray<T> {}
    
    let syn_logit_ptr: UnsafeArray<f32> = UnsafeArray(&mut syn_logit as *mut _);
    let c_new_ptr: UnsafeArray<f32> = UnsafeArray(&mut c_new as *mut _);
    let buf_new_ptr: UnsafeArray<f32> = UnsafeArray(&mut buf_new as *mut _);
    let rrp_new_ptr: UnsafeArray<f32> = UnsafeArray(&mut rrp_new as *mut _);
    let res_new_ptr: UnsafeArray<f32> = UnsafeArray(&mut res_new as *mut _);
    let pr_new_ptr: UnsafeArray<f32> = UnsafeArray(&mut pr_new as *mut _);
    let e_new_ptr: UnsafeArray<f32> = UnsafeArray(&mut e_new as *mut _);

    (0..b_dim).into_par_iter().for_each(move |b| {
        (0..h_dim).into_par_iter().for_each(move |h| {
            // Reconstruct pointers with explicit types
            let syn_logit_p: *mut ArrayD<f32> = syn_logit_ptr.0;
            let c_new_p: *mut ArrayD<f32> = c_new_ptr.0;
            let buf_new_p: *mut ArrayD<f32> = buf_new_ptr.0;
            let rrp_new_p: *mut ArrayD<f32> = rrp_new_ptr.0;
            let res_new_p: *mut ArrayD<f32> = res_new_ptr.0;
            let pr_new_p: *mut ArrayD<f32> = pr_new_ptr.0;
            let e_new_p: *mut ArrayD<f32> = e_new_ptr.0;

            // Read-only slices
            let q_bh = q_arr.index_axis(Axis(0), b).index_axis(Axis(0), h);
            let k_bh = k_arr.index_axis(Axis(0), b).index_axis(Axis(0), h);
            let logits_bh = logits_arr.index_axis(Axis(0), b).index_axis(Axis(0), h);
            
            let c_bh = c_arr.index_axis(Axis(0), b).index_axis(Axis(0), h);
            let buf_bh = buf_arr.index_axis(Axis(0), b).index_axis(Axis(0), h);
            let rrp_bh = rrp_arr.index_axis(Axis(0), b).index_axis(Axis(0), h);
            let res_bh = res_arr.index_axis(Axis(0), b).index_axis(Axis(0), h);
            let pr_bh = pr_arr.index_axis(Axis(0), b).index_axis(Axis(0), h);
            let cl_bh = cl_arr.index_axis(Axis(0), b).index_axis(Axis(0), h);
            let e_bh = e_arr.index_axis(Axis(0), b).index_axis(Axis(0), h);

            // Local buffers
            let mut influx = Vec::with_capacity(t_dim);
            for t in 0..t_dim {
                let mut sum_drive = 0.0;
                for j in 0..=t {
                    let val = logits_bh[[t, j]];
                    let clamped = val.max(-20.0).min(20.0);
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
                
                let c_next = rho_c * c_val + c.alpha_ca * inf - c.alpha_buf_on * c_val * (1.0 - buf_val) + c.alpha_buf_off * buf_val;
                let buf_next = rho_b * buf_val + c.alpha_buf_on * c_val * (1.0 - buf_val) - c.alpha_buf_off * buf_val;
                
                c_new_vec.push(c_next.max(0.0));
                buf_new_vec.push(buf_next.max(0.0).min(1.0));
                
                let pr_val = pr_bh[t];
                let rrp_val = rrp_bh[t];
                let res_val = res_bh[t];
                let e_val = e_bh[t];
                
                let pr_mid = (rho_p * pr_val + c.alpha_prime * (1.0 - pr_val)).max(0.0).min(1.0);
                let rrp_refill = (rho_r * rrp_val + c.alpha_refill * res_val).max(0.0).min(1.0);
                let res_mid = (res_val - c.alpha_refill * res_val).max(0.0).min(1.0);
                let e_mid = (rho_e * e_val + c.energy_in).max(0.0).min(1.6);
                
                pr_mid_vec.push(pr_mid);
                rrp_refill_vec.push(rrp_refill);
                res_mid_vec.push(res_mid);
                e_mid_vec.push(e_mid);
            }
            
            let mut release_frac_bh = vec![vec![0.0; t_dim]; t_dim];
            let mut used_rrp = vec![0.0; t_dim];
            
            for t in 0..t_dim {
                let q_t = q_bh.index_axis(Axis(0), t);
                
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
                    let k_j = k_bh.index_axis(Axis(0), j);
                    let dot: f32 = q_t.iter().zip(k_j.iter()).map(|(a, b)| a * b).sum();
                    let d_bilin = sigmoid(dot / sqrt_d);
                    
                    let fuse_p = fuse_base * d_bilin;
                    let avail = rrp_refill_vec[t];
                    
                    let rr = (fuse_p * avail).max(0.0).min(1.0);
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
                let rrp_n = (rrp_refill_vec[t] - used).max(0.0).min(1.0);
                let res_n = (res_mid_vec[t] + used).max(0.0).min(1.0);
                let pr_n = (pr_mid_vec[t] - c.alpha_unprime * used).max(0.0).min(1.0);
                let e_n = (e_mid_vec[t] - c.energy_cost_rel * used - c.energy_cost_pump * (1.0 - res_n)).max(0.0).min(1.6);
                
                let qamp = sigmoid(c.q_beta * (e_n - 0.5)) * c.qmax;
                
                unsafe {
                    let idx = [b, h, t];
                    (*c_new_p).uget_mut(idx).clone_from(&c_new_vec[t]);
                    (*buf_new_p).uget_mut(idx).clone_from(&buf_new_vec[t]);
                    (*rrp_new_p).uget_mut(idx).clone_from(&rrp_n);
                    (*res_new_p).uget_mut(idx).clone_from(&res_n);
                    (*pr_new_p).uget_mut(idx).clone_from(&pr_n);
                    (*e_new_p).uget_mut(idx).clone_from(&e_n);
                }
                
                for j in 0..=t {
                    let rel = release_frac_bh[t][j];
                    let dist = ((t as f32) - (j as f32)).abs() / (t_dim.max(1) as f32);
                    let val = (rel * qamp).max(c.epsilon).ln() - c.barrier_strength * dist;
                    
                    unsafe {
                        (*syn_logit_p).uget_mut([b, h, t, j]).clone_from(&val);
                    }
                }
                for j in (t+1)..t_dim {
                     unsafe {
                        (*syn_logit_p).uget_mut([b, h, t, j]).clone_from(&c.epsilon.ln());
                    }
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

    Ok((syn_logit.into_pyarray(py).to_owned(), out_dict.to_owned()))
}

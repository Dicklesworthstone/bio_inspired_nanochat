use ndarray::{Array3, Array4};
use numpy::{IntoPyArray, PyArrayDyn, PyReadonlyArrayDyn};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

#[inline]
fn softplus(x: f32) -> f32 {
    if x > 20.0 { x } else { x.exp().ln_1p() }
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Parameters for the deterministic, sparse canonical decode equation.
///
/// This contract mirrors `SynapticPresyn.release_canonical` for Tq == 1.
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
        let _stochastic_train_frac: f32 = cfg.getattr("stochastic_train_frac")?.extract()?;
        let metriplectic_integrator: bool = cfg.getattr("metriplectic_integrator")?.extract()?;
        let learnable_kinetics: bool = cfg.getattr("learnable_kinetics")?.extract()?;
        // metriplectic/learnable genuinely change the dynamics — still rejected.
        // stochastic_train_frac does NOT: the Python canonical gates the
        // stochastic branch on `(train or mc_sampling) and frac > 0`, and this
        // kernel only serves eval (train=False, enforced by
        // _can_use_native_presyn_decode) — so eval is deterministic regardless
        // of the fraction. Refusing it forced operators to flip a global config
        // just to run CPU evaluation.
        if metriplectic_integrator || learnable_kinetics {
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
    ema_e: &Bound<'py, PyAny>,
) -> PyResult<(Bound<'py, PyArrayDyn<f32>>, Bound<'py, PyDict>)> {
    // Accept a Python float OR a 1-element tensor/array (the live caller holds
    // `self.ema_e` as an nn buffer). `.item()` covers both without forcing the
    // caller to remember `.item()` — which would add a device sync per step.
    let ema_e = match ema_e.extract::<f64>() {
        Ok(value) => value as f32,
        Err(_) => ema_e.call_method0("item")?.extract::<f64>()? as f32,
    };
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

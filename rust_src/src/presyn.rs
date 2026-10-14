use ndarray::{Array3, Array4};
use numpy::{IntoPyArray, PyArrayDyn, PyReadonlyArrayDyn, PyUntypedArrayMethods};
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
pub(crate) struct CanonicalConfig {
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
    lambda_loge: f32,
    epsilon: f32,
    loge_bias_clamp: f32,
}

type CanonicalConfigTuple = [f32; 22];

impl CanonicalConfig {
    fn from_py(cfg: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(tup_attr) = cfg.getattr("_rust_canonical_tuple")
            && let Ok(t) = tup_attr.extract::<CanonicalConfigTuple>()
        {
            return Ok(Self {
                rho_c: t[0],
                rho_b: t[1],
                alpha_ca: t[2],
                alpha_buf_on: t[3],
                alpha_buf_off: t[4],
                syt_fast_kd: t[5],
                syt_slow_kd: t[6],
                doc2_gain: t[7],
                complexin_bias: t[8],
                q_beta: t[9],
                qmax: t[10],
                prime_rate: t[11],
                unprime_per_release: t[12],
                nsf_recover: t[13],
                rec_rate: t[14],
                energy_fill: t[15],
                energy_max: t[16],
                energy_use: t[17],
                endo_delay: t[18] as usize,
                lambda_loge: t[19],
                epsilon: t[20],
                loge_bias_clamp: t[21],
            });
        }
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
            lambda_loge: cfg
                .getattr("lambda_loge")
                .and_then(|v| v.extract())
                .unwrap_or(1.0),
            epsilon: cfg
                .getattr("epsilon")
                .and_then(|v| v.extract())
                .unwrap_or(1e-6),
            loge_bias_clamp: cfg
                .getattr("loge_bias_clamp")
                .and_then(|v| v.extract())
                .unwrap_or(0.0),
        })
    }
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

const L1_TILE: usize = 512;

#[derive(Clone, Copy, Default)]
struct TouchedKey {
    key: usize,
    release_sum: f32,
    drive_sum: f32,
    c_prev: f32,
    buf_prev: f32,
    rrp_prev: f32,
    res_prev: f32,
    pr_prev: f32,
    cl_prev: f32,
    e_prev: f32,
    d0_prev: f32,
}

/// Pure-Rust single-(batch, head)-row canonical decode step.
///
/// First gathers pre-update state at the `topk` selected keys and records unique touched keys'
/// pre-update values in a tiny stack/vec buffer (`<= topk` entries). Second runs L1-tiled,
/// branchless, transcendental-free SIMD loops over `0..key_count` for passive background
/// decay/recovery. Third applies the exact active-influx and vesicle-depletion update at the
/// `<= topk` touched keys using their saved pre-update state. This eliminates all `key_count`-sized
/// scratch allocations and removes `softplus` from the `0..key_count` loop so LLVM auto-vectorizes it.
#[allow(clippy::too_many_arguments)]
pub(crate) fn canonical_decode_row(
    drive_row: &[f32],
    idx_row: &[i64],
    valid_row: Option<&[bool]>,
    c_row: &mut [f32],
    buf_row: &mut [f32],
    rrp_row: &mut [f32],
    res_row: &mut [f32],
    pr_row: &mut [f32],
    cl_row: &mut [f32],
    e_row: &mut [f32],
    mut d0_row: Option<&mut [f32]>,
    release_row: &mut [f32],
    mut logits_row: Option<&mut [f32]>,
    barrier_scale: f32,
    cfg: &CanonicalConfig,
    ema_denom: f32,
) {
    let topk = drive_row.len();
    let key_count = c_row.len();
    let mut stack_touched = [TouchedKey::default(); 64];
    let mut vec_touched: Vec<TouchedKey> = if topk > 64 {
        Vec::with_capacity(topk)
    } else {
        Vec::new()
    };
    let mut n_touched = 0usize;

    for edge in 0..topk {
        let edge_drive = drive_row[edge];
        let is_valid = match valid_row {
            Some(v) => v[edge],
            None => edge_drive.is_finite(),
        };
        if !is_valid {
            release_row[edge] = 0.0;
            continue;
        }
        let key = idx_row[edge] as usize;
        let c_prev = c_row[key];
        let buf_prev = buf_row[key];
        let c_edge = (cfg.rho_c * c_prev + cfg.alpha_ca * softplus(edge_drive)
            - cfg.alpha_buf_on * c_prev * (1.0 - buf_prev)
            + cfg.alpha_buf_off * buf_prev)
            .max(0.0);
        let fast = c_edge / (c_edge + cfg.syt_fast_kd);
        let slow = c_edge / (c_edge + cfg.syt_slow_kd);
        let syt = 0.7 * fast + 0.3 * slow + cfg.doc2_gain * sigmoid(4.0 * (c_edge - 0.12));
        let fuse_base =
            sigmoid(3.0 * syt + 2.0 * pr_row[key] - 2.0 * (cl_row[key] + cfg.complexin_bias));
        let probability = (fuse_base * sigmoid(edge_drive)).clamp(0.0, 1.0);
        let released = probability * rrp_row[key];
        let qamp = sigmoid(cfg.q_beta * (e_row[key] - 0.5)) * cfg.qmax;
        let rel_val = released * qamp / ema_denom;
        release_row[edge] = rel_val;
        if let Some(l_row) = logits_row.as_deref_mut() {
            let mut bias = cfg.lambda_loge * (cfg.epsilon + rel_val).ln();
            if cfg.loge_bias_clamp > 0.0 {
                bias = bias.clamp(-cfg.loge_bias_clamp, cfg.loge_bias_clamp);
            }
            l_row[key] += bias;
        }
        let slice: &mut [TouchedKey] = if topk <= 64 {
            &mut stack_touched[..n_touched]
        } else {
            &mut vec_touched[..]
        };
        if let Some(existing) = slice.iter_mut().find(|entry| entry.key == key) {
            existing.release_sum += released;
            existing.drive_sum += edge_drive;
        } else {
            let entry = TouchedKey {
                key,
                release_sum: released,
                drive_sum: edge_drive,
                c_prev,
                buf_prev,
                rrp_prev: rrp_row[key],
                res_prev: res_row[key],
                pr_prev: pr_row[key],
                cl_prev: cl_row[key],
                e_prev: e_row[key],
                d0_prev: d0_row.as_deref().map_or(0.0, |d| d[key]),
            };
            if topk <= 64 {
                stack_touched[n_touched] = entry;
                n_touched += 1;
            } else {
                vec_touched.push(entry);
            }
        }
    }

    if barrier_scale != 0.0
        && let Some(l_row) = logits_row
    {
        let last_k = key_count.saturating_sub(1) as f32;
        for (k, logit) in l_row[..key_count].iter_mut().enumerate() {
            *logit -= (last_k - (k as f32)) * barrier_scale;
        }
    }

    let mut tile_start = 0usize;
    while tile_start < key_count {
        let tile_end = (tile_start + L1_TILE).min(key_count);
        for (c, buf) in c_row[tile_start..tile_end]
            .iter_mut()
            .zip(buf_row[tile_start..tile_end].iter_mut())
        {
            let c_prev = *c;
            let buf_prev = *buf;
            let on_term = cfg.alpha_buf_on * c_prev * (1.0 - buf_prev);
            let off_term = cfg.alpha_buf_off * buf_prev;
            *c = (cfg.rho_c * c_prev - on_term + off_term).max(0.0);
            *buf = (cfg.rho_b * buf_prev + on_term - off_term).clamp(0.0, 1.0);
        }

        if let Some(d0_slice) = d0_row.as_deref_mut() {
            for ((res, rrp), d0) in res_row[tile_start..tile_end]
                .iter_mut()
                .zip(rrp_row[tile_start..tile_end].iter_mut())
                .zip(d0_slice[tile_start..tile_end].iter_mut())
            {
                let rrp_depleted = (*rrp).max(0.0);
                let reserve_refilled = *res + *d0;
                let take = reserve_refilled.min(1.0);
                let primed = cfg.prime_rate * take;
                *res = (reserve_refilled - primed).max(0.0);
                *rrp = (rrp_depleted + primed).clamp(0.0, 30.0);
                *d0 = 0.0;
            }
        } else {
            for (res, rrp) in res_row[tile_start..tile_end]
                .iter_mut()
                .zip(rrp_row[tile_start..tile_end].iter_mut())
            {
                let rrp_depleted = (*rrp).max(0.0);
                let reserve_refilled = *res;
                let take = reserve_refilled.min(1.0);
                let primed = cfg.prime_rate * take;
                *res = (reserve_refilled - primed).max(0.0);
                *rrp = (rrp_depleted + primed).clamp(0.0, 30.0);
            }
        }

        for ((pr, cl), e) in pr_row[tile_start..tile_end]
            .iter_mut()
            .zip(cl_row[tile_start..tile_end].iter_mut())
            .zip(e_row[tile_start..tile_end].iter_mut())
        {
            let pr_prev = *pr;
            *pr = (pr_prev + cfg.nsf_recover * (1.0 - pr_prev)).clamp(0.0, 1.0);
            *cl = (*cl * 0.995 + 0.005).clamp(0.0, 1.0);
            let e_prev = *e;
            *e = (e_prev + cfg.energy_fill * (cfg.energy_max - e_prev)).clamp(0.0, cfg.energy_max);
        }

        tile_start = tile_end;
    }

    let touched_slice: &[TouchedKey] = if topk <= 64 {
        &stack_touched[..n_touched]
    } else {
        &vec_touched[..]
    };
    for t in touched_slice {
        let key = t.key;
        let release_sum = t.release_sum;
        let influx = cfg.alpha_ca * softplus(t.drive_sum);
        let c_prev = t.c_prev;
        let buf_prev = t.buf_prev;
        c_row[key] = (cfg.rho_c * c_prev + influx - cfg.alpha_buf_on * c_prev * (1.0 - buf_prev)
            + cfg.alpha_buf_off * buf_prev)
            .max(0.0);
        buf_row[key] = (cfg.rho_b * buf_prev + cfg.alpha_buf_on * c_prev * (1.0 - buf_prev)
            - cfg.alpha_buf_off * buf_prev)
            .clamp(0.0, 1.0);
        let rrp_depleted = (t.rrp_prev - release_sum).max(0.0);
        let reserve_refilled = if d0_row.is_some() {
            t.res_prev + t.d0_prev
        } else {
            t.res_prev
        };
        let take = reserve_refilled.min(1.0);
        res_row[key] = (reserve_refilled - cfg.prime_rate * take).max(0.0);
        rrp_row[key] = (rrp_depleted + cfg.prime_rate * take).clamp(0.0, 30.0);
        pr_row[key] = (t.pr_prev * (1.0 - cfg.unprime_per_release * release_sum)
            + cfg.nsf_recover * (1.0 - t.pr_prev))
            .clamp(0.0, 1.0);
        cl_row[key] =
            (t.cl_prev * 0.995 + 0.005 - cfg.unprime_per_release * release_sum).clamp(0.0, 1.0);
        e_row[key] = (t.e_prev + cfg.energy_fill * (cfg.energy_max - t.e_prev)
            - cfg.energy_use * release_sum)
            .clamp(0.0, cfg.energy_max);
        if let Some(d0_slice) = d0_row.as_deref_mut() {
            d0_slice[key] = release_sum * cfg.rec_rate;
        }
    }
}

enum StateBuffer<'py> {
    InPlace {
        obj: Bound<'py, PyAny>,
        rw: numpy::PyReadwriteArrayDyn<'py, f32>,
    },
    Owned(Array3<f32>),
}

impl<'py> StateBuffer<'py> {
    fn from_dict(
        state: &Bound<'py, PyDict>,
        name: &str,
        expected_shape: Option<&[usize]>,
    ) -> PyResult<(Self, Vec<usize>)> {
        let obj = state.get_item(name)?.ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(format!("missing state[{name:?}]"))
        })?;
        Self::from_any(obj, name, expected_shape)
    }

    fn from_any(
        obj: Bound<'py, PyAny>,
        name: &str,
        expected_shape: Option<&[usize]>,
    ) -> PyResult<(Self, Vec<usize>)> {
        if let Ok(rw) = obj.extract::<numpy::PyReadwriteArrayDyn<'py, f32>>() {
            let shape = rw.shape().to_vec();
            if shape.len() != 3 {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "state[{name:?}] must be a float32 rank-3 array, got shape {shape:?}"
                )));
            }
            if let Some(exp) = expected_shape {
                ensure_shape(name, &shape, exp)?;
            }
            let can_in_place = {
                let view = rw.as_array();
                let strides = view.strides();
                rw.as_slice().is_ok()
                    || (strides.len() == 3
                        && strides[2] == 1
                        && strides[1] >= shape[2] as isize
                        && strides[0] >= strides[1] * (shape[1] as isize))
            };
            if can_in_place {
                return Ok((Self::InPlace { obj, rw }, shape));
            }
        }
        let ro = obj.extract::<PyReadonlyArrayDyn<'py, f32>>().map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "state[{name:?}] must be a float32 rank-3 array: {e}"
            ))
        })?;
        let arr = ro
            .as_array()
            .into_dimensionality::<ndarray::Ix3>()
            .map_err(|error| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "state[{name:?}] must be a float32 rank-3 array: {error}"
                ))
            })?;
        let shape = arr.shape().to_vec();
        if let Some(exp) = expected_shape {
            ensure_shape(name, &shape, exp)?;
        }
        Ok((Self::Owned(arr.to_owned()), shape))
    }

    fn view_mut(&mut self) -> ndarray::ArrayViewMut3<'_, f32> {
        match self {
            Self::InPlace { rw, .. } => rw
                .as_array_mut()
                .into_dimensionality::<ndarray::Ix3>()
                .expect("InPlace StateBuffer verified rank-3 at construction"),
            Self::Owned(arr) => arr.view_mut(),
        }
    }

    fn into_py_obj(self, py: Python<'py>) -> Bound<'py, PyAny> {
        match self {
            Self::InPlace { obj, .. } => obj,
            Self::Owned(arr) => arr.into_pyarray(py).into_any(),
        }
    }
}

/// Canonical sparse presynaptic CPU step for deterministic one-query decode.
///
/// The function consumes NumPy float32/int64/bool arrays so the fixture is language agnostic.  It
/// implements the same gather -> per-edge release -> duplicate-safe scatter -> whole-key state
/// advancement order as Python and the live Triton kernel.  Eval keeps `ema_e` frozen; callers
/// receive the release tensor already normalized by that scalar.
#[pyfunction]
#[pyo3(signature = (drive, idx, valid, state, cfg_obj, ema_e, logits=None, barrier_scale=0.0))]
#[allow(clippy::too_many_arguments)]
pub fn presyn_release_canonical_cpu<'py>(
    py: Python<'py>,
    drive: PyReadonlyArrayDyn<'py, f32>,
    idx: PyReadonlyArrayDyn<'py, i64>,
    valid: Option<PyReadonlyArrayDyn<'py, bool>>,
    state: Bound<'py, PyDict>,
    cfg_obj: Bound<'py, PyAny>,
    ema_e: &Bound<'py, PyAny>,
    mut logits: Option<numpy::PyReadwriteArrayDyn<'py, f32>>,
    barrier_scale: f32,
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
    let drive_arr = drive
        .as_array()
        .into_dimensionality::<ndarray::Ix4>()
        .map_err(|error| {
            pyo3::exceptions::PyValueError::new_err(format!("drive must be rank 4: {error}"))
        })?;
    let idx_arr = idx
        .as_array()
        .into_dimensionality::<ndarray::Ix4>()
        .map_err(|error| {
            pyo3::exceptions::PyValueError::new_err(format!("idx must be rank 4: {error}"))
        })?;
    let valid_arr = match valid.as_ref() {
        Some(v) => Some(
            v.as_array()
                .into_dimensionality::<ndarray::Ix4>()
                .map_err(|error| {
                    pyo3::exceptions::PyValueError::new_err(format!(
                        "valid must be rank 4: {error}"
                    ))
                })?,
        ),
        None => None,
    };
    let drive_shape = drive_arr.shape();
    if drive_shape[2] != 1 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "canonical Rust decode requires Tq == 1, got {}",
            drive_shape[2]
        )));
    }
    ensure_shape("idx", idx_arr.shape(), drive_shape)?;
    if let Some(ref v_arr) = valid_arr {
        ensure_shape("valid", v_arr.shape(), drive_shape)?;
    }

    let batch = drive_shape[0];
    let heads = drive_shape[1];
    let topk = drive_shape[3];

    let (mut calcium, state_shape) = StateBuffer::from_dict(&state, "C", None)?;
    if state_shape[0] != batch || state_shape[1] != heads {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "state batch/head dimensions must be ({batch}, {heads}), got {:?}",
            &state_shape[..2]
        )));
    }
    let key_count = state_shape[2];
    if let Some(ref rw_logits) = logits {
        ensure_shape("logits", rw_logits.shape(), &[batch, heads, 1, key_count])?;
    }
    let (mut buffer, _) = StateBuffer::from_dict(&state, "BUF", Some(&state_shape))?;
    let (mut rrp, _) = StateBuffer::from_dict(&state, "RRP", Some(&state_shape))?;
    let (mut reserve, _) = StateBuffer::from_dict(&state, "RES", Some(&state_shape))?;
    let (mut priming, _) = StateBuffer::from_dict(&state, "PR", Some(&state_shape))?;
    let (mut complexin, _) = StateBuffer::from_dict(&state, "CL", Some(&state_shape))?;
    let amp_obj = state
        .get_item("AMP")?
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("missing state[\"AMP\"]"))?;
    {
        let amp_ro = amp_obj
            .extract::<PyReadonlyArrayDyn<'py, f32>>()
            .map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "state[\"AMP\"] must be a float32 rank-3 array: {e}"
                ))
            })?;
        let amp_view = amp_ro
            .as_array()
            .into_dimensionality::<ndarray::Ix3>()
            .map_err(|error| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "state[\"AMP\"] must be a float32 rank-3 array: {error}"
                ))
            })?;
        ensure_shape("AMP", amp_view.shape(), &state_shape)?;
    }
    let (mut energy, _) = StateBuffer::from_dict(&state, "E", Some(&state_shape))?;

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
    let mut delay0: Option<StateBuffer<'py>> = None;
    let mut delay_rest: Vec<Bound<'py, PyAny>> =
        Vec::with_capacity(cfg.endo_delay.saturating_sub(1));
    for (position, item) in delay_list.iter().enumerate() {
        let name = format!("DELAY[{position}]");
        if position == 0 {
            let (buf0, _) = StateBuffer::from_any(item, &name, Some(&state_shape))?;
            delay0 = Some(buf0);
        } else {
            let ro = item
                .extract::<PyReadonlyArrayDyn<'py, f32>>()
                .map_err(|e| {
                    pyo3::exceptions::PyValueError::new_err(format!("{name} must be rank 3: {e}"))
                })?;
            let view = ro
                .as_array()
                .into_dimensionality::<ndarray::Ix3>()
                .map_err(|error| {
                    pyo3::exceptions::PyValueError::new_err(format!(
                        "{name} must be rank 3: {error}"
                    ))
                })?;
            ensure_shape(&name, view.shape(), &state_shape)?;
            delay_rest.push(item);
        }
    }

    let drive_cow = drive_arr.as_standard_layout();
    let idx_cow = idx_arr.as_standard_layout();
    let valid_cow = valid_arr.as_ref().map(|v| v.as_standard_layout());
    let drive_slice = drive_cow.as_slice().expect("standard layout is contiguous");
    let idx_slice = idx_cow.as_slice().expect("standard layout is contiguous");
    let valid_slice_opt = valid_cow
        .as_ref()
        .map(|v| v.as_slice().expect("standard layout is contiguous"));

    for &selected in idx_slice {
        if selected < 0 || selected as usize >= key_count {
            return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "idx value {selected} is outside [0, {key_count})"
            )));
        }
    }

    let mut release = Array4::<f32>::zeros((batch, heads, 1, topk));
    let release_slice = release.as_slice_mut().expect("fresh Array4 is contiguous");

    let mut logits_slice_opt = match logits.as_mut() {
        Some(rw) => Some(rw.as_slice_mut().map_err(|_| {
            pyo3::exceptions::PyValueError::new_err(
                "logits must be contiguous for in-place fused bias injection",
            )
        })?),
        None => None,
    };

    let mut c_view = calcium.view_mut();
    let mut buf_view = buffer.view_mut();
    let mut rrp_view = rrp.view_mut();
    let mut res_view = reserve.view_mut();
    let mut pr_view = priming.view_mut();
    let mut cl_view = complexin.view_mut();
    let mut e_view = energy.view_mut();
    let mut d0_view_opt = delay0.as_mut().map(StateBuffer::view_mut);

    let n_rows = batch * heads;
    let ema_denom = ema_e + 1e-6;

    if n_rows > 0 && key_count > 0 && topk > 0 {
        py.detach(|| {
            for b in 0..batch {
                for h in 0..heads {
                    let row = b * heads + h;
                    let ks = row * key_count;
                    let ke = ks + key_count;
                    let es = row * topk;
                    let ee = es + topk;
                    let c_row = c_view
                        .slice_mut(ndarray::s![b, h, ..])
                        .into_slice()
                        .expect("row contiguous");
                    let buf_row = buf_view
                        .slice_mut(ndarray::s![b, h, ..])
                        .into_slice()
                        .expect("row contiguous");
                    let rrp_row = rrp_view
                        .slice_mut(ndarray::s![b, h, ..])
                        .into_slice()
                        .expect("row contiguous");
                    let res_row = res_view
                        .slice_mut(ndarray::s![b, h, ..])
                        .into_slice()
                        .expect("row contiguous");
                    let pr_row = pr_view
                        .slice_mut(ndarray::s![b, h, ..])
                        .into_slice()
                        .expect("row contiguous");
                    let cl_row = cl_view
                        .slice_mut(ndarray::s![b, h, ..])
                        .into_slice()
                        .expect("row contiguous");
                    let e_row = e_view
                        .slice_mut(ndarray::s![b, h, ..])
                        .into_slice()
                        .expect("row contiguous");
                    let d0_row = d0_view_opt.as_mut().map(|d| {
                        d.slice_mut(ndarray::s![b, h, ..])
                            .into_slice()
                            .expect("row contiguous")
                    });
                    let logits_row = logits_slice_opt.as_deref_mut().map(|l| &mut l[ks..ke]);
                    canonical_decode_row(
                        &drive_slice[es..ee],
                        &idx_slice[es..ee],
                        valid_slice_opt.map(|v| &v[es..ee]),
                        c_row,
                        buf_row,
                        rrp_row,
                        res_row,
                        pr_row,
                        cl_row,
                        e_row,
                        d0_row,
                        &mut release_slice[es..ee],
                        logits_row,
                        barrier_scale,
                        &cfg,
                        ema_denom,
                    );
                }
            }
        });
    } else if n_rows > 0 && key_count > 0 {
        // topk == 0: still advance passive state across all keys
        py.detach(|| {
            for b in 0..batch {
                for h in 0..heads {
                    let row = b * heads + h;
                    let ks = row * key_count;
                    let ke = ks + key_count;
                    let c_row = c_view
                        .slice_mut(ndarray::s![b, h, ..])
                        .into_slice()
                        .expect("row contiguous");
                    let buf_row = buf_view
                        .slice_mut(ndarray::s![b, h, ..])
                        .into_slice()
                        .expect("row contiguous");
                    let rrp_row = rrp_view
                        .slice_mut(ndarray::s![b, h, ..])
                        .into_slice()
                        .expect("row contiguous");
                    let res_row = res_view
                        .slice_mut(ndarray::s![b, h, ..])
                        .into_slice()
                        .expect("row contiguous");
                    let pr_row = pr_view
                        .slice_mut(ndarray::s![b, h, ..])
                        .into_slice()
                        .expect("row contiguous");
                    let cl_row = cl_view
                        .slice_mut(ndarray::s![b, h, ..])
                        .into_slice()
                        .expect("row contiguous");
                    let e_row = e_view
                        .slice_mut(ndarray::s![b, h, ..])
                        .into_slice()
                        .expect("row contiguous");
                    let d0_row = d0_view_opt.as_mut().map(|d| {
                        d.slice_mut(ndarray::s![b, h, ..])
                            .into_slice()
                            .expect("row contiguous")
                    });
                    let logits_row = logits_slice_opt.as_deref_mut().map(|l| &mut l[ks..ke]);
                    canonical_decode_row(
                        &[],
                        &[],
                        None,
                        c_row,
                        buf_row,
                        rrp_row,
                        res_row,
                        pr_row,
                        cl_row,
                        e_row,
                        d0_row,
                        &mut [],
                        logits_row,
                        barrier_scale,
                        &cfg,
                        ema_denom,
                    );
                }
            }
        });
    }

    let out = PyDict::new(py);
    out.set_item("C", calcium.into_py_obj(py))?;
    out.set_item("BUF", buffer.into_py_obj(py))?;
    out.set_item("RRP", rrp.into_py_obj(py))?;
    out.set_item("RES", reserve.into_py_obj(py))?;
    out.set_item("PR", priming.into_py_obj(py))?;
    out.set_item("CL", complexin.into_py_obj(py))?;
    out.set_item("AMP", amp_obj)?;
    out.set_item("E", energy.into_py_obj(py))?;
    let delay_output = PyList::empty(py);
    for entry in delay_rest {
        delay_output.append(entry)?;
    }
    if let Some(d0_tail) = delay0 {
        delay_output.append(d0_tail.into_py_obj(py))?;
    }
    out.set_item("DELAY", delay_output)?;

    Ok((
        release.into_dyn().into_pyarray(py).to_owned(),
        out.to_owned(),
    ))
}

/// (edge release, pre-EMA drive gradient factor) of `presyn_detached_scan_cpu`.
type ScanOutputs<'py> = (
    Bound<'py, PyArrayDyn<f32>>,
    Bound<'py, PyArrayDyn<f32>>,
);

/// Exact detached causal scan of a whole query block on CPU (l7c9 / jyb.9).
///
/// The same contract as the Triton `presyn_detached_scan_kernel` and op-for-op the release and
/// state blocks of `synaptic._scripted_detached_presyn_scan`: for each query `t`, gather the
/// pre-update state of its top-k keys, release (deterministic, or the normal-reparameterised
/// stochastic branch from the caller's draws), then advance every key of the active prefix
/// `first_active + t` (duplicates accumulate like `scatter_add_`, invalid edges contribute
/// nothing). Rows (batch x head) are independent and run in parallel.
///
/// `state` is the (7, N, T_key) slab [C, BUF, RRP, RES, PR, CL, E] and `delay` the
/// (D, N, T_key) endocytosis queue; both are advanced in place. Returns the un-normalised edge
/// release `qamp * released` (N, T, K) (the cross-row EMA normaliser is applied by the caller),
/// and, when `record`, the exact closed-form pre-EMA derivative `d(qamp * released)/d(drive)`
/// (N, T, K) computed directly from the in-register forward intermediates.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn presyn_detached_scan_cpu<'py>(
    py: Python<'py>,
    mut state: numpy::PyReadwriteArrayDyn<'py, f32>,
    mut delay: numpy::PyReadwriteArrayDyn<'py, f32>,
    drive: PyReadonlyArrayDyn<'py, f32>,
    idx: PyReadonlyArrayDyn<'py, i64>,
    valid: PyReadonlyArrayDyn<'py, bool>,
    uniform: PyReadonlyArrayDyn<'py, f32>,
    noise: PyReadonlyArrayDyn<'py, f32>,
    cfg_obj: Bound<'py, PyAny>,
    first_active: usize,
    stochastic_frac: f32,
    count_cap: f32,
    record: bool,
) -> PyResult<ScanOutputs<'py>> {
    use ndarray::{Array3, Axis, Ix2, Ix3, Zip};

    let cfg = CanonicalConfig::from_py(&cfg_obj)?;
    let rank = |name: &str, error: ndarray::ShapeError| {
        pyo3::exceptions::PyValueError::new_err(format!("{name} has the wrong rank: {error}"))
    };
    let mut state = state
        .as_array_mut()
        .into_dimensionality::<Ix3>()
        .map_err(|e| rank("state", e))?;
    let mut delay = delay
        .as_array_mut()
        .into_dimensionality::<Ix3>()
        .map_err(|e| rank("delay", e))?;
    let drive = drive
        .as_array()
        .into_dimensionality::<Ix3>()
        .map_err(|e| rank("drive", e))?;
    let idx = idx
        .as_array()
        .into_dimensionality::<Ix3>()
        .map_err(|e| rank("idx", e))?;
    let valid = valid
        .as_array()
        .into_dimensionality::<Ix3>()
        .map_err(|e| rank("valid", e))?;
    let (n_rows, queries, topk) = drive.dim();
    ensure_shape("idx", idx.shape(), drive.shape())?;
    ensure_shape("valid", valid.shape(), drive.shape())?;
    if state.shape()[0] != 7 || state.shape()[1] != n_rows {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "state must be (7, {n_rows}, T_key), got {:?}",
            state.shape()
        )));
    }
    let key_count = state.shape()[2];
    ensure_shape("delay", &delay.shape()[1..], &[n_rows, key_count])?;
    if delay.shape()[0] != cfg.endo_delay {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "delay must hold endo_delay={} planes, got {}",
            cfg.endo_delay,
            delay.shape()[0]
        )));
    }
    if first_active < 1 || first_active + queries - 1 > key_count {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "queries [{first_active}, {}] exceed the key-state extent {key_count}",
            first_active + queries - 1
        )));
    }
    for (&selected, &ok) in idx.iter().zip(valid.iter()) {
        if ok && (selected < 0 || selected as usize >= key_count) {
            return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "valid idx value {selected} is outside [0, {key_count})"
            )));
        }
    }
    let stochastic = stochastic_frac > 0.0;
    let uniform = if stochastic {
        let u = uniform
            .as_array()
            .into_dimensionality::<Ix2>()
            .map_err(|e| rank("uniform", e))?;
        ensure_shape("uniform", u.shape(), &[n_rows, queries])?;
        Some(u)
    } else {
        None
    };
    let noise = if stochastic {
        let z = noise
            .as_array()
            .into_dimensionality::<Ix3>()
            .map_err(|e| rank("noise", e))?;
        ensure_shape("noise", z.shape(), drive.shape())?;
        Some(z)
    } else {
        None
    };

    let mut e_raw = Array3::<f32>::zeros((n_rows, queries, topk));
    let (rq, rk) = if record { (queries, topk) } else { (0, 0) };
    let mut grad_factor = Array3::<f32>::zeros((n_rows, rq, rk));
    let delay_len = cfg.endo_delay;

    py.detach(|| {
        Zip::indexed(state.axis_iter_mut(Axis(1)))
            .and(delay.axis_iter_mut(Axis(1)))
            .and(e_raw.axis_iter_mut(Axis(0)))
            .and(grad_factor.axis_iter_mut(Axis(0)))
            .par_for_each(|row, mut st, mut dl, mut out, mut gf| {
                let mut release_sum = vec![0.0f32; key_count];
                let mut drive_sum = vec![0.0f32; key_count];
                let mut access = vec![0.0f32; key_count];
                let mut keys = vec![0usize; topk];
                let mut released = vec![0.0f32; topk];
                let mut drives = vec![0.0f32; topk];
                let mut oks = vec![false; topk];
                for t in 0..queries {
                    let active = first_active + t;
                    let draw = uniform.as_ref().map(|u| u[[row, t]]);
                    for k in 0..topk {
                        let ok = valid[[row, t, k]];
                        let key = if ok { idx[[row, t, k]] as usize } else { 0 };
                        let edge_drive = drive[[row, t, k]];
                        let c_edge = st[[0, key]];
                        let buf_edge = st[[1, key]];
                        let rrp_edge = st[[2, key]];
                        let pr_edge = st[[4, key]];
                        let cl_edge = st[[5, key]];
                        let energy_edge = st[[6, key]];
                        let sig_drive = sigmoid(edge_drive);
                        let calcium_raw = cfg.rho_c * c_edge + cfg.alpha_ca * softplus(edge_drive)
                            - cfg.alpha_buf_on * c_edge * (1.0 - buf_edge)
                            + cfg.alpha_buf_off * buf_edge;
                        let calcium = calcium_raw.max(0.0);
                        let fast_denom = calcium + cfg.syt_fast_kd;
                        let slow_denom = calcium + cfg.syt_slow_kd;
                        let fast = calcium / fast_denom;
                        let slow = calcium / slow_denom;
                        let doc2_sig = sigmoid(4.0 * (calcium - 0.12));
                        let sensor = 0.7 * fast + 0.3 * slow + cfg.doc2_gain * doc2_sig;
                        let fuse_base = sigmoid(
                            3.0 * sensor + 2.0 * pr_edge - 2.0 * (cl_edge + cfg.complexin_bias),
                        );
                        let prob_raw = fuse_base * sig_drive;
                        let probability = prob_raw.clamp(0.0, 1.0);
                        let mut rel = probability * rrp_edge;
                        let mut d_rel_d_prob = rrp_edge;
                        if let (Some(u), Some(z)) = (draw, noise.as_ref()) {
                            let sampled_edge = ok && u < stochastic_frac;
                            if sampled_edge {
                                let count = rrp_edge.round_ties_even().clamp(0.0, count_cap);
                                let p32 = probability.clamp(1e-6, 1.0 - 1e-6);
                                let deviation = (count * p32 * (1.0 - p32) + 1e-6).sqrt();
                                let z_val = z[[row, t, k]];
                                let sampled_raw = count * p32 + deviation * z_val;
                                rel = sampled_raw.max(0.0).min(count);
                                if record {
                                    if count_cap <= 0.0 {
                                        d_rel_d_prob = 0.0;
                                    } else if sampled_raw >= 0.0
                                        && sampled_raw <= count
                                        && (1e-6..=1.0 - 1e-6).contains(&probability)
                                    {
                                        let d_dev = count * (1.0 - 2.0 * p32) / (2.0 * deviation);
                                        d_rel_d_prob = count + z_val * d_dev;
                                    } else {
                                        d_rel_d_prob = 0.0;
                                    }
                                }
                            }
                        }
                        if !ok {
                            rel = 0.0;
                        }
                        let qamp = sigmoid(cfg.q_beta * (energy_edge - 0.5)) * cfg.qmax;
                        out[[t, k]] = rel * qamp;
                        if record {
                            gf[[t, k]] = if ok && (0.0..=1.0).contains(&prob_raw) {
                                let d_sensor_d_ca =
                                    0.7 * cfg.syt_fast_kd / (fast_denom * fast_denom)
                                        + 0.3 * cfg.syt_slow_kd / (slow_denom * slow_denom)
                                        + 4.0 * cfg.doc2_gain * doc2_sig * (1.0 - doc2_sig);
                                let d_fuse_d_ca =
                                    3.0 * fuse_base * (1.0 - fuse_base) * d_sensor_d_ca;
                                let d_ca_d_drive = if calcium_raw >= 0.0 {
                                    cfg.alpha_ca * sig_drive
                                } else {
                                    0.0
                                };
                                let d_prob_d_drive = d_fuse_d_ca * d_ca_d_drive * sig_drive
                                    + fuse_base * sig_drive * (1.0 - sig_drive);
                                qamp * d_rel_d_prob * d_prob_d_drive
                            } else {
                                0.0
                            };
                        }
                        keys[k] = key;
                        released[k] = rel;
                        drives[k] = if ok { edge_drive } else { 0.0 };
                        oks[k] = ok;
                    }
                    for k in 0..topk {
                        release_sum[keys[k]] += released[k];
                        drive_sum[keys[k]] += drives[k];
                        if oks[k] {
                            access[keys[k]] += 1.0;
                        }
                    }
                    for key in 0..active {
                        let release_k = release_sum[key];
                        let c_prev = st[[0, key]];
                        let buf_prev = st[[1, key]];
                        let influx = if access[key] > 0.0 {
                            cfg.alpha_ca * softplus(drive_sum[key])
                        } else {
                            0.0
                        };
                        st[[0, key]] = (cfg.rho_c * c_prev + influx
                            - cfg.alpha_buf_on * c_prev * (1.0 - buf_prev)
                            + cfg.alpha_buf_off * buf_prev)
                            .max(0.0);
                        st[[1, key]] = (cfg.rho_b * buf_prev
                            + cfg.alpha_buf_on * c_prev * (1.0 - buf_prev)
                            - cfg.alpha_buf_off * buf_prev)
                            .clamp(0.0, 1.0);
                        let rrp_depleted = (st[[2, key]] - release_k).max(0.0);
                        let mut reserve = st[[3, key]];
                        if delay_len > 0 {
                            reserve += dl[[0, key]];
                            for slot in 0..delay_len - 1 {
                                dl[[slot, key]] = dl[[slot + 1, key]];
                            }
                            dl[[delay_len - 1, key]] = release_k * cfg.rec_rate;
                        }
                        let take = reserve.min(1.0);
                        st[[3, key]] = (reserve - cfg.prime_rate * take).max(0.0);
                        st[[2, key]] = (rrp_depleted + cfg.prime_rate * take).clamp(0.0, 30.0);
                        let pr_prev = st[[4, key]];
                        st[[4, key]] = (pr_prev * (1.0 - cfg.unprime_per_release * release_k)
                            + cfg.nsf_recover * (1.0 - pr_prev))
                            .clamp(0.0, 1.0);
                        st[[5, key]] = (st[[5, key]] * 0.995 + 0.005
                            - cfg.unprime_per_release * release_k)
                            .clamp(0.0, 1.0);
                        let energy_prev = st[[6, key]];
                        st[[6, key]] = (energy_prev
                            + cfg.energy_fill * (cfg.energy_max - energy_prev)
                            - cfg.energy_use * release_k)
                            .clamp(0.0, cfg.energy_max);
                    }
                    for &key in &keys {
                        release_sum[key] = 0.0;
                        drive_sum[key] = 0.0;
                        access[key] = 0.0;
                    }
                }
            });
    });

    Ok((
        e_raw.into_dyn().into_pyarray(py).to_owned(),
        grad_factor.into_dyn().into_pyarray(py).to_owned(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_cfg() -> CanonicalConfig {
        CanonicalConfig {
            rho_c: (-1.0f32 / 6.0).exp(),
            rho_b: (-1.0f32 / 24.0).exp(),
            alpha_ca: 0.55,
            alpha_buf_on: 0.12,
            alpha_buf_off: 0.03,
            syt_fast_kd: 0.4,
            syt_slow_kd: 1.0,
            doc2_gain: 0.08,
            complexin_bias: 0.0,
            q_beta: 4.0,
            qmax: 2.0,
            prime_rate: 0.075,
            unprime_per_release: 0.05,
            nsf_recover: 0.08,
            rec_rate: 0.1,
            energy_fill: 0.02,
            energy_max: 1.0,
            energy_use: 0.02,
            endo_delay: 3,
            lambda_loge: 1.0,
            epsilon: 1e-6,
            loge_bias_clamp: 10.0,
        }
    }

    #[test]
    fn canonical_decode_row_matches_reference_with_duplicates_and_invalid_edges() {
        let cfg = test_cfg();
        let drive = [0.6f32, 0.25, -0.4, 1.1];
        let idx = [1i64, 1, 3, 2];
        let valid = [true, true, false, true];
        let mut c = [0.2f32, 0.5, 0.1, 0.8];
        let mut buf = [0.1f32, 0.3, 0.2, 0.4];
        let mut rrp = [3.0f32, 4.0, 2.5, 1.5];
        let mut res = [0.5f32, 0.6, 0.7, 0.8];
        let mut pr = [0.4f32, 0.7, 0.5, 0.6];
        let mut cl = [0.2f32, 0.3, 0.1, 0.25];
        let mut e = [0.8f32, 0.9, 0.6, 0.7];
        let mut d0 = [0.05f32, 0.02, 0.01, 0.03];
        let mut rel = [0.0f32; 4];

        let c_init = c;
        let buf_init = buf;
        let rrp_init = rrp;
        let res_init = res;
        let pr_init = pr;
        let cl_init = cl;
        let e_init = e;
        let d0_init = d0;
        let ema_denom = 1.0f32 + 1e-6;

        canonical_decode_row(
            &drive,
            &idx,
            Some(&valid),
            &mut c,
            &mut buf,
            &mut rrp,
            &mut res,
            &mut pr,
            &mut cl,
            &mut e,
            Some(&mut d0),
            &mut rel,
            None,
            0.0,
            &cfg,
            ema_denom,
        );

        // Edge 2 is invalid (valid[2] == false) -> release must be 0.0 and key 3 must follow passive decay.
        assert_eq!(rel[2], 0.0);
        assert!(rel[0] > 0.0 && rel[1] > 0.0 && rel[3] > 0.0);
        // Key 0 and Key 3 were not validly accessed -> d0 tail is 0.0.
        assert_eq!(d0[0], 0.0);
        assert_eq!(d0[3], 0.0);
        // Key 1 was accessed twice -> d0[1] equals (released_0 + released_1) * rec_rate.
        let qamp_1 = sigmoid(cfg.q_beta * (e_init[1] - 0.5)) * cfg.qmax;
        let released_sum_1 = (rel[0] + rel[1]) * ema_denom / qamp_1;
        assert!((d0[1] - released_sum_1 * cfg.rec_rate).abs() < 1e-6);

        // Verify key 1 calcium includes softplus(drive[0] + drive[1]) influx.
        let expected_c1 = (cfg.rho_c * c_init[1] + cfg.alpha_ca * softplus(drive[0] + drive[1])
            - cfg.alpha_buf_on * c_init[1] * (1.0 - buf_init[1])
            + cfg.alpha_buf_off * buf_init[1])
            .max(0.0);
        assert!((c[1] - expected_c1).abs() < 1e-6);

        // Verify untouched key 3 followed passive equations using d0_init[3].
        let expected_c3 = (cfg.rho_c * c_init[3]
            - cfg.alpha_buf_on * c_init[3] * (1.0 - buf_init[3])
            + cfg.alpha_buf_off * buf_init[3])
            .max(0.0);
        assert!((c[3] - expected_c3).abs() < 1e-6);
        let refilled_3 = res_init[3] + d0_init[3];
        let take_3 = refilled_3.min(1.0);
        assert!((res[3] - (refilled_3 - cfg.prime_rate * take_3).max(0.0)).abs() < 1e-6);
        assert!((rrp[3] - (rrp_init[3] + cfg.prime_rate * take_3).clamp(0.0, 30.0)).abs() < 1e-6);
        assert!(
            (pr[3] - (pr_init[3] + cfg.nsf_recover * (1.0 - pr_init[3])).clamp(0.0, 1.0)).abs()
                < 1e-6
        );
        assert!((cl[3] - (cl_init[3] * 0.995 + 0.005).clamp(0.0, 1.0)).abs() < 1e-6);
        assert!(
            (e[3] - (e_init[3] + cfg.energy_fill * (cfg.energy_max - e_init[3])).clamp(0.0, 1.0))
                .abs()
                < 1e-6
        );
    }
}

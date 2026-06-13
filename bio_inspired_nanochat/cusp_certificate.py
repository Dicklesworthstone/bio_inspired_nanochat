"""Runtime retention certificate for the bistable CaMKII/PP1 latch (bead 0642.2.2.3).

Turns the *theory* of `docs/theory/singular_perturbation.md` into a runtime artifact: the closed-form
**hysteresis half-width** `δ* = (2/3√3)·(−a)^{3/2}` is the certified lower bound on how far the latch
control can drift before a latched memory is destroyed. We compute the cusp normal-form coefficients
`(a, b)` of the *actual* CaMKII equilibrium residual at a resting operating point, evaluate `δ*`, and
gate the claim on a timescale-separation (normal-hyperbolicity) check. When the latch is monostable
(`a ≥ 0`) or the fast calcium subsystem is not contractive enough to slave to the slow latch
(`ρ(M_cb) > cusp_eps_max`), the certificate is **dropped** and the model falls back to the heuristic
`sax.2` latch with no retention claim — the deterministic fail-closed discipline of the proof ledger
(§5/§7 of the note).

Scope: this module is the certificate + fallback gate (`0642.2.2.3`). The cusp-form latch *update*
and the minimum-energy write/erase pulses are `0642.2.2.1`; a full `ε = τ_fast/τ_slow` gauge is
`0642.10.1` — here `ε` is proxied by the fast-subsystem spectral radius `ρ(M_cb)`, exactly the
normal-hyperbolicity hypothesis (F1/F2) the certificate rests on.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from bio_inspired_nanochat.synaptic import SynapticConfig, cb_spectral_radius
from bio_inspired_nanochat.torch_imports import torch

_TWO_OVER_3SQRT3 = 2.0 / (3.0 * math.sqrt(3.0))


def hill_inflection(n: float, k: float) -> float:
    """Inflection of the self-excitation Hill `H(m)=m^n/(k^n+m^n)`: where `H''=0` (n>1)."""
    if n <= 1.0:
        return k
    return k * ((n - 1.0) / (n + 1.0)) ** (1.0 / n)


def _hill(m: float, n: float, k: float) -> float:
    mn = m ** n
    return mn / (k ** n + mn + 1e-12)


def _camkii_residual(m: float, cfg: SynapticConfig, drive: float, pp1: float) -> float:
    """The CaMKII equilibrium residual G(m) = α_ca·d·(1−m) − β_pp1·p·m + γ·H(m) (note §0/§3).

    Equilibria of the latch are the roots of G; the self-excitation Hill is the only nonconvex term,
    so the cusp lives at its inflection.
    """
    return (
        cfg.latch_alpha_ca * drive * (1.0 - m)
        - cfg.latch_beta_pp1 * pp1 * m
        + cfg.latch_gamma_auto * _hill(m, cfg.latch_hill_n, cfg.latch_hill_k)
    )


def resting_drive(cfg: SynapticConfig, c_rest: float) -> float:
    """The LTP drive `d(c)=σ(g·(c−θ_LTP))` at the resting calcium `c_rest`."""
    return 1.0 / (1.0 + math.exp(-cfg.latch_input_gain * (c_rest - cfg.camkii_thr)))


def _residual_taylor(cfg: SynapticConfig, drive: float, pp1: float, m_star: float, h: float = 1e-3) -> tuple[float, float, float]:
    """Central-difference Taylor coefficients `(C₀, C₁, C₃)` of the CaMKII residual at `m_*`.

    `C₀=G(m_*)`, `C₁=G'(m_*)`, `C₃=G'''(m_*)/6` for `G(m)=α_ca·drive·(1−m) − β_pp1·pp1·m + γ·H(m)`.
    Single source of truth for both the resting certificate (`cusp_coefficients`) and the live runtime
    latch (`CuspLatch`), so the implementation can never drift from the coefficients it is certified
    against. The quadratic `C₂` is omitted: it vanishes at the Hill inflection `m_*` (the cusp
    organizing center), which is the entire reason the reduction lands on the cusp normal form.
    """
    def g(m: float) -> float:
        return _camkii_residual(m, cfg, drive, pp1)

    c0 = g(m_star)
    c1 = (g(m_star + h) - g(m_star - h)) / (2.0 * h)
    c3 = (g(m_star + 2 * h) - 2 * g(m_star + h) + 2 * g(m_star - h) - g(m_star - 2 * h)) / (2.0 * h ** 3) / 6.0
    return c0, c1, c3


def cusp_coefficients(cfg: SynapticConfig, *, c_rest: float | None = None, h: float = 1e-3) -> tuple[float, float]:
    """The cusp normal-form coefficients `(a, b)` of `m̃³ + a·m̃ + b` at the resting operating point.

    `a = −C₁/|C₃|`, `b = −C₀/|C₃|` with `C₀=G(m_*)`, `C₁=G'(m_*)`, `C₃=G'''(m_*)/6` evaluated by
    central finite differences at the Hill inflection `m_*` (the cusp organizing center, where the
    quadratic term vanishes). PP1 sits at its basal floor at rest (the ON state's slaved value is
    floored there). `a < 0 ⟺ bistable` (self-excitation slope exceeds the linear decay).
    """
    if c_rest is None:
        c_rest = 0.5 * (cfg.latch_ltd_thr + cfg.camkii_thr)
    d = resting_drive(cfg, c_rest)
    p = cfg.latch_pp1_basal
    m_star = hill_inflection(cfg.latch_hill_n, cfg.latch_hill_k)
    c0, c1, c3 = _residual_taylor(cfg, d, p, m_star, h)
    # A negligible cubic coefficient (relative to the linear scale) means there is no self-excitation
    # nonlinearity — the residual is affine, the system is monostable, and there is no cusp scaling.
    # Report the monostable boundary a=0 rather than a finite-difference-noise blow-up.
    if abs(c3) < 1e-6 * (1.0 + abs(c1) + abs(c0)):
        return 0.0, 0.0
    return -c1 / abs(c3), -c0 / abs(c3)


def retention_delta_star(a: float) -> float:
    """Closed-form hysteresis half-width `δ*(a)=(2/3√3)·(−a)^{3/2}` (0 for the monostable `a≥0`)."""
    return _TWO_OVER_3SQRT3 * (-a) ** 1.5 if a < 0.0 else 0.0


def epsilon_gauge(cfg: SynapticConfig, n_beta: int = 21) -> float:
    """Normal-hyperbolicity proxy: worst-case fast-subsystem spectral radius `ρ(M_cb)` over β∈[0,1].

    Strong timescale separation (small `ε`) requires the fast calcium↔buffer map to be a clean
    contraction, i.e. `ρ(M_cb)` comfortably below 1 (a clear gap). This reuses the `yw9.7`
    closed-form `cb_spectral_radius`.
    """
    rho_c = torch.tensor(math.exp(-1.0 / cfg.tau_c), dtype=torch.float64)
    rho_b = torch.tensor(math.exp(-1.0 / cfg.tau_buf), dtype=torch.float64)
    a_on = torch.tensor(float(cfg.alpha_buf_on), dtype=torch.float64)
    a_off = torch.tensor(float(cfg.alpha_buf_off), dtype=torch.float64)
    betas = torch.linspace(0.0, 1.0, n_beta, dtype=torch.float64)
    rho = cb_spectral_radius(rho_c, rho_b, a_on, a_off, betas)
    return float(rho.max())


@dataclass(frozen=True)
class RetentionCertificate:
    """The runtime retention certificate + its fail-closed verdict."""

    a: float                 # cusp splitting parameter (a < 0 ⟺ bistable)
    b: float                 # cusp bias parameter
    delta_star: float        # certified retention half-width (0 if uncertified)
    eps: float               # ε proxy = ρ(M_cb), the fast-subsystem spectral radius
    bistable: bool           # a < 0
    separated: bool          # ε ≤ cusp_eps_max (normal hyperbolicity holds)
    certified: bool          # bistable AND separated
    use_heuristic_fallback: bool  # not certified ⟹ fall back to the sax.2 heuristic latch
    reason: str


def certify_retention(cfg: SynapticConfig, *, c_rest: float | None = None) -> RetentionCertificate:
    """Compute the retention certificate for ``cfg`` and apply the deterministic fallback rule.

    Certified iff the latch is bistable (`a < 0`) AND the fast subsystem is contractive enough
    (`ρ(M_cb) ≤ cusp_eps_max`). Otherwise the certificate is void, `δ*` is dropped to 0, and the
    caller should use the heuristic `sax.2` latch (no retention claim).
    """
    a, b = cusp_coefficients(cfg, c_rest=c_rest)
    eps = epsilon_gauge(cfg)
    bistable = a < 0.0
    separated = eps <= cfg.cusp_eps_max
    certified = bool(bistable and separated)
    delta = retention_delta_star(a) if certified else 0.0
    if certified:
        reason = (f"certified: bistable (a={a:.3g}<0) and separated (ρ_fast={eps:.3g}≤"
                  f"{cfg.cusp_eps_max:g}); retention ≥ δ*={delta:.3g}")
    elif not bistable:
        reason = (f"uncertified: monostable (a={a:.3g}≥0) — heuristic sax.2 fallback, no retention "
                  f"claim")
    else:
        reason = (f"uncertified: insufficient timescale separation (ρ_fast={eps:.3g}>"
                  f"{cfg.cusp_eps_max:g}) — heuristic sax.2 fallback")
    return RetentionCertificate(
        a=a, b=b, delta_star=delta, eps=eps, bistable=bistable, separated=separated,
        certified=certified, use_heuristic_fallback=not certified, reason=reason,
    )


# =========================================================================== #
# Runtime cusp latch + minimum-energy write/erase pulses (bead 0642.2.2.1)
# =========================================================================== #
#
# The certificate above turns the *resting* operating point into a retention bound δ*(a). This
# section turns that geometry into the actual runtime *update*: the latch state m evolves on the
# cusp normal form
#
#       Φ(m̃) = ¼·m̃⁴ + ½·a·m̃² + b(c)·m̃ ,        m̃ = m − m_* ,      Φ'(m̃) = m̃³ + a·m̃ + b ,
#       m̃ ← m̃ − η·Φ'(m̃)                          (gradient descent toward the nearest stable root),
#
# with the **splitting parameter `a` fixed at its certified resting value** and the **bias `b(c)`
# the live calcium control** (b = −C₀(c)/|C₃|, the same C₀ the certificate uses). Writing drives b
# below the lower fold −δ*(a) (only the ON root survives); erasing drives b above +δ*(a). Because the
# dynamics *is* the cusp cubic, the fold half-width δ*(a) is the **exact, tight** retention bound:
# a control perturbation of magnitude < δ* leaves all three roots intact and the latch holds; one of
# magnitude > δ* annihilates a fold and the state flips. When the certificate is void (monostable,
# or fast subsystem not contractive enough), the runtime uses the heuristic `sax.2` map instead — the
# deterministic fail-closed fallback (§5 of the theory note).


def hill_inflection_value(cfg: SynapticConfig) -> tuple[float, float]:
    """`(m_*, H(m_*))`: the cusp organizing center and the self-excitation level there."""
    m_star = hill_inflection(cfg.latch_hill_n, cfg.latch_hill_k)
    return m_star, _hill(m_star, cfg.latch_hill_n, cfg.latch_hill_k)


@dataclass(frozen=True)
class CuspConstants:
    """The calcium-independent constants of the live cusp latch, plus its certificate."""

    m_star: float            # Hill inflection (cusp organizing center)
    c3_abs: float            # |C₃| > 0 — the cubic scale that maps the residual G(m) to (a, b)
    h_star: float            # H(m_*) — self-excitation level at the center (in the bias C₀)
    a: float                 # certified resting splitting parameter (a < 0 ⟺ bistable)
    delta_star: float        # certified retention half-width δ*(a) = fold |bias|
    rate: float              # stability-capped gradient-flow step η (retention is rate-independent)
    certificate: RetentionCertificate


def cusp_constants(cfg: SynapticConfig, *, c_rest: float | None = None) -> CuspConstants:
    """Precompute the cusp latch constants for ``cfg`` (one-time, at module construction)."""
    cert = certify_retention(cfg, c_rest=c_rest)
    if c_rest is None:
        c_rest = 0.5 * (cfg.latch_ltd_thr + cfg.camkii_thr)
    m_star, h_star = hill_inflection_value(cfg)
    _, _, c3 = _residual_taylor(cfg, resting_drive(cfg, c_rest), cfg.latch_pp1_basal, m_star)
    # |C₃| floors at a tiny positive value so the monostable branch (a≈0, δ*=0) never divides by 0;
    # in that branch the latch reduces to a leaky linear relaxation and carries no retention claim.
    c3_abs = max(abs(c3), 1e-9)
    a = cert.a
    # Gradient-flow step: the faithful rate is η=|C₃| (it makes the cubic flow's linearization at m_*
    # match the sax.2 Euler map). Cap it so η·max|Φ''| < 1 over the admissible m̃∈[−m_*, 1−m_*],
    # guaranteeing monotone convergence to a stable root (the equilibria — hence δ* — are unchanged
    # by the rate, so the cap never weakens the certificate).
    u_lim = max(m_star, 1.0 - m_star)
    vpp_max = 3.0 * u_lim * u_lim + abs(a)
    rate = min(c3_abs, 0.9 / (vpp_max + 1e-9))
    return CuspConstants(
        m_star=m_star, c3_abs=c3_abs, h_star=h_star, a=a,
        delta_star=cert.delta_star, rate=rate, certificate=cert,
    )


def relax_cubic(u, a: float, b, *, rate: float, steps: int = 1):
    """Gradient-descent the cusp potential `Φ(u)=¼u⁴+½a u²+b u`: `u ← u − rate·(u³+a u+b)`.

    The pure normal-form flow shared by the live latch (`steps=1` per update) and the controllers /
    tests (multi-step relaxation to equilibrium). `b` may be a scalar or a tensor broadcasting on `u`.
    This is the object whose fold structure the certificate δ*(a) describes exactly.
    """
    for _ in range(steps):
        u = u - rate * (u * u * u + a * u + b)
    return u


class CuspLatch:
    """Certified cusp-normal-form CaMKII latch with minimum-energy write/erase pulses (0642.2.2.1).

    Drop-in replacement for the `sax.2` CaMKII/PP1 update inside `PostsynapticHebb.update`, active
    only when ``cfg.cusp_latch`` is set *and* the retention certificate holds. The state `m` (CaMKII)
    is advanced by one gradient step of the cusp cubic with the certified splitting parameter `a` and
    the live calcium bias `b(c)`; PP1 is slaved to its reduced quasi-steady value `p(m,c)`. When the
    certificate is void the latch is **not** used (`self.certified is False`) and the caller keeps the
    heuristic `sax.2` map — the fail-closed discipline of the theory note (§5).
    """

    def __init__(self, cfg: SynapticConfig, *, c_rest: float | None = None) -> None:
        self.cfg = cfg
        self.k = cusp_constants(cfg, c_rest=c_rest)
        self.certified = self.k.certificate.certified

    # -- live coefficients ---------------------------------------------------- #
    def _drive(self, c):
        """BCM LTP drive `d(c)=σ(g·(c−θ_LTP))` (calcium → CaMKII potentiation), torch-vectorized."""
        return torch.sigmoid(self.cfg.latch_input_gain * (c - self.cfg.camkii_thr))

    def _erase(self, c):
        """BCM LTD drive `e(c)=σ(g·(θ_LTD−c))` (low calcium → PP1 activation)."""
        return torch.sigmoid(self.cfg.latch_input_gain * (self.cfg.latch_ltd_thr - c))

    def slaved_pp1(self, m, c):
        """Reduced (†) quasi-steady PP1 `p(m,c)=α_pp1·e / (α_pp1·e + β_cam·m)`, floored at basal p₀.

        This is the slow-manifold slaving that collapses the 2-D `(m,p)` latch to the 1-D cusp flow
        in `m`; feeding it back into `b(c)` keeps the runtime consistent with the §3 reduction.
        """
        cfg = self.cfg
        e = self._erase(c)
        num = cfg.latch_alpha_pp1 * e
        p = num / (num + cfg.latch_beta_camkii * m + 1e-12)
        return p.clamp(min=cfg.latch_pp1_basal, max=1.0)

    def bias_at_calcium(self, c, *, pp1=None, gamma_scale=None, beta_pp1_scale=None):
        """Live cusp bias `b(c) = −C₀(c)/|C₃|`, the only calcium-driven control of the latch.

        `C₀(c) = α_ca·d(c)·(1−m_*) − β_pp1·p·m_* + γ·H(m_*)`. Higher calcium ⇒ larger `d` ⇒ larger
        `C₀` ⇒ **more negative `b`** ⇒ toward the ON fold (write). `gamma_scale`/`beta_pp1_scale`
        carry the per-expert genome modulation (they shift the operating bias; the bistability depth
        `a` stays the certified scalar). `pp1` defaults to the basal floor (the certificate's own
        convention, so `bias_at_calcium(c_rest)` reproduces the certificate's `b`).
        """
        cfg = self.cfg
        d = self._drive(c)
        p = cfg.latch_pp1_basal if pp1 is None else pp1
        gamma = cfg.latch_gamma_auto if gamma_scale is None else cfg.latch_gamma_auto * gamma_scale
        beta_pp1 = cfg.latch_beta_pp1 if beta_pp1_scale is None else cfg.latch_beta_pp1 * beta_pp1_scale
        c0 = (
            cfg.latch_alpha_ca * d * (1.0 - self.k.m_star)
            - beta_pp1 * p * self.k.m_star
            + gamma * self.k.h_star
        )
        return -c0 / self.k.c3_abs

    # -- the runtime update --------------------------------------------------- #
    def step(self, m, ca_proxy, *, gamma_scale=None, beta_pp1_scale=None):
        """One certified cusp step: returns `(m_new, p_new)`; both clamped to their physical ranges.

        Raises if the latch is uncertified — callers must check ``self.certified`` and fall back to
        the heuristic `sax.2` map (fail-closed). `ca_proxy` is the live calcium fed to the latch.
        """
        if not self.certified:
            raise RuntimeError(
                "CuspLatch.step called while uncertified; caller must use the sax.2 fallback "
                f"({self.k.certificate.reason})"
            )
        p = self.slaved_pp1(m, ca_proxy)
        b = self.bias_at_calcium(
            ca_proxy, pp1=p, gamma_scale=gamma_scale, beta_pp1_scale=beta_pp1_scale
        )
        u = m - self.k.m_star
        u_new = relax_cubic(u, self.k.a, b, rate=self.k.rate, steps=1)
        m_new = (u_new + self.k.m_star).clamp(0.0, 1.0)
        return m_new, p

    # -- retention geometry & minimum-energy pulses (0642.2.1.5) --------------- #
    @property
    def delta_star(self) -> float:
        """Certified retention half-width (fold |bias|); 0 when uncertified/monostable."""
        return self.k.delta_star

    def fold_biases(self) -> tuple[float, float]:
        """The two saddle-node bias values `(−δ*, +δ*)` bounding the bistable (three-root) wedge."""
        d = self.k.delta_star
        return (-d, d)

    def min_write_bias(self, margin: float = 1e-6) -> float:
        """Minimal bias that latches OFF→ON: just past the lower fold `b = −δ* − margin`."""
        return -self.k.delta_star - margin

    def min_erase_bias(self, margin: float = 1e-6) -> float:
        """Minimal bias that drops ON→OFF: just past the upper fold `b = +δ* + margin`."""
        return self.k.delta_star + margin

    def _calcium_for_bias(self, target_b: float, *, m_eval: float | None = None,
                          c_lo: float = 0.0, c_hi: float | None = None, iters: int = 60):
        """Invert the (monotone-decreasing) `b(c)` to the calcium that reaches `target_b`.

        Returns the boundary calcium, or ``None`` if `target_b` is unreachable on `[c_lo, c_hi]`.
        Evaluated at `m=m_eval` (default the organizing center `m_*`) with the slaved PP1 there, so
        the calcium pulse is the §3 reduced-flow control. `b(c)` is strictly decreasing in `c`, so a
        plain bisection on the sign of `b(c) − target_b` converges.
        """
        m_eval = self.k.m_star if m_eval is None else m_eval
        if c_hi is None:
            # Comfortably above the LTP threshold: the BCM sigmoid has saturated by θ_LTP + 6/g.
            c_hi = self.cfg.camkii_thr + 6.0 / max(self.cfg.latch_input_gain, 1e-6)

        def b_of(c: float) -> float:
            ct = torch.tensor(float(c), dtype=torch.float64)
            m = torch.tensor(float(m_eval), dtype=torch.float64)
            p = self.slaved_pp1(m, ct)
            return float(self.bias_at_calcium(ct, pp1=p))

        b_lo, b_hi = b_of(c_lo), b_of(c_hi)  # b_lo = b(low c) ≥ b_hi = b(high c)
        if not (b_hi - 1e-12 <= target_b <= b_lo + 1e-12):
            return None
        lo, hi = c_lo, c_hi
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            if b_of(mid) > target_b:   # still above target ⇒ need more calcium
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    def min_write_calcium(self, margin: float = 1e-6):
        """Smallest calcium that crosses the lower fold (OFF disappears ⇒ ON), or ``None``."""
        return self._calcium_for_bias(self.min_write_bias(margin))

    def min_erase_calcium(self, margin: float = 1e-6):
        """Largest calcium that still holds the upper fold (ON disappears ⇒ OFF), or ``None``."""
        return self._calcium_for_bias(self.min_erase_bias(margin))

    def pulse_table(self) -> list[dict]:
        """Write/erase minimum-energy pulse table (bead 0642.2.1.5).

        Each row is the minimal control that flips the latch: the target bias at the fold, the calcium
        that reaches it (the reduced-flow pulse amplitude), and a relative energy proxy = the calcium
        excursion from the neutral resting point `c_rest = ½(θ_LTD+θ_LTP)`. The minimal pulse reaches
        *exactly* the fold; any deeper drive is wasted energy, any shallower fails to flip — the cusp
        geometry makes ``|c_target − c_rest|`` the optimal-control cost.
        """
        c_rest = 0.5 * (self.cfg.latch_ltd_thr + self.cfg.camkii_thr)
        rows: list[dict] = []
        for name, b_target, c_target in (
            ("write", self.min_write_bias(), self.min_write_calcium()),
            ("erase", self.min_erase_bias(), self.min_erase_calcium()),
        ):
            energy = None if c_target is None else abs(c_target - c_rest)
            rows.append({
                "action": name,
                "bias_target": b_target,
                "calcium_target": c_target,
                "energy_proxy": energy,
                "reachable": c_target is not None,
            })
        return rows

    # -- slow-manifold projector (Fenichel reconstruction, §2) ---------------- #
    def quasi_steady_calcium(self, influx, *, steps: int = 500, tol: float = 1e-10):
        """Slow-manifold value `h(influx)`: the calcium↔buffer fixed point under constant influx.

        Iterates the *faithful* live C/BUF map (the one in `release_canonical`) from rest until the
        fast subsystem settles. `influx` already includes the `α_ca·softplus(drive)` scaling. Used by
        the Fenichel reconstruction check (the live calcium relaxes to `h` at rate `ρ(M_cb)=ε`, so the
        reduced latch — calcium slaved to `h` — tracks the full coupled latch to `O(ε)`).
        """
        cfg = self.cfg
        rho_c = math.exp(-1.0 / cfg.tau_c)
        rho_b = math.exp(-1.0 / cfg.tau_buf)
        a_on, a_off = cfg.alpha_buf_on, cfg.alpha_buf_off
        influx_t = torch.as_tensor(influx, dtype=torch.float64)
        c = torch.zeros_like(influx_t)
        buf = torch.zeros_like(influx_t)
        for _ in range(steps):
            c_next = (rho_c * c + influx_t - a_on * c * (1.0 - buf) + a_off * buf).clamp(min=0.0)
            buf_next = (rho_b * buf + a_on * c * (1.0 - buf) - a_off * buf).clamp(0.0, 1.0)
            if float((c_next - c).abs().max()) < tol and float((buf_next - buf).abs().max()) < tol:
                c, buf = c_next, buf_next
                break
            c, buf = c_next, buf_next
        return c

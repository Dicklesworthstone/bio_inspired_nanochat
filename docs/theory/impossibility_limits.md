# Two limits on release precision and leaky memory

The first result applies to the continuous-time reference sampler. The second applies to the raw leaky fast-weight channel with independent innovations. Both have counterexamples if their assumptions are dropped. The precision result uses dimensionless entropy. Converting it to ATP consumption or joules would require a measured energy calibration.

## Precision of the Poisson release current

Let independent forward and reverse event counts have distributions $N_+\sim\mathrm{Poisson}(at)$ and $N_-\sim\mathrm{Poisson}(bt)$ for $t>0$, $a,b>0$, and $a\ne b$. For $J=N_+-N_-$ assign dimensionless entropy $\Sigma=J\log(a/b)$. Then

$$\frac{\operatorname{Var}(J)}{\mathbb E[J]^2}\geq\frac{2}{\mathbb E[\Sigma]}.$$

**Proof.** Independence gives variance $(a+b)t$ and mean $(a-b)t$. Relabel the two directions to make $x=a/b>1$. The inequality reduces to $(x+1)\log x\geq2(x-1)$. The function $\log x-2(x-1)/(x+1)$ vanishes at one and its derivative is $(x-1)^2/[x(x+1)^2]\geq0$. At equilibrium the relative variance is undefined. Its ratio to the bound tends to one as the drive approaches equilibrium.

**Counter-protocol.** Independent Bernoulli release and recovery with probabilities 0.9 and 0.1 per tick give mean current 0.8, variance 0.18 and affinity $\log 81$. The product of relative variance and mean entropy divided by two is about 0.494. A continuous-time certificate would fail here even with infinitely many observations. The existing finite-pool measurements also show this distinction. [Finite-binomial study](stochastic_thermodynamics.md#72-ft-and-tur-scope-one-pass-one-important-limitation)

## Reconstruction of a noisy fast weight

Let $X=W_0\in\mathbb R^d$ satisfy $\operatorname{tr}\operatorname{Cov}(X)\leq dP$. After each of $H\geq1$ writes, let $W_i=\lambda W_{i-1}+U_i+\xi_i$, for $0\leq\lambda<1$. The innovations $\xi_i$ are independent $N(0,qI)$ with $q>0$ and are independent of $X$. The intervening writes $U_{1:H}$ are jointly independent of $X$ and of the noise. The decoder observes only $W_H$. Define $G_H=\sum_{j=0}^{H-1}\lambda^{2j}$. The initial state's retained information, in nats, satisfies

$$I(X;W_H)\leq\frac d2\log\left(1+\frac{\lambda^{2H}P}{qG_H}\right).$$

**Proof.** Revealing all intervening writes can only help the decoder. After subtracting their known contribution, the channel output is $\lambda^H X+Z$ with independent $Z\sim N(0,qG_H I)$. Subtract the differential entropy of $Z$ from the output entropy. Gaussian maximum entropy at fixed covariance and the arithmetic–geometric mean bound on covariance eigenvalues give the displayed inequality. At fixed $q,P,\lambda<1$ this tends to zero. For isotropic Gaussian $X\sim N(0,PI)$ even a decoder given the intervening writes has per-coordinate mean squared error $PqG_H/(\lambda^{2H}P+qG_H)$ by Gaussian conditioning. This is a lower bound on the error of every decoder observing just $W_H$.

**Boundary cases.** Intervening writes that re-encode the original signal break the independence assumption. With zero noise, exact real arithmetic and no later writes, $\lambda^H X$ determines $X$ for every finite $H$ when $\lambda>0$. A decaying norm alone establishes no loss of information. The experiment drives the raw `PostsynapticHebb.hebb_fast` recurrence. Learned feedback, slow weights, normalized writes, nonlinear latches and trained-model retention require separate evidence.

## Replication

`tests/test_impossibility_limits.py` draws both Poisson counts from the project's sampler and estimates their relative variance. For the memory curve it sends independent signal coordinates, distractors and noise through the actual molecular write and measures reconstruction error and channel gain. The decoder coefficients and exact Gaussian MMSE are fixed before sampling. Each regression compares an observation with the exact reduced-model value using six estimated standard errors. The counter-protocol must independently violate the continuous-time bound. The JUnit `curve` properties contain the numerical values.

### Measured curves

Each point used 65,536 independent draws. For memory, the original signal variance was 1, the per-write noise variance 0.09 and the additional distractor variance 0.04.

| Forward / reverse rate | Measured relative variance | Exact | Entropy floor |
|---:|---:|---:|---:|
| 2 / 1 | 0.37310 | 0.37500 | 0.36067 |
| 4 / 1 | 0.06950 | 0.06944 | 0.06011 |
| 8 / 1 | 0.02303 | 0.02296 | 0.01717 |
| 16 / 1 | 0.00953 | 0.00944 | 0.00601 |

| Decay | Lag | Measured reconstruction MSE | Exact MMSE | Noise-only floor |
|---:|---:|---:|---:|---:|
| 0.75 | 1 | 0.1900 | 0.1877 | 0.1379 |
| 0.75 | 3 | 0.5827 | 0.5785 | 0.4872 |
| 0.75 | 6 | 0.8991 | 0.9008 | 0.8628 |
| 0.90 | 1 | 0.1399 | 0.1383 | 0.1000 |
| 0.90 | 3 | 0.3792 | 0.3763 | 0.2946 |
| 0.90 | 6 | 0.6319 | 0.6348 | 0.5462 |

All ten point estimates exceeded their respective floors and followed the independently predicted curve within six estimated standard errors. The discrete counter-protocol failed the Poisson floor as predicted. DSR `b0cea86b-d7a6-4cb6-b745-81e8a6bdb012` passed all seven test cases. The information bound has the analytic proof above. The already compiled Lean result covers the reduced Poisson inequality. [Full precision, standard errors and input hashes](../../results/reality_check_followup_2026-11-22.json)

"""Core Monte Carlo exotic option pricing library.

Vectorized NumPy and Numba GBM path simulators, European/Asian/barrier
payoffs, variance reduction (antithetic variates, control variates), and
finite-difference Greeks with common random numbers (CRN).
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np
from numba import njit, prange
from scipy.stats import norm


@dataclass
class Params:
    S0: float
    K: float
    r: float
    sigma: float
    T: float
    n_steps: int
    n_paths: int
    barrier: Optional[float] = None
    seed: int = 42


# --------------------------------------------------------------------------
# Black-Scholes closed form (European call)
# --------------------------------------------------------------------------
def _d1(S0, K, r, sigma, T):
    return (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))


def bs_call_price(S0, K, r, sigma, T):
    d1 = _d1(S0, K, r, sigma, T)
    d2 = d1 - sigma * np.sqrt(T)
    return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_call_delta(S0, K, r, sigma, T):
    return norm.cdf(_d1(S0, K, r, sigma, T))


def bs_call_vega(S0, K, r, sigma, T):
    return S0 * norm.pdf(_d1(S0, K, r, sigma, T)) * np.sqrt(T)


# --------------------------------------------------------------------------
# Random draws
# --------------------------------------------------------------------------
def draw_normals(n_paths, n_steps, seed, antithetic=False):
    """Standard normal draws, shape (n_paths, n_steps).

    When antithetic=True, the second half is exactly -1 * the first half,
    so path i and path i + n_paths/2 form an antithetic pair.
    """
    rng = np.random.default_rng(seed)
    if antithetic:
        half = n_paths // 2
        z = rng.standard_normal((half, n_steps))
        return np.vstack([z, -z])
    return rng.standard_normal((n_paths, n_steps))


# --------------------------------------------------------------------------
# GBM path simulation (exact lognormal increments, not Euler discretization)
# --------------------------------------------------------------------------
def simulate_paths_numpy(S0, r, sigma, T, Z):
    """Vectorized NumPy GBM path simulator. Z has shape (n_paths, n_steps)."""
    n_paths, n_steps = Z.shape
    dt = T / n_steps
    drift = (r - 0.5 * sigma ** 2) * dt
    vol = sigma * np.sqrt(dt)
    log_increments = drift + vol * Z
    log_paths = np.cumsum(log_increments, axis=1)
    paths = np.empty((n_paths, n_steps + 1))
    paths[:, 0] = S0
    paths[:, 1:] = S0 * np.exp(log_paths)
    return paths


@njit(parallel=True, cache=True)
def simulate_paths_numba(S0, r, sigma, T, Z):
    """Numba njit GBM path simulator, same math as simulate_paths_numpy."""
    n_paths, n_steps = Z.shape
    dt = T / n_steps
    drift = (r - 0.5 * sigma ** 2) * dt
    vol = sigma * np.sqrt(dt)
    paths = np.empty((n_paths, n_steps + 1))
    for i in prange(n_paths):
        paths[i, 0] = S0
        log_s = 0.0
        for j in range(n_steps):
            log_s += drift + vol * Z[i, j]
            paths[i, j + 1] = S0 * np.exp(log_s)
    return paths


# --------------------------------------------------------------------------
# Payoffs
# --------------------------------------------------------------------------
def european_call_payoff(paths, K):
    return np.maximum(paths[:, -1] - K, 0.0)


def asian_call_payoff(paths, K):
    """Arithmetic-average call, averaged over monitoring dates t_1..t_n (excludes t0)."""
    avg = paths[:, 1:].mean(axis=1)
    return np.maximum(avg - K, 0.0)


def up_and_out_barrier_call_payoff(paths, K, barrier):
    """Up-and-out call: knocked out (pays 0) if the path ever reaches the barrier."""
    payoff = np.maximum(paths[:, -1] - K, 0.0).copy()
    knocked_out = paths.max(axis=1) >= barrier
    payoff[knocked_out] = 0.0
    return payoff


# --------------------------------------------------------------------------
# Pricing / variance reduction
# --------------------------------------------------------------------------
def price_from_payoffs(payoffs, r, T, antithetic=False):
    """Discount payoffs and return (price, standard error).

    If antithetic=True, payoffs must be laid out as [plain_half, mirrored_half]
    (as produced by draw_normals(..., antithetic=True)); pairs are averaged
    before computing sample statistics, which is what actually captures the
    variance reduction from the negative correlation between a path and its
    mirror.
    """
    disc = np.exp(-r * T)
    if antithetic:
        n = len(payoffs)
        half = n // 2
        sample = disc * (payoffs[:half] + payoffs[half:]) / 2.0
    else:
        sample = disc * payoffs
    price = sample.mean()
    se = sample.std(ddof=1) / np.sqrt(len(sample))
    return price, se


def control_variate_price(target_payoffs, control_payoffs, control_true_price, r, T):
    """Control-variate estimator: target adjusted by the (known-mean) control.

    control_payoffs must come from the SAME simulated paths as target_payoffs
    so that they are correlated. control_true_price is the exact (e.g. BS)
    discounted expectation of the control payoff.
    """
    disc = np.exp(-r * T)
    X = disc * target_payoffs
    C = disc * control_payoffs
    cov = np.cov(X, C, ddof=1)[0, 1]
    var_c = np.var(C, ddof=1)
    beta = cov / var_c
    Y = X - beta * (C - control_true_price)
    price = Y.mean()
    se = Y.std(ddof=1) / np.sqrt(len(Y))
    return price, se, beta

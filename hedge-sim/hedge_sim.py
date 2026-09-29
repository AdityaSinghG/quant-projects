"""Delta-hedging simulator core library.

Simulates GBM stock paths, sells a European call at t=0 for the Black-Scholes
premium, and delta-hedges the short position under several rebalancing rules:
fixed calendar frequencies (daily/weekly/monthly/intraday) and a no-trade
band that only rebalances when the held hedge drifts from the BS delta by
more than a threshold. Proportional transaction costs are charged on every
share traded.

The GBM path generator and Black-Scholes formulas are copied from mc-pricer
(not imported) so this package is self-contained.
"""
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm


@dataclass
class Config:
    S0: float = 100.0
    K: float = 100.0
    r: float = 0.05
    sigma: float = 0.20
    T: float = 1.0
    n_days: int = 252       # base simulation grid: one step per trading day
    n_paths: int = 10_000
    seed: int = 42
    tc_bps: float = 5.0     # proportional transaction cost, in bps of trade notional


# --------------------------------------------------------------------------
# Black-Scholes closed form (European call) -- copied from mc-pricer
# --------------------------------------------------------------------------
def _d1(S0, K, r, sigma, T):
    return (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))


def bs_call_price(S0, K, r, sigma, T):
    d1 = _d1(S0, K, r, sigma, T)
    d2 = d1 - sigma * np.sqrt(T)
    return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_call_delta(S0, K, r, sigma, T):
    return norm.cdf(_d1(S0, K, r, sigma, T))


# --------------------------------------------------------------------------
# GBM path simulation (exact lognormal increments) -- copied from mc-pricer
# --------------------------------------------------------------------------
def draw_normals(n_paths, n_steps, seed):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_paths, n_steps))


def simulate_paths(S0, r, sigma, T, Z):
    """Vectorized GBM path simulator. Z has shape (n_paths, n_steps)."""
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


def simulate_gbm(cfg: Config, n_steps=None):
    """Simulate GBM paths for the given config on a grid with n_steps steps
    (defaults to cfg.n_days). Uses cfg.seed, so repeated calls with the same
    n_steps reproduce the same paths."""
    n_steps = cfg.n_days if n_steps is None else n_steps
    Z = draw_normals(cfg.n_paths, n_steps, cfg.seed)
    return simulate_paths(cfg.S0, cfg.r, cfg.sigma, cfg.T, Z)


# --------------------------------------------------------------------------
# Delta hedging simulation
# --------------------------------------------------------------------------
def _initial_cash(cfg: Config):
    return bs_call_price(cfg.S0, cfg.K, cfg.r, cfg.sigma, cfg.T)


def run_fixed_frequency_hedge(S, cfg: Config, stride, tc_bps=None):
    """Delta-hedge rebalancing every `stride` base-grid steps (1=daily,
    5=weekly, 21=monthly on a 252-step grid). Returns (pnl0, cost) arrays of
    shape (n_paths,): pnl0 is terminal hedging P&L discounted to t=0, cost is
    total (undiscounted) transaction cost paid per path.
    """
    tc = (cfg.tc_bps if tc_bps is None else tc_bps) / 1e4
    n_paths, n_cols = S.shape
    n_days = n_cols - 1
    dt = cfg.T / n_days

    cash = np.full(n_paths, _initial_cash(cfg))
    shares = np.zeros(n_paths)
    total_cost = np.zeros(n_paths)

    for d in range(0, n_days, stride):
        tau = cfg.T - d * dt
        Sd = S[:, d]
        target = bs_call_delta(Sd, cfg.K, cfg.r, cfg.sigma, tau)
        trade = target - shares
        cost = tc * np.abs(trade) * Sd
        cash -= trade * Sd + cost
        shares = target
        total_cost += cost
        next_d = d + stride if d + stride < n_days else n_days
        cash *= np.exp(cfg.r * dt * (next_d - d))

    ST = S[:, -1]
    payoff = np.maximum(ST - cfg.K, 0.0)
    WT = cash + shares * ST - payoff
    pnl0 = WT * np.exp(-cfg.r * cfg.T)
    return pnl0, total_cost


def run_band_hedge(S, cfg: Config, band, tc_bps=None):
    """Delta-hedge with daily monitoring, but only trade when the held hedge
    ratio deviates from the current BS delta by more than `band`. The
    initial hedge at t=0 is always established. Returns (pnl0, cost)."""
    tc = (cfg.tc_bps if tc_bps is None else tc_bps) / 1e4
    n_paths, n_cols = S.shape
    n_days = n_cols - 1
    dt = cfg.T / n_days

    cash = np.full(n_paths, _initial_cash(cfg))
    shares = np.zeros(n_paths)
    total_cost = np.zeros(n_paths)

    for d in range(n_days):
        tau = cfg.T - d * dt
        Sd = S[:, d]
        target = bs_call_delta(Sd, cfg.K, cfg.r, cfg.sigma, tau)
        if d == 0:
            trade_mask = np.ones(n_paths, dtype=bool)
        else:
            trade_mask = np.abs(target - shares) > band
        trade = np.where(trade_mask, target - shares, 0.0)
        cost = tc * np.abs(trade) * Sd
        cash -= trade * Sd + cost
        shares = shares + trade
        total_cost += cost
        cash *= np.exp(cfg.r * dt)

    ST = S[:, -1]
    payoff = np.maximum(ST - cfg.K, 0.0)
    WT = cash + shares * ST - payoff
    pnl0 = WT * np.exp(-cfg.r * cfg.T)
    return pnl0, total_cost


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def summarize(pnl, cost, cvar_level=0.05):
    """mean/std of discounted terminal P&L, 5% CVaR (mean of the worst
    cvar_level fraction of outcomes), and mean total transaction cost."""
    cutoff = np.percentile(pnl, 100 * cvar_level)
    cvar = pnl[pnl <= cutoff].mean()
    return dict(
        mean=pnl.mean(),
        std=pnl.std(ddof=1),
        cvar5=cvar,
        mean_cost=cost.mean(),
    )

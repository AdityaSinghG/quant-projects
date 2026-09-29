import numpy as np
import pytest

from hedge_sim import Config, run_fixed_frequency_hedge, simulate_gbm

cfg = Config(S0=100.0, K=100.0, r=0.05, sigma=0.20, T=1.0,
             n_days=252, n_paths=10_000, seed=42, tc_bps=5.0)

S = simulate_gbm(cfg)


def test_zero_cost_std_shrinks_with_rebalancing_frequency():
    """Finer rebalancing tracks the option's delta more closely, so the
    zero-cost hedging error std should shrink as frequency increases:
    monthly > weekly > daily."""
    pnl_daily, _ = run_fixed_frequency_hedge(S, cfg, stride=1, tc_bps=0.0)
    pnl_weekly, _ = run_fixed_frequency_hedge(S, cfg, stride=5, tc_bps=0.0)
    pnl_monthly, _ = run_fixed_frequency_hedge(S, cfg, stride=21, tc_bps=0.0)

    std_daily = pnl_daily.std(ddof=1)
    std_weekly = pnl_weekly.std(ddof=1)
    std_monthly = pnl_monthly.std(ddof=1)

    assert std_daily < std_weekly < std_monthly


def test_higher_transaction_cost_gives_lower_mean_pnl():
    """Transaction costs are a pure drag on the hedger's P&L: a higher cost
    rate must reduce the mean hedging P&L for the same rebalancing rule."""
    pnl_low, _ = run_fixed_frequency_hedge(S, cfg, stride=1, tc_bps=5.0)
    pnl_high, _ = run_fixed_frequency_hedge(S, cfg, stride=1, tc_bps=50.0)

    assert pnl_high.mean() < pnl_low.mean()


@pytest.mark.parametrize("stride", [1, 5, 21])
def test_zero_cost_mean_pnl_close_to_zero(stride):
    """With no transaction costs, the BS premium received at t=0 is exactly
    the risk-neutral discounted expected payoff, so the average hedging P&L
    across many paths should be close to zero (up to Monte Carlo/discretization
    noise), regardless of rebalancing frequency."""
    pnl, _ = run_fixed_frequency_hedge(S, cfg, stride=stride, tc_bps=0.0)
    se = pnl.std(ddof=1) / np.sqrt(len(pnl))
    assert abs(pnl.mean()) < 5 * se

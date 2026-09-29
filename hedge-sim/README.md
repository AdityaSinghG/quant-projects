# hedge-sim

Delta-hedging simulator: sell one European call for the Black-Scholes
premium at t=0, then delta-hedge it under simulated GBM paths using the BS
delta, and measure the resulting terminal hedging P&L for different
rebalancing rules.

The GBM path generator and Black-Scholes formulas are copied from
[`mc-pricer`](../mc-pricer) into `hedge_sim.py` (no cross-import — this
folder is self-contained).

## Setup

- S0 = K = 100, r = 5%, sigma = 20% (hedger uses the true sigma, no vol
  misspecification), T = 1 year, 252 trading days
- 10,000 GBM paths, fixed seed (42)
- Proportional transaction cost: 5 bps of trade notional per trade
  (configurable via `Config.tc_bps`)
- Black-Scholes premium received at t=0: **10.4506**

## Strategies compared

1. **Fixed frequency**: rebalance to the BS delta every day / week (5
   trading days) / month (21 trading days). An optional intraday mode
   (`INCLUDE_INTRADAY` in `run_analysis.py`) simulates a finer grid (4-hourly
   / hourly, 8-hour trading day convention) and rebalances every step.
2. **No-trade band**: monitor the BS delta daily, but only trade when the
   held hedge ratio drifts from it by more than a threshold (0.02, 0.05,
   0.10 tested).

All strategies hedge the same short call, and fixed-frequency vs. zero-cost
runs reuse the same simulated paths (same seed) so comparisons aren't
confounded by Monte Carlo noise.

## Results

Discounted terminal hedging P&L (t=0 dollars), 5% CVaR = mean of the worst
5% of outcomes, mean total cost = average (undiscounted) transaction cost
paid per path.

| Strategy | Mean P&L | Std P&L | 5% CVaR | Mean Total Cost |
|---|---|---|---|---|
| Daily | -0.2679 | 0.4283 | -1.3215 | 0.2758 |
| Weekly | -0.1401 | 0.9127 | -2.3162 | 0.1416 |
| Monthly | -0.0646 | 1.8270 | -4.3419 | 0.0821 |
| 4-Hourly (intraday) | -0.3661 | 0.3268 | -1.2051 | 0.3771 |
| Hourly (intraday) | -0.7034 | 0.3024 | -1.3943 | 0.7220 |
| Daily (zero cost) | 0.0018 | 0.4151 | -0.9471 | 0.0000 |
| Weekly (zero cost) | -0.0012 | 0.9070 | -2.1249 | 0.0000 |
| Monthly (zero cost) | 0.0162 | 1.8222 | -4.2389 | 0.0000 |
| Band 0.02 | -0.2269 | 0.4624 | -1.3120 | 0.2318 |
| Band 0.05 | -0.1654 | 0.6358 | -1.5647 | 0.1655 |
| Band 0.10 | -0.1152 | 1.0064 | -2.2837 | 0.1153 |

**No-trade band vs. daily rebalancing** (same 5 bps cost):

| Band | Mean cost (% of daily) | Std P&L (% of daily) |
|---|---|---|
| 0.02 | 84.1% | 107.9% |
| 0.05 | 60.0% | 148.4% |
| 0.10 | 41.8% | 235.0% |

Widening the no-trade band cuts transaction costs substantially (a 0.10
band trades only ~42% of daily's cost) but the hedging error grows faster
than the cost falls — the P&L std more than doubles at the widest band
tested. The band sits strictly between daily and weekly on the cost axis
while offering a smoother cost/risk trade-off than jumping to a coarser
calendar frequency, since it only trades when the hedge actually needs it.

## Why gamma causes hedge error

A delta hedge is only exact in continuous time. Between rebalances, the
stock moves and the option's actual delta moves with it — at a rate given by
gamma (`d(delta)/dS`), which is largest for this at-the-money, one-year call.
Over a discrete interval, the hedger's P&L on a delta-neutral book is, to
second order,

```
d(hedge P&L) ~ -1/2 * Gamma * (dS)^2 + theta*dt  (theta/gamma offset in expectation, noise in realization)
```

The `(dS)^2` term is what actually varies path to path: realized local
variance over the interval fluctuates around its expected value `sigma^2 *
S^2 * dt`, and gamma converts that fluctuation directly into hedging P&L
noise. A wider rebalancing interval lets `dS` (and hence the delta
mismatch) grow larger before it's corrected, so the variance of the
hedging error scales up with the interval length — which is exactly the
monotone std pattern from monthly > weekly > daily > 4-hourly > hourly in
the zero-cost rows above. In the continuous-time limit the gamma/theta
terms cancel exactly (the classic Black-Scholes replication argument), so
mean P&L is a pure Monte Carlo noise around zero when there are no costs;
transaction costs then bias the mean P&L downward without changing that
gamma-driven variance story, and more frequent trading pays that bias more
often (see the cost-vs-std trade-off in the plot below).

## Plots

- `plots/pnl_hist_daily.png`, `plots/pnl_hist_weekly.png`,
  `plots/pnl_hist_monthly.png` — P&L histograms per fixed frequency (5 bps
  cost)
- `plots/std_vs_cost.png` — hedging error std vs. mean transaction cost
  across every strategy tested (fixed frequencies, intraday, and no-trade
  bands), showing the cost/risk frontier

## Running

```
pip install -r requirements.txt
python run_analysis.py   # prints the table above, writes results_table.md, saves plots/*.png
pytest -v                # 5 tests
```

## Tests (`test_hedge_sim.py`)

1. `test_zero_cost_std_shrinks_with_rebalancing_frequency` — zero-cost
   hedging error std strictly decreases monthly > weekly > daily.
2. `test_higher_transaction_cost_gives_lower_mean_pnl` — raising the cost
   rate (5 bps -> 50 bps) lowers mean P&L for the same rebalancing rule.
3. `test_zero_cost_mean_pnl_close_to_zero` (parametrized over daily/weekly/
   monthly) — with no transaction costs, mean hedging P&L is within 5
   standard errors of zero, since the BS premium is the exact risk-neutral
   discounted expected payoff.

All 5 tests pass.

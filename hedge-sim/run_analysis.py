"""Run the full delta-hedging simulation analysis.

Sells one European call at t=0 for the Black-Scholes premium and delta-hedges
it under GBM, comparing fixed rebalancing frequencies (daily/weekly/monthly,
plus optional intraday) against a no-trade-band strategy, with proportional
transaction costs. Prints a results table, writes it to results_table.md,
and saves PNG plots.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from hedge_sim import (
    Config,
    run_band_hedge,
    run_fixed_frequency_hedge,
    simulate_gbm,
    summarize,
)

# Set INCLUDE_INTRADAY = True to also hedge on a finer intraday grid
# (4-hourly / hourly), using an 8-hour trading day convention.
INCLUDE_INTRADAY = True

cfg = Config(S0=100.0, K=100.0, r=0.05, sigma=0.20, T=1.0,
             n_days=252, n_paths=10_000, seed=42, tc_bps=5.0)

rows = []       # results table rows
scatter_pts = []  # (label, mean_cost, std) for the cost-vs-risk plot


def section(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def add_row(name, pnl, cost):
    m = summarize(pnl, cost)
    rows.append(dict(name=name, **m))
    scatter_pts.append((name, m["mean_cost"], m["std"]))
    print(f"{name:<28}{m['mean']:>12.4f}{m['std']:>12.4f}{m['cvar5']:>12.4f}{m['mean_cost']:>14.4f}")
    return m


def header():
    h = f"{'Strategy':<28}{'Mean P&L':>12}{'Std P&L':>12}{'5% CVaR':>12}{'Mean Cost':>14}"
    print(h)
    print("-" * len(h))


# --------------------------------------------------------------------------
# 1. Base simulation: 10,000 GBM paths, 252 trading days
# --------------------------------------------------------------------------
section("1. Simulating GBM paths")
S = simulate_gbm(cfg)
print(f"S0={cfg.S0} K={cfg.K} r={cfg.r} sigma={cfg.sigma} T={cfg.T} "
      f"n_days={cfg.n_days} n_paths={cfg.n_paths} seed={cfg.seed}")
print(f"Terminal price: mean={S[:, -1].mean():.4f}, std={S[:, -1].std():.4f}")

from hedge_sim import bs_call_price
premium = bs_call_price(cfg.S0, cfg.K, cfg.r, cfg.sigma, cfg.T)
print(f"Black-Scholes call premium received at t=0: {premium:.4f}")

# --------------------------------------------------------------------------
# 2. Fixed rebalancing frequencies: daily, weekly, monthly (5 bps cost)
# --------------------------------------------------------------------------
section(f"2. Fixed-frequency hedging (transaction cost = {cfg.tc_bps} bps/trade)")
header()

freqs = {"Daily": 1, "Weekly": 5, "Monthly": 21}
freq_results = {}
for name, stride in freqs.items():
    pnl, cost = run_fixed_frequency_hedge(S, cfg, stride)
    freq_results[name] = (pnl, cost)
    add_row(name, pnl, cost)

# --------------------------------------------------------------------------
# 2b. Optional intraday frequencies on a finer grid (configurable)
# --------------------------------------------------------------------------
if INCLUDE_INTRADAY:
    section("2b. Optional intraday hedging (configurable, 8h trading day)")
    header()
    intraday_defs = {"4-Hourly": 2, "Hourly": 8}  # steps per day on a finer grid
    for name, steps_per_day in intraday_defs.items():
        S_fine = simulate_gbm(cfg, n_steps=cfg.n_days * steps_per_day)
        pnl, cost = run_fixed_frequency_hedge(S_fine, cfg, stride=1)
        add_row(name, pnl, cost)

# --------------------------------------------------------------------------
# 3. Zero-cost sanity check (no transaction costs)
# --------------------------------------------------------------------------
section("3. Zero-cost sanity check (frictionless hedging)")
header()
zero_cost_results = {}
for name, stride in freqs.items():
    pnl0, cost0 = run_fixed_frequency_hedge(S, cfg, stride, tc_bps=0.0)
    zero_cost_results[name] = (pnl0, cost0)
    add_row(f"{name} (zero cost)", pnl0, cost0)

# --------------------------------------------------------------------------
# 4. No-trade band strategy: 2-3 thresholds vs daily rebalancing
# --------------------------------------------------------------------------
section(f"4. No-trade band hedging (transaction cost = {cfg.tc_bps} bps/trade)")
header()
bands = [0.02, 0.05, 0.10]
band_results = {}
for band in bands:
    pnl, cost = run_band_hedge(S, cfg, band)
    band_results[band] = (pnl, cost)
    add_row(f"Band {band:.2f}", pnl, cost)

daily_pnl, daily_cost = freq_results["Daily"]
print("\nBand vs. daily rebalancing:")
for band in bands:
    pnl, cost = band_results[band]
    print(f"  band={band:.2f}: mean cost {cost.mean():.4f} "
          f"({cost.mean() / daily_cost.mean() * 100:.1f}% of daily), "
          f"std P&L {pnl.std(ddof=1):.4f} "
          f"({pnl.std(ddof=1) / daily_pnl.std(ddof=1) * 100:.1f}% of daily)")

# --------------------------------------------------------------------------
# 5. Plots: P&L histograms per frequency, and std vs cost scatter
# --------------------------------------------------------------------------
section("5. Plots")

for name in ["Daily", "Weekly", "Monthly"]:
    pnl, _ = freq_results[name]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(pnl, bins=60, color="#4472C4", alpha=0.85, edgecolor="white")
    ax.axvline(0, color="black", linewidth=1, linestyle="--")
    ax.set_xlabel("Discounted terminal hedging P&L")
    ax.set_ylabel("Number of paths")
    ax.set_title(f"{name} Rebalancing -- Hedging P&L Distribution\n"
                 f"mean={pnl.mean():.3f}, std={pnl.std(ddof=1):.3f}")
    fig.tight_layout()
    fname = f"plots/pnl_hist_{name.lower()}.png"
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"Saved {fname}")

fig, ax = plt.subplots(figsize=(7.5, 5.5))
for label, cost, std in scatter_pts:
    if "zero cost" in label:
        continue
    ax.scatter(cost, std, s=60)
    ax.annotate(label, (cost, std), textcoords="offset points", xytext=(6, 4), fontsize=8)
ax.set_xlabel("Mean total transaction cost per path")
ax.set_ylabel("Std dev of hedging P&L")
ax.set_title("Hedging Error Std vs. Transaction Cost Across Strategies")
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig("plots/std_vs_cost.png", dpi=150)
plt.close(fig)
print("Saved plots/std_vs_cost.png")

# --------------------------------------------------------------------------
# 6. Results table
# --------------------------------------------------------------------------
section("Summary Results Table")
header()
for row in rows:
    print(f"{row['name']:<28}{row['mean']:>12.4f}{row['std']:>12.4f}"
          f"{row['cvar5']:>12.4f}{row['mean_cost']:>14.4f}")

with open("results_table.md", "w") as f:
    f.write(f"Black-Scholes call premium at t=0: **{premium:.4f}**\n\n")
    f.write(f"Config: S0=K={cfg.S0}, r={cfg.r}, sigma={cfg.sigma}, T={cfg.T}yr, "
            f"{cfg.n_days} trading days, {cfg.n_paths:,} paths, seed={cfg.seed}, "
            f"cost={cfg.tc_bps}bps/trade\n\n")
    f.write("| Strategy | Mean P&L | Std P&L | 5% CVaR | Mean Total Cost |\n")
    f.write("|---|---|---|---|---|\n")
    for row in rows:
        f.write(f"| {row['name']} | {row['mean']:.4f} | {row['std']:.4f} | "
                f"{row['cvar5']:.4f} | {row['mean_cost']:.4f} |\n")

print("\nWrote results_table.md")

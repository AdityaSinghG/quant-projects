"""Run the full Monte Carlo exotic options analysis.

Prices a European call (validated vs Black-Scholes), an arithmetic Asian
call (control variate = European price), and an up-and-out barrier call
(antithetic variates); benchmarks the NumPy vs Numba simulators; computes
Delta/Vega by finite differences with common random numbers; and produces
a convergence plot. Prints a results table and writes it to
results_table.md, and saves convergence.png.
"""
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mc_pricer import (
    Params,
    asian_call_payoff,
    bs_call_delta,
    bs_call_price,
    bs_call_vega,
    control_variate_price,
    draw_normals,
    european_call_payoff,
    price_from_payoffs,
    simulate_paths_numba,
    simulate_paths_numpy,
    up_and_out_barrier_call_payoff,
)

BASE = Params(S0=100.0, K=100.0, r=0.05, sigma=0.2, T=1.0,
              n_steps=252, n_paths=200_000, barrier=120.0, seed=42)

rows = []  # rows for the final summary table


def section(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# --------------------------------------------------------------------------
# 1. European call: plain MC vs antithetic MC, validated against BS
# --------------------------------------------------------------------------
section("1. European Call -- Monte Carlo vs Black-Scholes")

bs_price = bs_call_price(BASE.S0, BASE.K, BASE.r, BASE.sigma, BASE.T)
print(f"Black-Scholes price: {bs_price:.6f}")

Z_plain = draw_normals(BASE.n_paths, BASE.n_steps, BASE.seed, antithetic=False)
paths_plain = simulate_paths_numpy(BASE.S0, BASE.r, BASE.sigma, BASE.T, Z_plain)
payoff_euro_plain = european_call_payoff(paths_plain, BASE.K)
price_euro_plain, se_euro_plain = price_from_payoffs(payoff_euro_plain, BASE.r, BASE.T)

Z_anti = draw_normals(BASE.n_paths, BASE.n_steps, BASE.seed, antithetic=True)
paths_anti = simulate_paths_numpy(BASE.S0, BASE.r, BASE.sigma, BASE.T, Z_anti)
payoff_euro_anti = european_call_payoff(paths_anti, BASE.K)
price_euro_anti, se_euro_anti = price_from_payoffs(payoff_euro_anti, BASE.r, BASE.T, antithetic=True)

vr_euro = (1 - (se_euro_anti / se_euro_plain) ** 2) * 100

print(f"Plain MC:      price={price_euro_plain:.6f}  se={se_euro_plain:.6f}  "
      f"err={abs(price_euro_plain - bs_price):.6f}")
print(f"Antithetic MC: price={price_euro_anti:.6f}  se={se_euro_anti:.6f}  "
      f"err={abs(price_euro_anti - bs_price):.6f}  (variance reduction {vr_euro:.1f}%)")

rows.append(dict(name="European (plain)", price=price_euro_plain, se=se_euro_plain,
                  err=abs(price_euro_plain - bs_price), vr="--"))
rows.append(dict(name="European (antithetic)", price=price_euro_anti, se=se_euro_anti,
                  err=abs(price_euro_anti - bs_price), vr=f"{vr_euro:.1f}%"))

# --------------------------------------------------------------------------
# 2. Asian call: plain MC vs control-variate MC (control = European price)
# --------------------------------------------------------------------------
section("2. Asian (arithmetic average) Call -- plain vs control variate")

payoff_asian_plain = asian_call_payoff(paths_plain, BASE.K)
price_asian_plain, se_asian_plain = price_from_payoffs(payoff_asian_plain, BASE.r, BASE.T)

price_asian_cv, se_asian_cv, beta = control_variate_price(
    payoff_asian_plain, payoff_euro_plain, bs_price, BASE.r, BASE.T)

vr_asian = (1 - (se_asian_cv / se_asian_plain) ** 2) * 100

print(f"Plain MC:           price={price_asian_plain:.6f}  se={se_asian_plain:.6f}")
print(f"Control variate MC: price={price_asian_cv:.6f}  se={se_asian_cv:.6f}  "
      f"(beta={beta:.4f}, variance reduction {vr_asian:.1f}%)")

rows.append(dict(name="Asian (plain)", price=price_asian_plain, se=se_asian_plain,
                  err="n/a", vr="--"))
rows.append(dict(name="Asian (control variate)", price=price_asian_cv, se=se_asian_cv,
                  err="n/a", vr=f"{vr_asian:.1f}%"))

# --------------------------------------------------------------------------
# 3. Up-and-out barrier call: plain MC vs antithetic MC
# --------------------------------------------------------------------------
section("3. Up-and-out Barrier Call -- plain vs antithetic")
print(f"Barrier level: {BASE.barrier}")

payoff_bar_plain = up_and_out_barrier_call_payoff(paths_plain, BASE.K, BASE.barrier)
price_bar_plain, se_bar_plain = price_from_payoffs(payoff_bar_plain, BASE.r, BASE.T)

payoff_bar_anti = up_and_out_barrier_call_payoff(paths_anti, BASE.K, BASE.barrier)
price_bar_anti, se_bar_anti = price_from_payoffs(payoff_bar_anti, BASE.r, BASE.T, antithetic=True)

vr_bar = (1 - (se_bar_anti / se_bar_plain) ** 2) * 100

print(f"Plain MC:      price={price_bar_plain:.6f}  se={se_bar_plain:.6f}")
print(f"Antithetic MC: price={price_bar_anti:.6f}  se={se_bar_anti:.6f}  "
      f"(variance reduction {vr_bar:.1f}%)")

rows.append(dict(name="Barrier (plain)", price=price_bar_plain, se=se_bar_plain,
                  err="n/a", vr="--"))
rows.append(dict(name="Barrier (antithetic)", price=price_bar_anti, se=se_bar_anti,
                  err="n/a", vr=f"{vr_bar:.1f}%"))

# --------------------------------------------------------------------------
# 4. Benchmark: vectorized NumPy vs Numba njit simulator
# --------------------------------------------------------------------------
section("4. Simulator benchmark -- vectorized NumPy vs Numba njit")

bench_n_paths = 500_000
bench_n_steps = 252
Z_bench = draw_normals(bench_n_paths, bench_n_steps, BASE.seed)

# Warm up JIT compilation (not timed).
_ = simulate_paths_numba(BASE.S0, BASE.r, BASE.sigma, BASE.T, Z_bench[:100])

n_reps = 5
numpy_times = []
numba_times = []
for _ in range(n_reps):
    t0 = time.perf_counter()
    paths_np_bench = simulate_paths_numpy(BASE.S0, BASE.r, BASE.sigma, BASE.T, Z_bench)
    numpy_times.append(time.perf_counter() - t0)

    t0 = time.perf_counter()
    paths_nb_bench = simulate_paths_numba(BASE.S0, BASE.r, BASE.sigma, BASE.T, Z_bench)
    numba_times.append(time.perf_counter() - t0)

assert np.allclose(paths_np_bench, paths_nb_bench, atol=1e-8), "NumPy and Numba paths disagree!"

t_numpy = min(numpy_times)
t_numba = min(numba_times)
speedup = t_numpy / t_numba

print(f"Paths x steps: {bench_n_paths:,} x {bench_n_steps}, best of {n_reps} runs")
print(f"NumPy (vectorized): {t_numpy * 1000:.1f} ms")
print(f"Numba (njit, parallel): {t_numba * 1000:.1f} ms")
print(f"Speedup: {speedup:.2f}x")
print("(Outputs verified identical to 1e-8 tolerance.)")

# --------------------------------------------------------------------------
# 5. Greeks by finite differences with common random numbers (CRN)
# --------------------------------------------------------------------------
section("5. Delta & Vega -- finite differences with common random numbers")

h_S = 1.0
h_sigma = 0.01
Z_greeks = draw_normals(BASE.n_paths, BASE.n_steps, BASE.seed, antithetic=True)


def price_for(S0, sigma, payoff_fn, **kw):
    paths = simulate_paths_numpy(S0, BASE.r, sigma, BASE.T, Z_greeks)
    payoff = payoff_fn(paths, **kw)
    return price_from_payoffs(payoff, BASE.r, BASE.T, antithetic=True)[0]


def fd_greeks(payoff_fn, **kw):
    p_s_up = price_for(BASE.S0 + h_S, BASE.sigma, payoff_fn, **kw)
    p_s_dn = price_for(BASE.S0 - h_S, BASE.sigma, payoff_fn, **kw)
    delta = (p_s_up - p_s_dn) / (2 * h_S)

    p_v_up = price_for(BASE.S0, BASE.sigma + h_sigma, payoff_fn, **kw)
    p_v_dn = price_for(BASE.S0, BASE.sigma - h_sigma, payoff_fn, **kw)
    vega = (p_v_up - p_v_dn) / (2 * h_sigma)
    return delta, vega


delta_euro, vega_euro = fd_greeks(european_call_payoff, K=BASE.K)
bs_delta = bs_call_delta(BASE.S0, BASE.K, BASE.r, BASE.sigma, BASE.T)
bs_vega = bs_call_vega(BASE.S0, BASE.K, BASE.r, BASE.sigma, BASE.T)
print(f"European: delta={delta_euro:.6f} (BS={bs_delta:.6f}, "
      f"diff={abs(delta_euro - bs_delta):.6f})")
print(f"          vega ={vega_euro:.6f} (BS={bs_vega:.6f}, "
      f"diff={abs(vega_euro - bs_vega):.6f})")

delta_asian, vega_asian = fd_greeks(asian_call_payoff, K=BASE.K)
print(f"Asian:    delta={delta_asian:.6f}  vega={vega_asian:.6f}")

delta_bar, vega_bar = fd_greeks(up_and_out_barrier_call_payoff, K=BASE.K, barrier=BASE.barrier)
print(f"Barrier:  delta={delta_bar:.6f}  vega={vega_bar:.6f}")

# --------------------------------------------------------------------------
# 6. Convergence plot: error vs number of paths, with/without antithetic
# --------------------------------------------------------------------------
section("6. Convergence plot -- error vs number of paths")

n_list = np.unique(np.geomspace(1_000, 1_000_000, 14).astype(int))
n_list = np.array([n + (n % 2) for n in n_list])  # keep even (needed for antithetic split)
n_repeats = 8  # independent repeats per n, averaged, to smooth single-draw noise

errors_plain = []
errors_anti = []
for n in n_list:
    plain_errs = []
    anti_errs = []
    for rep in range(n_repeats):
        seed = 1000 * int(n) + rep
        Z_p = draw_normals(n, 1, seed, antithetic=False)  # n_steps=1: exact terminal GBM draw
        p_plain = simulate_paths_numpy(BASE.S0, BASE.r, BASE.sigma, BASE.T, Z_p)
        price_p, _ = price_from_payoffs(european_call_payoff(p_plain, BASE.K), BASE.r, BASE.T)
        plain_errs.append(abs(price_p - bs_price))

        Z_a = draw_normals(n, 1, seed, antithetic=True)
        p_anti = simulate_paths_numpy(BASE.S0, BASE.r, BASE.sigma, BASE.T, Z_a)
        price_a, _ = price_from_payoffs(european_call_payoff(p_anti, BASE.K), BASE.r, BASE.T,
                                         antithetic=True)
        anti_errs.append(abs(price_a - bs_price))
    errors_plain.append(np.mean(plain_errs))
    errors_anti.append(np.mean(anti_errs))

fig, ax = plt.subplots(figsize=(7.5, 5.5))
ax.loglog(n_list, errors_plain, "o-", label="Plain MC")
ax.loglog(n_list, errors_anti, "s-", label="Antithetic MC")
ref = errors_plain[0] * np.sqrt(n_list[0] / n_list)
ax.loglog(n_list, ref, "k--", alpha=0.5, label=r"$O(1/\sqrt{n})$ reference")
ax.set_xlabel("Number of paths")
ax.set_ylabel(f"Mean absolute error vs BS price (avg of {n_repeats} runs)")
ax.set_title("European Call MC Convergence")
ax.legend()
ax.grid(True, which="both", alpha=0.3)
fig.tight_layout()
fig.savefig("convergence.png", dpi=150)
print("Saved convergence.png")

# --------------------------------------------------------------------------
# 7. Summary table
# --------------------------------------------------------------------------
section("Summary Results Table")

header = f"{'Option':<26}{'Price':>10}{'Std Err':>12}{'Err vs BS':>12}{'Var. Reduction':>16}"
sep = "-" * len(header)
print(header)
print(sep)
lines_md = ["| Option | Price | Std Error | Error vs BS | Variance Reduction |",
            "|---|---|---|---|---|"]
for row in rows:
    err_str = f"{row['err']:.6f}" if isinstance(row["err"], float) else row["err"]
    print(f"{row['name']:<26}{row['price']:>10.6f}{row['se']:>12.6f}{err_str:>12}{row['vr']:>16}")
    lines_md.append(f"| {row['name']} | {row['price']:.6f} | {row['se']:.6f} | {err_str} | {row['vr']} |")

print(sep)
print(f"Black-Scholes reference price: {bs_price:.6f}")
print(f"\nSimulator benchmark ({bench_n_paths:,} paths x {bench_n_steps} steps, "
      f"best of {n_reps}):")
print(f"  NumPy:  {t_numpy * 1000:.1f} ms")
print(f"  Numba:  {t_numba * 1000:.1f} ms")
print(f"  Speedup: {speedup:.2f}x")
print(f"\nGreeks (finite difference, CRN, h_S={h_S}, h_sigma={h_sigma}):")
print(f"  European: delta={delta_euro:.6f} (BS {bs_delta:.6f})  vega={vega_euro:.6f} (BS {bs_vega:.6f})")
print(f"  Asian:    delta={delta_asian:.6f}  vega={vega_asian:.6f}")
print(f"  Barrier:  delta={delta_bar:.6f}  vega={vega_bar:.6f}")

with open("results_table.md", "w") as f:
    f.write(f"Black-Scholes reference price: **{bs_price:.6f}**\n\n")
    f.write("\n".join(lines_md))
    f.write("\n\n### Simulator benchmark\n\n")
    f.write(f"NumPy vs Numba njit, {bench_n_paths:,} paths x {bench_n_steps} steps, best of {n_reps} runs:\n\n")
    f.write("| Simulator | Time (ms) |\n|---|---|\n")
    f.write(f"| NumPy (vectorized) | {t_numpy * 1000:.1f} |\n")
    f.write(f"| Numba (njit, parallel) | {t_numba * 1000:.1f} |\n\n")
    f.write(f"**Speedup: {speedup:.2f}x**\n\n")
    f.write("### Greeks (finite differences, common random numbers)\n\n")
    f.write(f"h_S = {h_S}, h_sigma = {h_sigma}\n\n")
    f.write("| Option | Delta | Vega | BS Delta | BS Vega |\n|---|---|---|---|---|\n")
    f.write(f"| European | {delta_euro:.6f} | {vega_euro:.6f} | {bs_delta:.6f} | {bs_vega:.6f} |\n")
    f.write(f"| Asian | {delta_asian:.6f} | {vega_asian:.6f} | n/a | n/a |\n")
    f.write(f"| Barrier | {delta_bar:.6f} | {vega_bar:.6f} | n/a | n/a |\n")

print("\nWrote results_table.md")

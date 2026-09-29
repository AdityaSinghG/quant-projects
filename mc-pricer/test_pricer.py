import numpy as np
import pytest

from mc_pricer import (
    Params,
    asian_call_payoff,
    bs_call_price,
    control_variate_price,
    draw_normals,
    european_call_payoff,
    price_from_payoffs,
    simulate_paths_numba,
    simulate_paths_numpy,
    up_and_out_barrier_call_payoff,
)

P = Params(S0=100.0, K=100.0, r=0.05, sigma=0.2, T=1.0,
           n_steps=100, n_paths=200_000, barrier=120.0, seed=123)


def test_bs_price_known_value():
    # Standard textbook case: S=K=100, r=5%, sigma=20%, T=1 -> ~10.4506
    price = bs_call_price(100, 100, 0.05, 0.2, 1.0)
    assert price == pytest.approx(10.4506, abs=1e-3)


def test_european_mc_matches_bs_within_confidence_interval():
    Z = draw_normals(P.n_paths, P.n_steps, P.seed, antithetic=False)
    paths = simulate_paths_numpy(P.S0, P.r, P.sigma, P.T, Z)
    payoff = european_call_payoff(paths, P.K)
    price, se = price_from_payoffs(payoff, P.r, P.T)

    bs_price = bs_call_price(P.S0, P.K, P.r, P.sigma, P.T)
    assert abs(price - bs_price) < 4 * se


def test_antithetic_reduces_standard_error():
    Z_plain = draw_normals(P.n_paths, P.n_steps, P.seed, antithetic=False)
    paths_plain = simulate_paths_numpy(P.S0, P.r, P.sigma, P.T, Z_plain)
    _, se_plain = price_from_payoffs(european_call_payoff(paths_plain, P.K), P.r, P.T)

    Z_anti = draw_normals(P.n_paths, P.n_steps, P.seed, antithetic=True)
    paths_anti = simulate_paths_numpy(P.S0, P.r, P.sigma, P.T, Z_anti)
    _, se_anti = price_from_payoffs(european_call_payoff(paths_anti, P.K), P.r, P.T,
                                     antithetic=True)

    assert se_anti < se_plain


def test_control_variate_reduces_standard_error_for_asian():
    Z = draw_normals(P.n_paths, P.n_steps, P.seed, antithetic=False)
    paths = simulate_paths_numpy(P.S0, P.r, P.sigma, P.T, Z)
    payoff_asian = asian_call_payoff(paths, P.K)
    payoff_euro = european_call_payoff(paths, P.K)
    bs_price = bs_call_price(P.S0, P.K, P.r, P.sigma, P.T)

    _, se_plain = price_from_payoffs(payoff_asian, P.r, P.T)
    _, se_cv, _ = control_variate_price(payoff_asian, payoff_euro, bs_price, P.r, P.T)

    assert se_cv < se_plain


def test_asian_call_cheaper_than_european_call():
    # Averaging reduces effective volatility, so the arithmetic Asian call
    # should be worth less than the European call on the same paths.
    Z = draw_normals(P.n_paths, P.n_steps, P.seed, antithetic=False)
    paths = simulate_paths_numpy(P.S0, P.r, P.sigma, P.T, Z)
    price_asian, _ = price_from_payoffs(asian_call_payoff(paths, P.K), P.r, P.T)
    price_euro, _ = price_from_payoffs(european_call_payoff(paths, P.K), P.r, P.T)
    assert price_asian < price_euro


def test_barrier_call_cheaper_than_vanilla_european_call():
    Z = draw_normals(P.n_paths, P.n_steps, P.seed, antithetic=False)
    paths = simulate_paths_numpy(P.S0, P.r, P.sigma, P.T, Z)
    price_barrier, _ = price_from_payoffs(
        up_and_out_barrier_call_payoff(paths, P.K, P.barrier), P.r, P.T)
    price_euro, _ = price_from_payoffs(european_call_payoff(paths, P.K), P.r, P.T)
    assert price_barrier < price_euro


def test_numba_simulator_matches_numpy_simulator():
    Z = draw_normals(1000, P.n_steps, P.seed, antithetic=False)
    paths_np = simulate_paths_numpy(P.S0, P.r, P.sigma, P.T, Z)
    paths_nb = simulate_paths_numba(P.S0, P.r, P.sigma, P.T, Z)
    assert np.allclose(paths_np, paths_nb, atol=1e-8)

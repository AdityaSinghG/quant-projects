Black-Scholes reference price: **10.450584**

| Option | Price | Std Error | Error vs BS | Variance Reduction |
|---|---|---|---|---|
| European (plain) | 10.504703 | 0.033110 | 0.054119 | -- |
| European (antithetic) | 10.460014 | 0.023399 | 0.009431 | 50.1% |
| Asian (plain) | 5.804757 | 0.017935 | n/a | -- |
| Asian (control variate) | 5.780086 | 0.009687 | n/a | 70.8% |
| Barrier (plain) | 1.318460 | 0.007575 | n/a | -- |
| Barrier (antithetic) | 1.316179 | 0.007097 | n/a | 12.2% |

### Simulator benchmark

NumPy vs Numba njit, 500,000 paths x 252 steps, best of 5 runs:

| Simulator | Time (ms) |
|---|---|
| NumPy (vectorized) | 6155.8 |
| Numba (njit, parallel) | 2218.2 |

**Speedup: 2.78x**

### Greeks (finite differences, common random numbers)

h_S = 1.0, h_sigma = 0.01

| Option | Delta | Vega | BS Delta | BS Vega |
|---|---|---|---|---|
| European | 0.637140 | 37.588867 | 0.636831 | 37.524035 |
| Asian | 0.588598 | 21.821929 | n/a | n/a |
| Barrier | -0.019950 | -13.531247 | n/a | n/a |

# Bayesian Return Prediction on NIFTY 50

> *Does yesterday's market volatility tell us anything about today's return?*

A full Bayesian inference pipeline in Julia — live data fetch, probabilistic model, NUTS sampling, and posterior analysis. Built to explore Julia's probabilistic programming ecosystem via a question that actually matters in quantitative finance.

**Data:** NIFTY 50 daily closes, pulled live from Yahoo Finance (~3 years)  
**Sampler:** NUTS (No-U-Turn Sampler) via [Turing.jl](https://turinglang.org/)  
**Author:** Krishna Vallabha Goswami | IIT Madras BS Data Science

---

## The Question

Standard regression would give you a point estimate: *"beta = 0.03, so volatility has a small positive effect."* But in finance, that's not enough. You need to know:

- **How confident are you?** Is beta reliably positive, or does it plausibly include zero?
- **What's the noise level?** Is the model capturing signal or just fitting noise?
- **How does uncertainty propagate** into predictions?

Bayesian inference answers all three. Instead of a single number for each parameter, you get a full posterior distribution.

---

## Model

```
alpha ~ Normal(0, 0.1)       # baseline daily return (prior: small, near zero)
beta  ~ Normal(0, 1)         # effect of yesterday's volatility (prior: uninformative)
sigma ~ Exponential(1)       # observation noise (prior: positive, weakly regularised)

mu[i] = alpha + beta * volatility_lag[i]
return_today[i] ~ Normal(mu[i], sigma)
```

**Features (both standardised):**
- `volatility_lag` — absolute value of yesterday's log return (`|r_{t-1}|`)
- `return_today` — today's log return (`log(P_t / P_{t-1})`)

The choice of weakly informative priors is deliberate: the prior for `beta` is wide enough to let the data speak. If the data says beta is near zero, the posterior will reflect that.

---

## Results

### Posterior Distributions
![Posterior Distributions](plots/posterior_distributions.png)

The red dashed line marks zero. Key observation: **beta's posterior straddles zero** — meaning lagged volatility does not reliably predict the direction of tomorrow's return. This is consistent with the Efficient Market Hypothesis. The baseline return `alpha` is also near zero, as expected for a large-cap index.

### Trace Plots
![Trace Plots](plots/trace_plots.png)

Well-mixed chains with no drift — the NUTS sampler has converged. All R-hat values are close to 1.0 (see terminal output). This means the 1000 posterior samples are reliable draws from the true posterior.

### Credible Intervals
![Credible Intervals](plots/credible_intervals.png)

The 95% credible interval for `beta` crosses zero, confirming the finding above. The high `sigma` posterior (~1.0 on the standardised scale) tells us that most of the variance in daily returns is unexplained — the model is honestly saying *"I don't know"*, which is the right answer for daily stock returns.

### Posterior Predictive Check
![Posterior Predictive Check](plots/posterior_predictive_check.png)

Predicted vs actual returns using 300 posterior draws. The weak correlation is expected — if returns were easily predictable, they would already be arbitraged away. The scatter confirms the model is not overconfident.

### Raw Return Series
![Return Series](plots/return_series.png)

NIFTY 50 daily returns (blue) vs lagged volatility (orange). Notice that volatility clusters (large moves tend to follow large moves) but the *direction* of the next day's return is not predictable from magnitude alone.

---

## Why the "uncertain" result is the right result

A frequentist regression might report *"p = 0.04, beta is significant"* and call it a day. The Bayesian approach is more honest:

- The **posterior for beta is wide** — even if the mean is slightly positive, there is substantial probability mass below zero
- The **posterior for sigma is high** — the noise in daily returns dominates any predictive signal
- The **posterior predictive distribution** shows that individual-day predictions are essentially useless

This is a feature, not a bug. Bayesian inference gives you calibrated uncertainty. In finance, overconfident models are dangerous.

---

## Setup

```julia
# Clone
git clone https://github.com/yourusername/bayesian-nifty-returns
cd bayesian-nifty-returns

# Install dependencies
julia --project=. -e "using Pkg; Pkg.instantiate()"

# Run (fetches live data, samples, saves plots)
julia --project=. src/bayesian_regression.jl
```

**Julia 1.9+** required. First run takes ~4–6 minutes (package precompilation + NUTS warmup). Subsequent runs are faster.

The script pulls data live from Yahoo Finance — no manual CSV download needed.

---

## Package stack

| Package | Role |
|---|---|
| `Turing.jl` | Probabilistic model definition + NUTS sampling |
| `Distributions.jl` | Prior and likelihood distributions |
| `StatsPlots.jl` | Posterior density plots, trace plots, bar charts |
| `HTTP.jl` + `CSV.jl` | Live data fetch from Yahoo Finance |
| `DataFrames.jl` | Data wrangling |
| `Optim.jl` | MAP estimate for frequentist comparison |

---

## What I learned

**On Julia:** The `@model` macro in Turing.jl is surprisingly expressive — the model definition reads almost like the mathematical notation in a paper. Multiple dispatch means that switching between CPU and GPU sampling (via CUDA.jl) requires almost no code changes, which mirrors what I'm working on in my GSoC project on sparse GNN operations.

**On NUTS:** The No-U-Turn Sampler adapts its step size and trajectory length automatically during warmup. This is analogous to how cuSPARSE's SpMM kernel selection adapts to matrix structure — both are cases where the algorithm self-tunes based on the problem geometry.

**On Zygote.jl:** Turing uses Zygote for gradient computation during HMC. Working through how Zygote differentiates through the model's log-probability gave me useful intuition for the sparse backward pass problem in my GSoC proposal — specifically around how Zygote handles non-standard Julia types, which is the same challenge with `SparseMatrixCSC` and `CuSparseMatrixCSR`.

**On finance:** The Efficient Market Hypothesis is not just theory — it shows up clearly in posterior distributions. The high `sigma` posterior is an honest admission that daily returns are mostly noise.

---

## Connection to GSoC 2026

This project sits outside my GSoC proposal on purpose — it's a genuine exploration of Julia's ML ecosystem, not a proof-of-concept for the GNN work. That said, there are real technical overlaps:

- Turing's use of Zygote for gradient computation is directly relevant to the sparse backward pass problem I address in my proposal (§7.2)
- The dispatch model in Turing.jl (different samplers, different backends, one model definition) is the same pattern used in `GNNlib.jl` to switch between gather/scatter and SpMM execution paths
- Working in Julia outside of GNNs helped me become comfortable with the type system and multiple dispatch, which are essential for the low-level CUDA.jl work in my proposal

---

## References

- Ge et al., *Turing: A Language for Flexible Probabilistic Inference*, AISTATS 2018
- Fama, E., *Efficient Capital Markets: A Review of Theory and Evidence*, Journal of Finance, 1970
- [Turing.jl documentation](https://turinglang.org/docs/)
- [Yahoo Finance NIFTY 50](https://finance.yahoo.com/quote/%5ENSEI/)

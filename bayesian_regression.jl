#  Bayesian linear regression with turing.jl    
# Question: Does yesterday's volatility predict today's return?
#data : NIFTY50 Index pulled live from Yahoo Finance 

using Turing
using Distributions
using DataFrames
using CSV
using HTTP
using StatsPlots
using LinearAlgebra
using Statistics
using Random
using Dates
using Optim
 using DataFrames: nrow
Random.seed!(42)

# Load data from Yahoo Finance

println(" Bayesian Return Prediction on NIFTY50 Index using Turing.jl   ")

println("Loading data from Yahoo Finance...")
# Yahoo Finance API: period1/period2 are Unix timestamps
# Pulling ~3 years of daily data
period1 = string(round(Int, datetime2unix(DateTime(2021, 1, 1))))
period2 = string(round(Int, datetime2unix(DateTime(2024, 6, 1))))
ticker  = "%5ENSEI"   # ^NSEI = NIFTY 50
 
url = "https://query1.finance.yahoo.com/v7/finance/download/$(ticker)" *
      "?period1=$(period1)&period2=$(period2)&interval=1d&events=history"
 
headers = ["User-Agent" => "Mozilla/5.0"]
resp    = HTTP.get(url; headers=headers)
println("HTTP Status: ", resp.status)
println(String(resp.body)[1:200])  # print first 200 chars
df_raw  = CSV.read(IOBuffer(resp.body), DataFrame)
 
println("Downloaded $(nrow(df_raw)) trading days  " *
        "($(df_raw.Date[1]) → $(df_raw.Date[end]))")
println("Columns: ", names(df_raw))


# feature engineering
# Keep only rows with valid Close prices
df = dropmissing(df_raw, :Close)
sort!(df, :Date)
 
close_prices = Float64.(df.Close)
n_days       = length(close_prices)
 
# Daily log returns:  r_t = log(P_t / P_{t-1})
log_returns = diff(log.(close_prices))   # length = n_days - 1
 
# Volatility proxy: absolute value of previous day's return
#   |r_{t-1}|  →  how much did the market move yesterday?
volatility_lag = abs.(log_returns[1:end-1])   # yesterday's volatility
return_today   = log_returns[2:end]            # today's return
 
@assert length(volatility_lag) == length(return_today)
n = length(return_today)
println("\nUsable observations after feature engineering: $n")
 
# Standardise both series (zero mean, unit variance)
mu_vol  = mean(volatility_lag);  sig_vol  = std(volatility_lag)
mu_ret  = mean(return_today);    sig_ret  = std(return_today)
 
x = (volatility_lag .- mu_vol) ./ sig_vol    # standardised predictor
y = (return_today   .- mu_ret) ./ sig_ret    # standardised target
 
println("Return stats  — mean: $(round(mu_ret*100, digits=4))%  " *
        "std: $(round(sig_ret*100, digits=4))%")
println("Volatility stats — mean: $(round(mu_vol*100, digits=4))%  " *
        "std: $(round(sig_vol*100, digits=4))%")

# model
#   alpha  ~ Normal(0, 0.1)    baseline daily return
#   beta   ~ Normal(0, 1)      effect of yesterday's volatility
#   sigma  ~ Exponential(1)    observation noise
#
#   return_today[i] ~ Normal(alpha + beta * volatility_lag[i], sigma)
#
# Priors are weakly informative on the standardised scale.
# We deliberately keep the beta prior wide so the data can
# pull it toward zero if that is what it wants.
 
@model function return_model(x, y)
    alpha ~ Normal(0, 0.1)
    beta  ~ Normal(0, 1)
    sigma ~ Exponential(1)
 
    mu = alpha .+ beta .* x
    for i in eachindex(y)
        y[i] ~ Normal(mu[i], sigma)
    end
end


# NUTS Sampling
println("\nRunning NUTS sampler...")
println("  • 1000 posterior samples  |  500 warmup steps")
println("  • target acceptance rate : 0.65\n")
 
model  = return_model(x, y)
chain  = sample(model, NUTS(500, 0.65), 1000; progress=true)
 
println("\n", chain)


# MAP ESTIMATE  (frequentist comparison)
map_est = optimize(model, MAP())
println("\nMAP estimates (point estimates for comparison):")
println("  alpha = ", round(map_est.values[:alpha], digits=5))
println("  beta  = ", round(map_est.values[:beta],  digits=5))
println("  sigma = ", round(map_est.values[:sigma], digits=5))


# etract posterior samples
alpha_samples = Array(chain[:alpha])
beta_samples  = Array(chain[:beta])
sigma_samples = Array(chain[:sigma])
 
alpha_mean = mean(alpha_samples);  alpha_ci = quantile(alpha_samples, [0.025, 0.975])
beta_mean  = mean(beta_samples);   beta_ci  = quantile(beta_samples,  [0.025, 0.975])
sigma_mean = mean(sigma_samples);  sigma_ci = quantile(sigma_samples, [0.025, 0.975])
 
println("\n=== POSTERIOR SUMMARY ===")
println("alpha  mean=$(round(alpha_mean,digits=4))  " *
        "95% CI: [$(round(alpha_ci[1],digits=4)), $(round(alpha_ci[2],digits=4))]")
println("beta   mean=$(round(beta_mean,digits=4))   " *
        "95% CI: [$(round(beta_ci[1],digits=4)), $(round(beta_ci[2],digits=4))]")
println("sigma  mean=$(round(sigma_mean,digits=4))  " *
        "95% CI: [$(round(sigma_ci[1],digits=4)), $(round(sigma_ci[2],digits=4))]")
 
# Does the 95% CI for beta include zero?
beta_includes_zero = beta_ci[1] < 0 < beta_ci[2]
println("\nDoes the 95% credible interval for beta include zero?  ",
        beta_includes_zero ? "YES — volatility is NOT a reliable predictor of direction" :
                             "NO  — volatility has a statistically meaningful effect")



#plots and stuff
mkpath("plots")
 
#  Posterior density plots 
p_post = plot(layout=(1,3), size=(1000, 350), dpi=150,
              plot_title="Posterior Distributions — NIFTY 50 Return Model",
              plot_titlefontsize=11)
 
density!(p_post[1], alpha_samples;
         title="alpha (baseline return)", xlabel="value", ylabel="density",
         fill=(0, 0.25), color=:steelblue, legend=false, linewidth=2)
vline!(p_post[1], [0]; color=:red, linestyle=:dash, linewidth=1.5)
 
density!(p_post[2], beta_samples;
         title="beta (volatility effect)", xlabel="value",
         fill=(0, 0.25), color=:darkorange, legend=false, linewidth=2)
vline!(p_post[2], [0]; color=:red, linestyle=:dash, linewidth=1.5,
       label="zero")
 
density!(p_post[3], sigma_samples;
         title="sigma (noise)", xlabel="value",
         fill=(0, 0.25), color=:seagreen, legend=false, linewidth=2)
 
savefig(p_post, "plots/posterior_distributions.png")
println("\nSaved plots/posterior_distributions.png")
 
#  Trace plots (convergence check) 
p_trace = plot(chain[[:alpha, :beta, :sigma]];
               title=["alpha trace" "beta trace" "sigma trace"],
               legend=false, size=(900, 500), dpi=150)
savefig(p_trace, "plots/trace_plots.png")
println("Saved plots/trace_plots.png")
 
# Credible interval bar chart
coef_names  = ["alpha", "beta", "sigma"]
coef_means  = [alpha_mean, beta_mean, sigma_mean]
coef_lo     = coef_means .- [alpha_ci[1], beta_ci[1], sigma_ci[1]]
coef_hi     = [alpha_ci[2], beta_ci[2], sigma_ci[2]] .- coef_means
 
p_ci = bar(coef_names, coef_means;
           yerror=(coef_lo, coef_hi),
           title="Posterior Means with 95% Credible Intervals",
           xlabel="Parameter", ylabel="Posterior mean (standardised scale)",
           color=[:steelblue :darkorange :seagreen],
           legend=false, size=(650, 450), dpi=150)
hline!(p_ci, [0]; color=:red, linestyle=:dash, linewidth=1.5)
savefig(p_ci, "plots/credible_intervals.png")
println("Saved plots/credible_intervals.png")
 
# Posterior predictive check 
n_draws   = 300
draw_idx  = rand(1:length(alpha_samples), n_draws)
y_pred_mean = mean([alpha_samples[i] .+ beta_samples[i] .* x
                    for i in draw_idx])
 
p_ppc = scatter(y, y_pred_mean;
                xlabel="Actual return (standardised)",
                ylabel="Posterior predicted return (standardised)",
                title="Posterior Predictive Check",
                alpha=0.35, markersize=3, color=:steelblue,
                legend=false, size=(600, 500), dpi=150)
plot!(p_ppc, [-4, 4], [-4, 4]; color=:red, linestyle=:dash,
      linewidth=2, label="perfect prediction")
savefig(p_ppc, "plots/posterior_predictive_check.png")
println("Saved plots/posterior_predictive_check.png")
 
#  Raw return series with volatility overlay 
dates_plot = df.Date[3:end]   # aligned with return_today
p_series = plot(dates_plot, return_today .* 100;
                label="Daily return (%)", color=:steelblue,
                alpha=0.6, linewidth=0.8,
                title="NIFTY 50 — Daily Returns vs Lagged Volatility",
                xlabel="Date", ylabel="Return (%)",
                size=(900, 400), dpi=150)
plot!(p_series, dates_plot, volatility_lag .* 100;
      label="Lagged volatility (%)", color=:darkorange,
      alpha=0.7, linewidth=0.8)
savefig(p_series, "plots/return_series.png")
println("Saved plots/return_series.png")
 

println("\n FINAL INTERPRETATION ")
println("The posterior for beta $(beta_includes_zero ?
    "straddles zero — consistent with the Efficient Market Hypothesis." :
    "is away from zero — lagged volatility carries some predictive signal.")")
println("High sigma posterior ($(round(sigma_mean, digits=3))) confirms")
println("that daily returns are mostly noise, not signal.")
println("\nThis is the correct answer. Markets are hard to predict.")
println("The value of the Bayesian approach here is that we get")
println("a full uncertainty distribution, not false confidence.")  
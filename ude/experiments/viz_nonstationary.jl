# experiments/viz_nonstationary.jl
#
# Diagnostic visualisation for exp_nonstationary results.
# Re-runs simulation + SINDy for one representative config (seed=1) and
# reads summary.csv for the aggregated metrics plot.
#
# Produces 5 PNG files in the most recent (or specified) results directory:
#   1. timeseries_overlay.png  — v(t), u(t): ground truth vs SINDy
#   2. phase_portrait.png      — phase space coloured by regime
#   3. regime_errors.png       — per-regime RMSE across noise levels
#   4. derivative_residual.png — ||ẋ_true - ẋ_sindy|| vs t
#   5. noise_sensitivity.png   — phase portraits across noise levels (seed=1)
#
# Usage:
#   julia --project=. experiments/viz_nonstationary.jl
#   julia --project=. experiments/viz_nonstationary.jl results/exp_nonstationary/2026-03-03_12-00-00

include(joinpath(@__DIR__, "..", "src", "ScientificML.jl"))
using .ScientificML

include(joinpath(@__DIR__, "..", "src", "simulate_nonstationary.jl"))

using Plots
using LinearAlgebra
using DelimitedFiles
using Statistics

gr()   # GR backend: reliable PNG output, no display server needed

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

VIZ_SEED        = 1
VIZ_NOISE_CLEAN = 0.0          # for plots 1–4
VIZ_NOISES      = [0.0, 1e-3, 1e-2, 5e-2]   # for plot 5

saveat    = 0.5
polyorder = 3
nu        = 1.0
λs        = exp10.(-6:0.5:0)
prox      = ScientificML.DataDrivenSparse.SoftThreshold()

x0_true, regime_configs, breakpoints = nonstationary_goodwin_configs()
REGIME_LABELS = ["R1 [0,200)", "R2 [200,290)", "Tr [290,300)", "R3 [300,500]"]
REGIME_COLORS = [:steelblue, :darkorange, :mediumpurple, :forestgreen]
BOUNDARY_TIMES = [200.0, 290.0, 300.0]

# R equilibria for phase portrait reference markers
eq_r1 = [0.05/0.10, 0.05/0.10]   # [v*, u*] = [0.5, 0.5]
eq_r3 = [0.05/0.10, 0.04/0.12]   # [v*, u*] = [0.5, 0.333...]

# ─────────────────────────────────────────────────────────────────────────────
# Locate results directory
# ─────────────────────────────────────────────────────────────────────────────

function find_latest_rundir(expname; root = "results")
    base = joinpath(root, expname)
    isdir(base) || error("No results directory found at: $base")
    entries = filter(isdir, readdir(base; join = true))
    isempty(entries) && error("No run directories found in: $base")
    return last(sort(entries))   # lexicographic timestamp sort
end

script_dir = @__DIR__
results_root = joinpath(script_dir, "..", "results")

if length(ARGS) >= 1
    outdir = ARGS[1]
else
    outdir = find_latest_rundir("exp_nonstationary"; root = results_root)
end

isdir(outdir) || error("Results directory not found: $outdir")
println("Visualising results from: ", outdir)

# ─────────────────────────────────────────────────────────────────────────────
# Simulation + SINDy for the representative run (seed=1, noise=0)
# ─────────────────────────────────────────────────────────────────────────────

println("Re-running simulation (seed=$VIZ_SEED, noise=$VIZ_NOISE_CLEAN)…")
rng_clean = ScientificML.StableRNGs.StableRNG(VIZ_SEED)

t_full, X_clean = simulate_nonstationary(x0_true, regime_configs; saveat = saveat)
masks = get_regime_masks(t_full, breakpoints)

Xn_clean = (VIZ_NOISE_CLEAN == 0.0) ? copy(X_clean) :
            add_relative_noise(X_clean, rng_clean; noise_magnitude = VIZ_NOISE_CLEAN)

println("Fitting SINDy on clean trajectory…")
res_clean, system_clean, _ = fit_sindy_sr3(
    t_full, Xn_clean;
    polyorder = polyorder, λs = λs, nu = nu, proximal = prox, rng = rng_clean
)
Xhat_clean = predict_sindy(res_clean, X_clean[:, 1], t_full)

# ─────────────────────────────────────────────────────────────────────────────
# Shared plotting utilities
# ─────────────────────────────────────────────────────────────────────────────

"Add vertical dashed lines at regime boundaries."
function add_boundaries!(p)
    for tb in BOUNDARY_TIMES
        vline!(p, [tb]; linestyle = :dash, color = :gray60, linewidth = 1.2, label = "")
    end
end

"Shade regime background bands on a plot with known y limits."
function add_regime_bands!(p, y_lo, y_hi)
    for i in 1:4
        lo = breakpoints[i]
        hi = breakpoints[i + 1]
        xs = [lo, hi, hi, lo, lo]
        ys = [y_lo, y_lo, y_hi, y_hi, y_lo]
        plot!(p, xs, ys;
              seriestype = :shape, fillalpha = 0.08,
              color = REGIME_COLORS[i], linewidth = 0, label = "")
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# Plot 1 — Time series overlay
# ─────────────────────────────────────────────────────────────────────────────

println("Plot 1: time series overlay…")

function plot_timeseries(t, X_true, X_sindy, masks, outpath)
    v_true = X_true[1, :];  v_hat = X_sindy[1, :]
    u_true = X_true[2, :];  u_hat = X_sindy[2, :]

    pv = plot(; xlabel = "t", ylabel = "v  (employment rate)",
                title = "v(t): ground truth vs SINDy",
                legend = :topright, size = (700, 280), dpi = 150,
                leftmargin = 5Plots.mm, bottommargin = 5Plots.mm)
    add_regime_bands!(pv, minimum(v_true) - 0.02, maximum(v_true) + 0.02)
    # Colour ground truth by regime
    for i in 1:4
        idx = findall(masks[i])
        isempty(idx) && continue
        plot!(pv, t[idx], v_true[idx];
              color = REGIME_COLORS[i], linewidth = 1.8,
              label = (i == 1 ? "True" : ""))
    end
    plot!(pv, t, v_hat; color = :black, linewidth = 1.5,
          linestyle = :dash, label = "SINDy")
    add_boundaries!(pv)
    ylims!(pv, minimum(v_true) - 0.02, maximum(v_true) + 0.02)

    pu = plot(; xlabel = "t", ylabel = "u  (wage share)",
                title = "u(t): ground truth vs SINDy",
                legend = :topright, size = (700, 280), dpi = 150,
                leftmargin = 5Plots.mm, bottommargin = 5Plots.mm)
    add_regime_bands!(pu, minimum(u_true) - 0.02, maximum(u_true) + 0.02)
    for i in 1:4
        idx = findall(masks[i])
        isempty(idx) && continue
        plot!(pu, t[idx], u_true[idx];
              color = REGIME_COLORS[i], linewidth = 1.8,
              label = (i == 1 ? "True" : ""))
    end
    plot!(pu, t, u_hat; color = :black, linewidth = 1.5,
          linestyle = :dash, label = "SINDy")
    add_boundaries!(pu)
    ylims!(pu, minimum(u_true) - 0.02, maximum(u_true) + 0.02)

    p = plot(pv, pu; layout = (2, 1), size = (700, 560), dpi = 150)
    savefig(p, outpath)
    println("  → ", outpath)
end

plot_timeseries(t_full, X_clean, Xhat_clean, masks,
                joinpath(outdir, "timeseries_overlay.png"))

# ─────────────────────────────────────────────────────────────────────────────
# Plot 2 — Phase portrait
# ─────────────────────────────────────────────────────────────────────────────

println("Plot 2: phase portrait…")

function plot_phase(X_true, X_sindy, masks, eq_r1, eq_r3, outpath)
    p = plot(; xlabel = "v  (employment rate)", ylabel = "u  (wage share)",
               title = "Phase portrait",
               legend = :outertopright, size = (620, 500), dpi = 150,
               leftmargin = 5Plots.mm, bottommargin = 5Plots.mm)

    # Ground truth coloured by regime
    for i in 1:4
        idx = findall(masks[i])
        isempty(idx) && continue
        plot!(p, X_true[1, idx], X_true[2, idx];
              color = REGIME_COLORS[i], linewidth = 1.6,
              label = REGIME_LABELS[i])
    end

    # SINDy rollout
    plot!(p, X_sindy[1, :], X_sindy[2, :];
          color = :black, linewidth = 1.4, linestyle = :dash, label = "SINDy")

    # Equilibrium markers
    scatter!(p, [eq_r1[1]], [eq_r1[2]]; marker = :star5, markersize = 10,
             color = REGIME_COLORS[1], label = "Eq R1")
    scatter!(p, [eq_r3[1]], [eq_r3[2]]; marker = :star5, markersize = 10,
             color = REGIME_COLORS[4], label = "Eq R3")

    # Start marker
    scatter!(p, [X_true[1, 1]], [X_true[2, 1]]; marker = :circle, markersize = 6,
             color = :black, label = "x₀")

    savefig(p, outpath)
    println("  → ", outpath)
end

plot_phase(X_clean, Xhat_clean, masks, eq_r1, eq_r3,
           joinpath(outdir, "phase_portrait.png"))

# ─────────────────────────────────────────────────────────────────────────────
# Plot 3 — Per-regime errors from summary.csv
# ─────────────────────────────────────────────────────────────────────────────

println("Plot 3: per-regime errors from summary.csv…")

function plot_regime_errors(summary_path, outpath)
    isfile(summary_path) || (println("  summary.csv not found; skipping"); return)

    data, header = readdlm(summary_path, ',', header = true)
    header = vec(header)

    col(name) = findfirst(==(name), header)

    noise_col = col("noise")
    r_rmse_cols = [col("r1_RMSE"), col("r2_RMSE"), col("tr_RMSE"), col("r3_RMSE")]

    all_noises = sort(unique(parse.(Float64, string.(data[:, noise_col]))))

    # Mean RMSE per (regime × noise), averaged across seeds
    n_noises  = length(all_noises)
    n_regimes = 4
    mean_rmse = fill(NaN, n_noises, n_regimes)

    for (ni, nv) in enumerate(all_noises)
        rows = [parse(Float64, string(data[r, noise_col])) ≈ nv
                for r in axes(data, 1)]
        for ri in 1:n_regimes
            vals = [parse(Float64, string(data[r, r_rmse_cols[ri]]))
                    for r in findall(rows)
                    if !isnan(parse(Float64, string(data[r, r_rmse_cols[ri]])))]
            isempty(vals) || (mean_rmse[ni, ri] = mean(vals))
        end
    end

    noise_labels = ["σ=" * string(n) for n in all_noises]

    # Manual grouped bar using base Plots.jl (no StatsPlots dependency).
    # Each group = one noise level; each bar within a group = one regime.
    bar_w    = 0.18
    group_w  = n_regimes * bar_w + 0.1
    centers  = [1 + (i - 1) * group_w for i in 1:n_noises]

    p = plot(; xlabel = "Noise level",
               ylabel = "Mean RMSE  (across seeds)",
               title  = "Per-regime SINDy error vs noise",
               size   = (700, 420), dpi = 150,
               leftmargin = 6Plots.mm, bottommargin = 6Plots.mm,
               legend = :topleft,
               xticks = (centers, noise_labels))

    for ri in 1:n_regimes
        offset = (ri - (n_regimes + 1) / 2) * bar_w
        xs = centers .+ offset
        ys = replace(mean_rmse[:, ri], NaN => 0.0)
        bar!(p, xs, ys;
             bar_width = bar_w * 0.9,
             color     = REGIME_COLORS[ri],
             label     = REGIME_LABELS[ri])
    end

    savefig(p, outpath)
    println("  → ", outpath)
end

plot_regime_errors(joinpath(outdir, "summary.csv"),
                  joinpath(outdir, "regime_errors.png"))

# ─────────────────────────────────────────────────────────────────────────────
# Plot 4 — Derivative residual over time
# ─────────────────────────────────────────────────────────────────────────────

println("Plot 4: derivative residual…")

"""
    eval_sindy_at_states(t, X, res) -> dX

Evaluate the discovered SINDy ODE at each column of X (the true states).
Returns dX shaped (n_states × n_time).  Falls back to central finite
differences on Xhat if the symbolic evaluation path is unavailable.
"""
function eval_sindy_at_states(t, X, res, Xhat_fallback)
    n, T = size(X)
    try
        # Extract the compiled ODE function from the DataDrivenDiffEq result.
        # ODEProblem(system, x0, tspan, pvec) creates a standard ODE problem
        # whose .f can be called as f(du, u, p, t).
        sindy_sys  = ScientificML.DataDrivenDiffEq.get_basis(res)
        sindy_pmap = Dict(ScientificML.ModelingToolkit.get_parameter_map(sindy_sys))
        sindy_pvec = ScientificML.ModelingToolkit.varmap_to_vars(
                         sindy_pmap, ScientificML.ModelingToolkit.parameters(sindy_sys))

        prob_tmp = ODEProblem(sindy_sys, copy(X[:, 1]), (t[1], t[end]), sindy_pvec)
        f_sindy  = prob_tmp.f
        p_sindy  = prob_tmp.p      # may differ from sindy_pvec in MTK ordering

        dX = zeros(n, T)
        du = zeros(n)
        for j in eachindex(t)
            f_sindy(du, X[:, j], p_sindy, t[j])
            dX[:, j] .= du
        end
        println("    (evaluated SINDy at true states)")
        return dX

    catch e
        # Fallback: central finite differences on the SINDy rollout.
        println("    (SINDy-at-states failed: $(typeof(e)); using FD on rollout)")
        dX = zeros(n, T)
        for j in 2:T-1
            dt_j = t[j+1] - t[j-1]
            dX[:, j] = (Xhat_fallback[:, j+1] - Xhat_fallback[:, j-1]) / dt_j
        end
        dX[:, 1]   = (Xhat_fallback[:, 2]   - Xhat_fallback[:, 1])   / (t[2]   - t[1])
        dX[:, end] = (Xhat_fallback[:, end] - Xhat_fallback[:, end-1]) / (t[end] - t[end-1])
        return dX
    end
end

function plot_derivative_residual(t, X_true, X_sindy_rollout, res_sindy,
                                   regime_configs, breakpoints, masks, outpath)

    # True derivatives: exact ODE evaluation at the true states.
    dX_true = evaluate_true_derivatives(t, X_true, regime_configs, breakpoints)

    # SINDy derivatives: evaluate the discovered model at the TRUE states.
    # This is a pure model-fit residual, independent of rollout error.
    dX_sindy = eval_sindy_at_states(t, X_true, res_sindy, X_sindy_rollout)

    # Residual norm at each time point
    resid = [norm(dX_true[:, j] - dX_sindy[:, j]) for j in eachindex(t)]

    p = plot(; xlabel = "t",
               ylabel = "‖ẋ_true − ẋ_SINDy‖",
               title  = "Derivative residual  (SINDy evaluated at true states, noise=0)",
               legend = :topright, size = (700, 320), dpi = 150,
               leftmargin = 6Plots.mm, bottommargin = 6Plots.mm)

    y_lo, y_hi = 0.0, maximum(resid) * 1.1
    add_regime_bands!(p, y_lo, y_hi)
    plot!(p, t, resid; color = :firebrick, linewidth = 1.6, label = "‖residual‖")
    add_boundaries!(p)
    ylims!(p, y_lo, y_hi)

    savefig(p, outpath)
    println("  → ", outpath)
end

plot_derivative_residual(t_full, X_clean, Xhat_clean, res_clean,
                          regime_configs, breakpoints,
                          masks, joinpath(outdir, "derivative_residual.png"))

# ─────────────────────────────────────────────────────────────────────────────
# Plot 5 — Noise sensitivity panel
# ─────────────────────────────────────────────────────────────────────────────

println("Plot 5: noise sensitivity panel (seed=$VIZ_SEED)…")

function plot_noise_sensitivity(t, X_clean, masks, eq_r1, eq_r3, noise_levels,
                                 regime_configs, breakpoints, outpath)

    subplots = Plots.Plot[]

    for nv in noise_levels
        rng = ScientificML.StableRNGs.StableRNG(VIZ_SEED + 100)   # separate RNG per panel
        Xn  = (nv == 0.0) ? copy(X_clean) :
               add_relative_noise(X_clean, rng; noise_magnitude = nv)

        r2_str = "—"
        Xhat   = nothing

        try
            res, _, _ = fit_sindy_sr3(
                t, Xn;
                polyorder = polyorder, λs = λs, nu = nu, proximal = prox, rng = rng
            )
            Xhat = predict_sindy(res, X_clean[:, 1], t)
            gm   = trajectory_metrics(X_clean, Xhat)
            r2_str = "R²=$(round(gm.r2, sigdigits=3))"
        catch
            r2_str = "failed"
        end

        noise_label = nv == 0.0 ? "σ=0" : "σ=$(nv)"
        sp = plot(; xlabel = "v", ylabel = "u",
                    title  = "$noise_label  $r2_str",
                    legend = false, size = (300, 280), dpi = 100,
                    titlefontsize = 9,
                    leftmargin = 3Plots.mm, bottommargin = 3Plots.mm)

        # Ground truth coloured by regime
        for i in 1:4
            idx = findall(masks[i])
            isempty(idx) && continue
            plot!(sp, X_clean[1, idx], X_clean[2, idx];
                  color = REGIME_COLORS[i], linewidth = 1.2, label = "")
        end

        # SINDy rollout
        if Xhat !== nothing
            plot!(sp, Xhat[1, :], Xhat[2, :];
                  color = :black, linewidth = 1.2, linestyle = :dash, label = "")
        end

        # Equilibrium markers
        scatter!(sp, [eq_r1[1]], [eq_r1[2]]; marker = :star5, markersize = 6,
                 color = REGIME_COLORS[1], label = "")
        scatter!(sp, [eq_r3[1]], [eq_r3[2]]; marker = :star5, markersize = 6,
                 color = REGIME_COLORS[4], label = "")

        push!(subplots, sp)
    end

    p = plot(subplots...;
             layout = (2, 2),
             size   = (700, 560),
             dpi    = 150,
             plot_title = "SINDy noise sensitivity (seed=$VIZ_SEED)")
    savefig(p, outpath)
    println("  → ", outpath)
end

plot_noise_sensitivity(t_full, X_clean, masks, eq_r1, eq_r3, VIZ_NOISES,
                        regime_configs, breakpoints,
                        joinpath(outdir, "noise_sensitivity.png"))

println("\nAll plots saved to: ", outdir)

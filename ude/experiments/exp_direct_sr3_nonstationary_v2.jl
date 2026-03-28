#!/usr/bin/env julia
#= ──────────────────────────────────────────────────────────────────────
   exp_direct_sr3_nonstationary_v2.jl
   Direct PySINDy SR3 (L0) on nonstationary Goodwin — matching Colab settings.
   PySINDy handles differentiation via SmoothedFiniteDifference.
   ────────────────────────────────────────────────────────────────────── =#

include(joinpath(@__DIR__, "..", "src", "ScientificML.jl"))
using .ScientificML
include(joinpath(@__DIR__, "..", "src", "simulate_nonstationary.jl"))
using PyCall, StableRNGs, Statistics, JLD2

# ── ensure pysindy ───────────────────────────────────────────────────
try; pyimport("pysindy"); catch
    pip = PyCall.pyprogramname
    try; run(`$pip -m pip install numpy pysindy --quiet`)
    catch; run(`$pip -m pip install numpy pysindy --quiet --break-system-packages`); end
end

# ── generate clean trajectory + true derivatives ─────────────────────
x0, regime_configs, breakpoints = nonstationary_goodwin_configs()
t_clean, X_clean = simulate_nonstationary(x0, regime_configs; saveat=0.5)
t_clean = collect(t_clean)
dX_true = evaluate_true_derivatives(t_clean, X_clean, regime_configs, breakpoints)
masks   = get_regime_masks(t_clean, breakpoints)
N       = length(t_clean)

@info "Trajectory: $(size(X_clean)), N=$N"

# ── PySINDy imports ──────────────────────────────────────────────────
const np = pyimport("numpy")
const ps = pyimport("pysindy")
const SR3_py = pyimport("pysindy.optimizers").SR3
const SFD_py = pyimport("pysindy.differentiation").SmoothedFiniteDifference

# ── Build clean-state library once for evaluation ────────────────────
# PySINDy PolynomialLibrary(degree=2, include_bias=False) produces:
#   [x, y, x², xy, y²] → [v, u, v², vu, u²]
v_clean = X_clean[1, :]
u_clean = X_clean[2, :]
Theta_clean = hcat(v_clean, u_clean, v_clean.^2, v_clean.*u_clean, u_clean.^2)

# ── Direct SR3 fitting function (Colab settings) ────────────────────
function fit_direct_sr3(X_data::Matrix{Float64}, t_vec::Vector{Float64})
    X_py = np.array(collect(X_data'))   # N×2
    t_py = np.array(t_vec)

    lib  = ps.PolynomialLibrary(degree=2, include_interaction=true, include_bias=false)
    diff = SFD_py(smoother_kws=Dict("window_length" => 9))
    opt  = SR3_py(
        reg_weight_lam    = 1e-3,
        regularizer       = "L0",
        relax_coeff_nu    = 10.0,
        tol               = 1e-10,
        max_iter          = 10000,
        normalize_columns = true,
    )
    model = ps.SINDy(feature_library=lib, optimizer=opt, differentiation_method=diff)

    # Fit — try with feature_names, fall back without
    try
        model.fit(X_py, t=t_py, feature_names=["v", "u"])
    catch
        model.fit(X_py, t=t_py)
    end

    # Extract results
    coef_matrix = convert(Matrix{Float64}, model.coefficients())  # 2×n_feat
    feat_names  = String[string(f) for f in model.get_feature_names()]
    n_feat      = length(feat_names)

    # Predicted derivatives at training states
    Xdot_pred = convert(Matrix{Float64}, model.predict(X_py))  # N×2

    # Active terms (|coeff| > 0.001)
    active = Dict{String, Vector{Tuple{String, Float64}}}()
    for (k, label) in enumerate(["f1", "f2"])
        coeffs = coef_matrix[k, :]
        active[label] = [(feat_names[i], coeffs[i]) for i in 1:n_feat if abs(coeffs[i]) > 0.001]
    end

    # Print discovered equations
    println("  Discovered equations:")
    for (k, label) in enumerate(["f1 (dv/dt)", "f2 (du/dt)"])
        terms = join(["$(c >= 0 ? "+" : "")$(round(c, sigdigits=4))·$n"
                       for (n, c) in active[k == 1 ? "f1" : "f2"]], " ")
        println("    $label = $terms")
    end

    return (; coef_matrix, feat_names, active, Xdot_pred)
end

# ── Per-regime RMSE helper ───────────────────────────────────────────
function regime_rmses(dX_true, dX_pred, masks)
    [sqrt(mean((dX_true[:, m] .- dX_pred[:, m]) .^ 2)) for m in masks]
end

# ── Experiment loop ──────────────────────────────────────────────────
noise_levels = [0.0, 1e-3, 1e-2, 5e-2]
seeds        = 1:5
regime_names = ["R1", "R2", "Trans", "R3"]

results_all  = Dict{Tuple{Float64,Int}, Dict{String,Any}}()
summary_rows = []

for noise in noise_levels, seed in seeds
    @info "noise=$noise, seed=$seed"

    # 1. Add noise
    X_noisy = if noise == 0.0
        copy(X_clean)
    else
        add_relative_noise(X_clean, StableRNG(seed); noise_magnitude=noise)
    end

    # 2. Fit direct SR3
    result = fit_direct_sr3(X_noisy, t_clean)

    # 3. Evaluate at clean states
    c_f1 = result.coef_matrix[1, :]
    c_f2 = result.coef_matrix[2, :]
    dX_discovered = zeros(2, N)
    dX_discovered[1, :] = Theta_clean * c_f1
    dX_discovered[2, :] = Theta_clean * c_f2

    # 4. Metrics
    resid  = dX_true .- dX_discovered
    rmse   = sqrt(mean(resid .^ 2))
    ss_res = sum(resid .^ 2)
    ss_tot = sum((dX_true .- mean(dX_true, dims=2)) .^ 2)
    r2     = 1.0 - ss_res / ss_tot

    # Per-regime RMSE
    reg_rmse = regime_rmses(dX_true, dX_discovered, masks)

    # Format equations
    f1_str = join(["$(n):$(round(c,sigdigits=4))" for (n,c) in result.active["f1"]], "; ")
    f2_str = join(["$(n):$(round(c,sigdigits=4))" for (n,c) in result.active["f2"]], "; ")

    println("  RMSE=$(round(rmse, sigdigits=4))  R²=$(round(r2, sigdigits=4))")
    for (i, rn) in enumerate(regime_names)
        println("    $rn RMSE: $(round(reg_rmse[i], sigdigits=4))")
    end

    push!(summary_rows, (seed=seed, noise=noise, rmse=rmse, r2=r2,
                          r1_rmse=reg_rmse[1], r2_rmse=reg_rmse[2],
                          tr_rmse=reg_rmse[3], r3_rmse=reg_rmse[4],
                          f1_active=f1_str, f2_active=f2_str,
                          n_f1=length(result.active["f1"]),
                          n_f2=length(result.active["f2"])))

    results_all[(noise, seed)] = Dict(
        "dX_discovered" => dX_discovered,
        "coeffs_f1" => c_f1, "coeffs_f2" => c_f2,
        "active_f1" => result.active["f1"],
        "active_f2" => result.active["f2"],
        "feat_names" => result.feat_names,
        "rmse" => rmse, "r2" => r2,
        "regime_rmse" => reg_rmse,
    )
end

# ── Save results ─────────────────────────────────────────────────────
outdir = "results/direct_sr3_nonstationary"
mkpath(outdir)

# CSV
csv_path = joinpath(outdir, "summary.csv")
open(csv_path, "w") do io
    println(io, "seed,noise,rmse,r2,r1_rmse,r2_rmse,tr_rmse,r3_rmse,n_f1,n_f2,f1_active,f2_active")
    for r in summary_rows
        println(io, "$(r.seed),$(r.noise),$(r.rmse),$(r.r2),$(r.r1_rmse),$(r.r2_rmse),$(r.tr_rmse),$(r.r3_rmse),$(r.n_f1),$(r.n_f2),\"$(r.f1_active)\",\"$(r.f2_active)\"")
    end
end
@info "CSV: $csv_path"

# JLD2
jld2_path = joinpath(outdir, "results.jld2")
JLD2.jldsave(jld2_path;
    results_all  = results_all,
    summary_rows = summary_rows,
    t = t_clean, X_clean = X_clean, dX_true = dX_true,
    breakpoints = breakpoints, masks = masks,
    noise_levels = noise_levels, seeds = collect(seeds))
@info "JLD2: $jld2_path"

# ── Summary table ────────────────────────────────────────────────────
println("\n" * "="^80)
println("  Direct SR3 (Colab settings) — Aggregated summary")
println("="^80)
for noise in noise_levels
    rows  = filter(r -> r.noise == noise, summary_rows)
    rmses = [r.rmse for r in rows]
    r2s   = [r.r2 for r in rows]
    println("  noise=$(rpad(string(noise), 6))  " *
            "RMSE = $(round(mean(rmses), sigdigits=4)) ± $(round(std(rmses), sigdigits=2)),  " *
            "R² = $(round(mean(r2s), sigdigits=4)) ± $(round(std(r2s), sigdigits=2))")
end
println("="^80)

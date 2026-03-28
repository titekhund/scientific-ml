#!/usr/bin/env julia
#=
  Real-data SINDy experiment: U.S. macroeconomic Goodwin dynamics
  Part 1 — Data loading, TVDiff derivative estimation, and SR3 SINDy
  for both Classical (v, u) and Structuralist (tcu, u) specifications.
=#

## ── Setup ────────────────────────────────────────────────────────────────────

include(joinpath(@__DIR__, "..", "src", "ScientificML.jl"))
using .ScientificML

using Pkg
for pkg in ["Downloads", "XLSX", "NoiseRobustDifferentiation"]
    haskey(Pkg.project().dependencies, pkg) || Pkg.add(pkg)
end

using Downloads, XLSX
using NoiseRobustDifferentiation
using Statistics, LinearAlgebra, Printf
using Plots; gr()

import .ScientificML: fit_sindy_sr3, predict_sindy, sindy_poly_basis,
                      eval_sindy_at_states, trajectory_metrics,
                      write_csv, append_csv_row
using .ScientificML: DataDrivenSparse, DataDrivenDiffEq, ModelingToolkit

const OUTDIR = joinpath(@__DIR__, "..", "results", "realdata")
mkpath(OUTDIR)

## ── Data Loading ─────────────────────────────────────────────────────────────

println("Downloading data...")
datafile = joinpath(OUTDIR, "data.xlsx")
if !isfile(datafile)
    Downloads.download(
        "https://raw.githubusercontent.com/titekhund/scientific_ml_tato/main/data.xlsx",
        datafile)
end

xf    = XLSX.readxlsx(datafile)
sheet = xf["Sheet1"][:]  # Matrix{Any}, row 1 = headers

to_float(x) = x isa Number ? Float64(x) : NaN

unrate_raw     = [to_float(sheet[i, 2]) for i in 2:size(sheet, 1)]
tcu_raw        = [to_float(sheet[i, 3]) for i in 2:size(sheet, 1)]
wage_share_raw = [to_float(sheet[i, 4]) for i in 2:size(sheet, 1)]
println("Loaded $(length(unrate_raw)) raw rows from Sheet1")

# Diagnostic: print last 5 rows to check for trailing empties
println("Last 5 rows of raw data:")
for i in max(1, length(unrate_raw)-4):length(unrate_raw)
    @printf("  row %d: unrate=%.4f  tcu=%s  wage_share=%.4f\n",
            i, unrate_raw[i],
            isnan(tcu_raw[i]) ? "NaN" : @sprintf("%.2f", tcu_raw[i]),
            wage_share_raw[i])
end

# Trim rows where core columns (unrate, wage_share) are NaN
valid_core = .!isnan.(unrate_raw) .& .!isnan.(wage_share_raw)
unrate     = unrate_raw[valid_core]
tcu        = tcu_raw[valid_core]
wage_share = wage_share_raw[valid_core]
println("After trimming NaN rows: $(length(unrate)) valid rows (dropped $(sum(.!valid_core)))")

# ── Classical specification: v = 1 - unrate/100,  u = wage_share/100
# Full sample 1948:Q1–2025:Q3
v_class = 1.0 .- unrate ./ 100.0
u_class = wage_share ./ 100.0
N_class = length(v_class)
t_class = Float64.(0:N_class-1)
println("Classical spec: N = $N_class (expect 311)")
println("  v range: [$(minimum(v_class)), $(maximum(v_class))], NaN=$(sum(isnan.(v_class)))")
println("  u range: [$(minimum(u_class)), $(maximum(u_class))], NaN=$(sum(isnan.(u_class)))")

# ── Structuralist specification: tcu (raw), u = wage_share/100
# Drop rows with missing tcu (NaN) → starts ~1967:Q1
mask_tcu   = .!isnan.(tcu)
tcu_struct = tcu[mask_tcu] ./ 100.0
u_struct   = wage_share[mask_tcu] ./ 100.0
N_struct   = length(tcu_struct)
t_struct   = Float64.(0:N_struct-1)
println("Structuralist spec: N = $N_struct (expect 235)")

## ── TVDiff Derivative Estimation ─────────────────────────────────────────────

alphas = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
default_alpha = 0.01
tvdiff_iter   = 1000

function tvdiff_sweep(state1, state2, label1, label2, alphas, t, spec_name)
    N = length(state1)
    p = plot(layout=(2, length(alphas)), size=(300*length(alphas), 500),
             title="TVDiff alpha sweep — $spec_name")
    for (i, α) in enumerate(alphas)
        d1 = tvdiff(state1, tvdiff_iter, α; scale="large", dx=1.0)
        d2 = tvdiff(state2, tvdiff_iter, α; scale="large", dx=1.0)
        # Trim to match N if needed
        d1 = d1[1:min(N, length(d1))]
        d2 = d2[1:min(N, length(d2))]
        plot!(p, t[1:length(d1)], d1; subplot=i, label="α=$α",
              title="d($label1)/dt, α=$α", titlefontsize=8, lw=1)
        plot!(p, t[1:length(d2)], d2; subplot=length(alphas)+i, label="α=$α",
              title="d($label2)/dt, α=$α", titlefontsize=8, lw=1)
    end
    savefig(p, joinpath(OUTDIR, "tvdiff_alpha_sweep_$(spec_name).png"))
    println("Saved TVDiff sweep plot: tvdiff_alpha_sweep_$(spec_name).png")
    return p
end

function compute_tvdiff(state1, state2, α, iter)
    N = length(state1)
    d1 = tvdiff(state1, iter, α; scale="large", dx=1.0)
    d2 = tvdiff(state2, iter, α; scale="large", dx=1.0)
    d1 = d1[1:min(N, length(d1))]
    d2 = d2[1:min(N, length(d2))]
    # Ensure both match length
    Nmin = min(length(d1), length(d2), N)
    return d1[1:Nmin], d2[1:Nmin], Nmin
end

println("\n--- TVDiff alpha sweep (Classical) ---")
tvdiff_sweep(v_class, u_class, "v", "u", alphas, t_class, "classical")

println("\n--- TVDiff alpha sweep (Structuralist) ---")
tvdiff_sweep(tcu_struct, u_struct, "tcu", "u", alphas, t_struct, "structuralist")

# Compute derivatives at default alpha
println("\nComputing TVDiff derivatives at α = $default_alpha ...")
dv_class, du_class, Nc = compute_tvdiff(v_class, u_class, default_alpha, tvdiff_iter)
println("  Classical TVDiff results (N=$Nc):")
println("  dv: min=$(minimum(dv_class)), max=$(maximum(dv_class)), NaN count=$(sum(isnan.(dv_class)))")
println("  du: min=$(minimum(du_class)), max=$(maximum(du_class)), NaN count=$(sum(isnan.(du_class)))")

dtcu_struct, du_struct_d, Ns = compute_tvdiff(tcu_struct, u_struct, default_alpha, tvdiff_iter)
println("  Structuralist TVDiff results (N=$Ns):")
println("  dtcu: min=$(minimum(dtcu_struct)), max=$(maximum(dtcu_struct)), NaN count=$(sum(isnan.(dtcu_struct)))")
println("  du:   min=$(minimum(du_struct_d)), max=$(maximum(du_struct_d)), NaN count=$(sum(isnan.(du_struct_d)))")

# Trim state arrays to match derivative lengths
v_c  = v_class[1:Nc];   u_c  = u_class[1:Nc];   t_c  = t_class[1:Nc]
tcu_s = tcu_struct[1:Ns]; u_s = u_struct[1:Ns];   t_s  = t_struct[1:Ns]

## ── Part 1: SINDy ───────────────────────────────────────────────────────────

polyorder = 2

## ── Exhaustive OLS (primary method — pure linear algebra, cannot fail) ───────

# Build polynomial feature library for 2-state system up to order 3
# For states (a, b): {1, a, b, a², ab, b², a³, a²b, ab², b³}
function build_poly_library(s1, s2; order=3, varnames=("s1", "s2"))
    N = length(s1)
    names  = String[]
    cols   = Vector{Vector{Float64}}()
    for p in 0:order, q in 0:(order - p)
        push!(names, _mono_name(p, q, varnames))
        push!(cols, (s1 .^ p) .* (s2 .^ q))
    end
    return hcat(cols...), names   # N × n_features
end

function _mono_name(p, q, varnames)
    p == 0 && q == 0 && return "1"
    parts = String[]
    p == 1 && push!(parts, varnames[1])
    p >  1 && push!(parts, "$(varnames[1])^$p")
    q == 1 && push!(parts, varnames[2])
    q >  1 && push!(parts, "$(varnames[2])^$q")
    return join(parts, "*")
end

function ols_fit(Phi, y, idx)
    Phi_sub = Phi[:, idx]
    coeffs  = Phi_sub \ y
    mse     = mean((y .- Phi_sub * coeffs) .^ 2)
    return coeffs, mse
end

function exhaustive_sindy_search(Phi, y, feat_names; max_terms=2)
    nf = size(Phi, 2)
    results = []
    # single-term models
    for i in 1:nf
        c, m = ols_fit(Phi, y, [i])
        push!(results, (idx=[i], coeffs=c, mse=m))
    end
    # two-term models
    if max_terms >= 2
        for i in 1:nf, j in (i+1):nf
            c, m = ols_fit(Phi, y, [i, j])
            push!(results, (idx=[i, j], coeffs=c, mse=m))
        end
    end
    sort!(results, by=r -> r.mse)
    return results
end

function format_terms(r, feat_names)
    join([@sprintf("%.6f*%s", r.coeffs[k], feat_names[r.idx[k]]) for k in 1:length(r.idx)], " + ")
end

function run_exhaustive_ols(X, dX, t, spec_name, state_labels)
    println("\n=== Exhaustive OLS (primary) — $spec_name ===")
    n = size(X, 1)
    N = size(X, 2)
    Phi, feat_names = build_poly_library(X[1, :], X[2, :];
                                         order=polyorder, varnames=tuple(state_labels...))

    eq_file = joinpath(OUTDIR, "sindy_$(spec_name)_equations.txt")
    csv_path = joinpath(OUTDIR, "sindy_$(spec_name)_deriv_residuals.csv")

    open(eq_file, "w") do io
        println(io, "Exhaustive OLS SINDy — $spec_name")
        println(io, "polyorder=$polyorder, α_tvdiff=$default_alpha")
        println(io, "Features: ", join(feat_names, ", "))
        println(io, "")
    end

    header = reshape(["equation", "rank", "terms", "mse", "rmse"], 1, :)
    write_csv(csv_path, header)

    best_pred = zeros(n, N)   # store best model predictions for plotting

    for eq_i in 1:n
        y = dX[eq_i, :]
        results = exhaustive_sindy_search(Phi, y, feat_names; max_terms=2)

        println("  d($(state_labels[eq_i]))/dt — top 5:")
        open(eq_file, "a") do io
            println(io, "--- d($(state_labels[eq_i]))/dt ---")
            for (rank, r) in enumerate(results)
                terms = format_terms(r, feat_names)
                rmse  = sqrt(r.mse)
                rank <= 5 && @printf("    #%d MSE=%.6e RMSE=%.6e : %s\n", rank, r.mse, rmse, terms)
                println(io, @sprintf("  #%d MSE=%.6e : %s", rank, r.mse, terms))
                append_csv_row(csv_path,
                    reshape(["d$(state_labels[eq_i])/dt", rank,
                             replace(terms, ',' => ';'), r.mse, rmse], 1, :))
            end
            println(io, "")
        end

        # Best model prediction for plotting
        best = results[1]
        best_pred[eq_i, :] = Phi[:, best.idx] * best.coeffs
    end

    # Plot derivative residuals
    p = plot(layout=(n, 1), size=(900, 300*n))
    for i in 1:n
        residuals = dX[i, :] .- best_pred[i, :]
        rmse = sqrt(mean(residuals .^ 2))
        plot!(p, t, dX[i, :]; subplot=i, label="TVDiff", lw=1.5)
        plot!(p, t, best_pred[i, :]; subplot=i, label="best OLS", lw=1.5, ls=:dash)
        plot!(p, t, residuals; subplot=i, label="residual", lw=0.8, alpha=0.5)
        title!(p, "d($(state_labels[i]))/dt — best OLS RMSE=$(round(rmse, digits=6))"; subplot=i)
    end
    savefig(p, joinpath(OUTDIR, "sindy_deriv_residuals_$(spec_name).png"))
    println("  Saved: sindy_deriv_residuals_$(spec_name).png")
end

## ── Manual STLSQ threshold sweep (pure Julia, no DataDrivenDiffEq) ──────────

function stlsq_fit(Phi, y, threshold; max_iter=50)
    # Sequential thresholded least squares
    nf = size(Phi, 2)
    coeffs = Phi \ y  # initial OLS
    for _ in 1:max_iter
        mask = abs.(coeffs) .>= threshold
        any(mask) || break  # all zeroed out
        old_coeffs = copy(coeffs)
        coeffs .= 0.0
        coeffs[mask] = Phi[:, mask] \ y
        coeffs == old_coeffs && break
    end
    pred = Phi * coeffs
    mse  = mean((y .- pred) .^ 2)
    return coeffs, mse
end

function run_stlsq_sweep(X, dX, t, spec_name, state_labels)
    println("\n=== STLSQ Threshold Sweep — $spec_name ===")
    n = size(X, 1)
    Phi, feat_names = build_poly_library(X[1, :], X[2, :];
                                         order=polyorder, varnames=tuple(state_labels...))
    thresholds = [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5]

    sweep_file = joinpath(OUTDIR, "sindy_$(spec_name)_stlsq_sweep.txt")
    open(sweep_file, "w") do io
        println(io, "STLSQ Threshold Sweep — $spec_name")
        println(io, "Features: ", join(feat_names, ", "))
        println(io, "")

        for λ in thresholds
            println(io, "=== threshold = $λ ===")
            @printf("  threshold = %.4f:\n", λ)
            for eq_i in 1:n
                y = dX[eq_i, :]
                coeffs, mse = stlsq_fit(Phi, y, λ)
                active = findall(abs.(coeffs) .> 0)
                if isempty(active)
                    terms = "(all zeroed)"
                else
                    terms = join([@sprintf("%.6f*%s", coeffs[k], feat_names[k]) for k in active], " + ")
                end
                @printf("    d(%s)/dt [%d terms] MSE=%.6e : %s\n",
                        state_labels[eq_i], length(active), mse, terms)
                println(io, @sprintf("  d(%s)/dt [%d terms] MSE=%.6e : %s",
                        state_labels[eq_i], length(active), mse, terms))
            end
            println(io, "")
        end
    end
    println("  Saved: sindy_$(spec_name)_stlsq_sweep.txt")
end

## ── DataDrivenDiffEq STLSQ (supplementary — may fail) ──────────────────────

function run_sindy_ddd(X, dX, t, spec_name)
    println("\n=== DataDrivenDiffEq STLSQ (supplementary) — $spec_name ===")
    try
        n = size(X, 1)
        basis = sindy_poly_basis(n; polyorder=polyorder)
        prob  = DataDrivenDiffEq.ContinuousDataDrivenProblem(X, t, dX)
        opt   = DataDrivenSparse.STLSQ(exp10.(-4:0.5:0))
        opts  = DataDrivenDiffEq.DataDrivenCommonOptions(digits=4)

        res    = DataDrivenDiffEq.solve(prob, basis, opt; options=opts)
        system = DataDrivenDiffEq.get_basis(res)
        eqs    = ModelingToolkit.equations(system)

        println("  Discovered equations (DataDrivenDiffEq STLSQ):")
        for (i, eq) in enumerate(eqs)
            println("    eq$i: ", eq)
        end

        eq_file = joinpath(OUTDIR, "sindy_$(spec_name)_ddd_equations.txt")
        open(eq_file, "w") do io
            println(io, "DataDrivenDiffEq STLSQ — $spec_name")
            println(io, "polyorder=$polyorder, α_tvdiff=$default_alpha")
            println(io, "")
            for (i, eq) in enumerate(eqs)
                println(io, "eq$i: ", eq)
            end
        end

        # Trajectory rollout
        try
            Xhat = predict_sindy(res, X[:, 1], t)
            println("  Rollout succeeded: final state = ", Xhat[:, end])
        catch e
            println("  Rollout diverged or failed: ", e)
        end

        return res, system
    catch e
        println("  DataDrivenDiffEq STLSQ failed: ", e)
        eq_file = joinpath(OUTDIR, "sindy_$(spec_name)_ddd_equations.txt")
        open(eq_file, "w") do io
            println(io, "DataDrivenDiffEq STLSQ — $spec_name — FAILED")
            println(io, string(e))
        end
        return nothing, nothing
    end
end

## ── GaussianKernel robustness check (supplementary — may fail) ──────────────

function run_sindy_gaussian(X, t, spec_name)
    println("\n=== SINDy (GaussianKernel robustness check) — $spec_name ===")
    try
        res, system, _ = fit_sindy_sr3(t, X;
            polyorder=polyorder,
            λs=exp10.(-6:0.5:0), nu=1.0,
            proximal=DataDrivenSparse.SoftThreshold(),
            rng=ScientificML.StableRNGs.StableRNG(1))

        eqs = ModelingToolkit.equations(system)
        println("  Discovered equations (GaussianKernel + SR3):")
        for (i, eq) in enumerate(eqs)
            println("    eq$i: ", eq)
        end

        eq_file = joinpath(OUTDIR, "sindy_$(spec_name)_gaussian_equations.txt")
        open(eq_file, "w") do io
            println(io, "SINDy (GaussianKernel + SR3) — $spec_name (robustness check)")
            println(io, "polyorder=$polyorder")
            println(io, "")
            for (i, eq) in enumerate(eqs)
                println(io, "eq$i: ", eq)
            end
        end
        return res, system
    catch e
        println("  GaussianKernel SINDy failed: ", e)
        eq_file = joinpath(OUTDIR, "sindy_$(spec_name)_gaussian_equations.txt")
        open(eq_file, "w") do io
            println(io, "SINDy (GaussianKernel + SR3) — $spec_name — FAILED")
            println(io, string(e))
        end
        return nothing, nothing
    end
end

## ── Run all methods on both specifications ──────────────────────────────────

# Classical
X_class  = vcat(v_c', u_c')
dX_class = vcat(dv_class', du_class')

run_exhaustive_ols(X_class, dX_class, t_c, "classical", ["v", "u"])
run_stlsq_sweep(X_class, dX_class, t_c, "classical", ["v", "u"])
run_sindy_ddd(X_class, dX_class, t_c, "classical")
run_sindy_gaussian(X_class, t_c, "classical")

# Structuralist
X_struct  = vcat(tcu_s', u_s')
dX_struct = vcat(dtcu_struct', du_struct_d')

run_exhaustive_ols(X_struct, dX_struct, t_s, "structuralist", ["tcu", "u"])
run_stlsq_sweep(X_struct, dX_struct, t_s, "structuralist", ["tcu", "u"])
run_sindy_ddd(X_struct, dX_struct, t_s, "structuralist")
run_sindy_gaussian(X_struct, t_s, "structuralist")

println("\n✓ Part 1 complete. Results saved to $OUTDIR")

#!/usr/bin/env julia
#=
  Real-data UDE+SINDy experiment: U.S. macroeconomic Goodwin dynamics
  Part 2 — UDE training and post-training exhaustive SINDy
  for the Classical (v, u) specification only.
=#

## ── Setup ────────────────────────────────────────────────────────────────────

include(joinpath(@__DIR__, "..", "src", "ScientificML.jl"))
using .ScientificML

using Pkg
for pkg in ["Downloads", "XLSX", "NoiseRobustDifferentiation",
            "Lux", "ComponentArrays", "Optimization", "OptimizationOptimisers",
            "OptimizationOptimJL", "LineSearches", "SciMLSensitivity",
            "Zygote", "JLD2"]
    haskey(Pkg.project().dependencies, pkg) || Pkg.add(pkg)
end

using Downloads, XLSX
using NoiseRobustDifferentiation
using Statistics, LinearAlgebra, Printf
using Plots; gr()

using Lux, ComponentArrays
using OrdinaryDiffEq, SciMLSensitivity
using Optimization, OptimizationOptimisers, OptimizationOptimJL
using LineSearches, Zygote
using JLD2
using Random

import .ScientificML: write_csv, append_csv_row

const OUTDIR = joinpath(@__DIR__, "..", "results", "realdata")
mkpath(OUTDIR)

## ── Data Loading (same as Part 1) ────────────────────────────────────────────

println("Loading data...")
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
wage_share_raw = [to_float(sheet[i, 4]) for i in 2:size(sheet, 1)]

# Trim rows where core columns are NaN (trailing empty rows from XLSX)
valid_core = .!isnan.(unrate_raw) .& .!isnan.(wage_share_raw)
unrate     = unrate_raw[valid_core]
wage_share = wage_share_raw[valid_core]
println("After trimming NaN rows: $(length(unrate)) valid (dropped $(sum(.!valid_core)))")

# Classical specification
v_all  = 1.0 .- unrate ./ 100.0
u_all  = wage_share ./ 100.0
N      = length(v_all)
t_all  = Float64.(0:N-1)
X_data = vcat(v_all', u_all')  # 2 × N
println("Classical spec: N = $N")

# TVDiff derivatives (for eta estimation and post-training comparison)
α_tvdiff = 0.01
iter_tvdiff = 1000
dv_tvd = tvdiff(v_all, iter_tvdiff, α_tvdiff; scale="large", dx=1.0)[1:N]
du_tvd = tvdiff(u_all, iter_tvdiff, α_tvdiff; scale="large", dx=1.0)[1:N]
dX_tvd = vcat(dv_tvd', du_tvd')

## ── UDE Structure ────────────────────────────────────────────────────────────

# RBF activation
rbf(x) = exp.(-(x .^ 2))

# 4-layer RBF network: 2 → 15 → 15 → 15 → 2
nn = Lux.Chain(
    Lux.Dense(2, 15, rbf),
    Lux.Dense(15, 15, rbf),
    Lux.Dense(15, 15, rbf),
    Lux.Dense(15, 2),
)

rng_init = Lux.replicate(Random.MersenneTwister(42))
ps_init, nn_st = Lux.setup(rng_init, nn)
nn_st = Lux.testmode(nn_st)

# ── Approach A: Estimate eta from TVDiff derivatives
eta1_A = median(dv_tvd ./ v_all)
eta2_A = -median(du_tvd ./ u_all)
println(@sprintf("Approach A — η₁ = %.6f, η₂ = %.6f", eta1_A, eta2_A))

# ── Approach B: Neutral baseline
eta1_B = 0.01
eta2_B = 0.01

## ── Multiple Shooting Setup ──────────────────────────────────────────────────

const WIN_SIZE   = 20    # window size in points (~20 quarters)
const WIN_STRIDE = 10    # stride
const λ_CONT     = 0.1   # continuity penalty weight
const λ_L2       = 1e-5  # L2 regularization

# Window indices
win_starts = collect(1:WIN_STRIDE:N)
# Ensure last window covers the end
win_ranges = [(s, min(s + WIN_SIZE - 1, N)) for s in win_starts]
# Drop trivially short windows
win_ranges = filter(w -> w[2] - w[1] >= 2, win_ranges)
n_wins = length(win_ranges)
println("Multiple shooting: $n_wins windows of size ≤$WIN_SIZE, stride=$WIN_STRIDE")

function build_ude_fn(eta1, eta2, nn_model, st)
    function ude!(du, x, p, t)
        nn_out, _ = nn_model(x, p, st)
        du[1] = eta1 * x[1] + nn_out[1]
        du[2] = -eta2 * x[2] + nn_out[2]
    end
    return ude!
end

function ms_loss(p, extra, ude_fn!)
    t, X = extra
    data_loss = 0.0
    cont_loss = 0.0
    n_ok = 0

    for (i, (i0, i1)) in enumerate(win_ranges)
        t_seg = t[i0:i1]
        ic    = X[:, i0]

        prob = ODEProblem(ude_fn!, ic, (t_seg[1], t_seg[end]), p)
        sol  = solve(prob, Tsit5(); abstol=1e-6, reltol=1e-6,
                     saveat=t_seg, maxiters=50_000,
                     sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)))

        if sol.retcode != ReturnCode.Success || length(sol.t) != length(t_seg)
            continue
        end

        Xhat       = Array(sol)
        data_loss += mean((X[:, i0:i1] .- Xhat) .^ 2)
        n_ok      += 1

        # Continuity: predicted end should match next window start
        if i < n_wins
            next_start = win_ranges[i + 1][1]
            cont_loss += mean((Xhat[:, end] .- X[:, next_start]) .^ 2)
        end
    end

    n_ok == 0 && return Inf
    return data_loss / n_ok + λ_CONT * cont_loss / max(n_ok - 1, 1) + λ_L2 * sum(abs2, p)
end

## ── Training Function ────────────────────────────────────────────────────────

function train_ude(eta1, eta2, label)
    println("\n=== UDE Training — $label (η₁=$eta1, η₂=$eta2) ===")

    ude_fn! = build_ude_fn(eta1, eta2, nn, nn_st)
    p0 = ComponentArray(ps_init)
    extra = (t_all, X_data)

    losses = Float64[]

    function callback(state, loss)
        push!(losses, loss)
        if length(losses) % 100 == 0
            @printf("  [%s] iter %5d  loss = %.6e\n", label, length(losses), loss)
        end
        return false
    end

    opt_fn = Optimization.OptimizationFunction(
        (p, e) -> ms_loss(p, e, ude_fn!),
        Optimization.AutoZygote())

    p_trained = p0
    try
        # Phase 1: Adam(0.01) × 3000
        println("  Phase 1: Adam(lr=0.01), 3000 iters")
        prob1 = Optimization.OptimizationProblem(opt_fn, p0, extra)
        res1  = Optimization.solve(prob1, OptimizationOptimisers.Adam(0.01);
                                   maxiters=3000, callback=callback)
        p_trained = res1.u

        # Phase 2: Adam(0.001) × 2000
        println("  Phase 2: Adam(lr=0.001), 2000 iters")
        prob2 = Optimization.OptimizationProblem(opt_fn, res1.u, extra)
        res2  = Optimization.solve(prob2, OptimizationOptimisers.Adam(0.001);
                                   maxiters=2000, callback=callback)
        p_trained = res2.u

        # Phase 3: BFGS × 1000
        println("  Phase 3: BFGS, 1000 iters")
        prob3 = Optimization.OptimizationProblem(opt_fn, res2.u, extra)
        res3  = Optimization.solve(prob3,
                    OptimizationOptimJL.BFGS(linesearch=LineSearches.BackTracking());
                    maxiters=1000, callback=callback)
        p_trained = res3.u
    catch e
        println("  Training stopped: ", e)
    end

    final_loss = isempty(losses) ? Inf : losses[end]
    println(@sprintf("  Final loss (%s): %.6e  (%d iters)", label, final_loss, length(losses)))

    return p_trained, losses
end

## ── Run Both Approaches ──────────────────────────────────────────────────────

p_A, losses_A = train_ude(eta1_A, eta2_A, "Approach_A")
p_B, losses_B = train_ude(eta1_B, eta2_B, "Approach_B")

# Pick best
best_label = losses_A[end] < losses_B[end] ? "A" : "B"
p_trained  = best_label == "A" ? p_A : p_B
losses     = best_label == "A" ? losses_A : losses_B
eta1_best  = best_label == "A" ? eta1_A : eta1_B
eta2_best  = best_label == "A" ? eta2_A : eta2_B
println("\nBest approach: $best_label (final loss: $(losses[end]))")

## ── Training Loss Curve ──────────────────────────────────────────────────────

p_loss = plot(size=(800, 400))
plot!(p_loss, losses_A; label="Approach A (data-estimated η)", lw=1.5, yscale=:log10)
plot!(p_loss, losses_B; label="Approach B (η=0.01 baseline)", lw=1.5, yscale=:log10)
xlabel!(p_loss, "Iteration")
ylabel!(p_loss, "Loss (log scale)")
title!(p_loss, "UDE Training Loss — Classical Spec")
savefig(p_loss, joinpath(OUTDIR, "ude_training_loss.png"))
println("Saved: ude_training_loss.png")

## ── Evaluate Trained UDE ─────────────────────────────────────────────────────

ude_fn_best! = build_ude_fn(eta1_best, eta2_best, nn, nn_st)

# Full trajectory rollout
println("\nAttempting full trajectory rollout...")
Xhat_ude = fill(NaN, 2, N)
try
    prob_full = ODEProblem(ude_fn_best!, X_data[:, 1], (t_all[1], t_all[end]), p_trained)
    sol_full  = solve(prob_full, Tsit5(); abstol=1e-8, reltol=1e-8,
                      saveat=t_all, maxiters=500_000)
    if sol_full.retcode == ReturnCode.Success
        Xhat_ude = Array(sol_full)
        println("Rollout succeeded.")
    else
        println("Rollout returned: ", sol_full.retcode)
    end
catch e
    println("Rollout failed: ", e)
end

# Trajectory fit plot
p_traj = plot(layout=(2, 1), size=(900, 500))
plot!(p_traj, t_all, v_all; subplot=1, label="data v", lw=1.5)
plot!(p_traj, t_all, Xhat_ude[1, :]; subplot=1, label="UDE v", lw=1.5, ls=:dash)
title!(p_traj, "Employment rate v"; subplot=1)
plot!(p_traj, t_all, u_all; subplot=2, label="data u", lw=1.5)
plot!(p_traj, t_all, Xhat_ude[2, :]; subplot=2, label="UDE u", lw=1.5, ls=:dash)
title!(p_traj, "Wage share u"; subplot=2)
xlabel!(p_traj, "Quarter"; subplot=2)
savefig(p_traj, joinpath(OUTDIR, "ude_trajectory_fit.png"))
println("Saved: ude_trajectory_fit.png")

## ── NN Outputs at Data States ────────────────────────────────────────────────

nn_out = zeros(2, N)
for j in 1:N
    out, _ = nn(X_data[:, j], p_trained, nn_st)
    nn_out[:, j] = out
end

# Save NN outputs
csv_path = joinpath(OUTDIR, "ude_classical_nn_outputs.csv")
header = reshape(["t", "v", "u", "N1", "N2"], 1, :)
write_csv(csv_path, header)
for j in 1:N
    row = reshape([t_all[j], v_all[j], u_all[j], nn_out[1, j], nn_out[2, j]], 1, :)
    append_csv_row(csv_path, row)
end
println("Saved: ude_classical_nn_outputs.csv")

## ── Primary: Sparse Regression on NN Outputs (ADMM + STLSQ) ────────────────

println("\n=== Primary: Sparse Regression on NN Outputs (Classical) ===")

import .ScientificML: fit_sindy_from_nn

sparse_reg_results_classical = Dict{String, Any}()
for meth in [:ADMM, :STLSQ, :SR3]
    label_m = string(meth)
    println("\n  Method: $label_m")
    try
        res_sr, sys_sr, params_sr = fit_sindy_from_nn(
            X_data, nn_out;
            polyorder = 2, method = meth,
            batchsize = min(30, N - 10),   # smaller for real data (N~311)
            rng = ScientificML.StableRNGs.StableRNG(42))

        eqs = ScientificML.ModelingToolkit.equations(sys_sr)
        println("  Discovered equations ($label_m):")
        for (i, eq) in enumerate(eqs)
            println("    eq$i: ", eq)
        end
        sparse_reg_results_classical[label_m] = (res = res_sr, system = sys_sr, params = params_sr)
    catch e
        println("  $label_m failed: ", e)
        sparse_reg_results_classical[label_m] = nothing
    end
end

# Save sparse regression equations to text file
sparse_eq_file = joinpath(OUTDIR, "ude_classical_sparse_equations.txt")
open(sparse_eq_file, "w") do io
    println(io, "Sparse Regression on NN Outputs — Classical Spec")
    println(io, "")
    for meth in ["ADMM", "STLSQ", "SR3"]
        sr = get(sparse_reg_results_classical, meth, nothing)
        if sr !== nothing
            eqs = ScientificML.ModelingToolkit.equations(sr.system)
            println(io, "--- $meth ---")
            for (i, eq) in enumerate(eqs)
                println(io, "  eq$i: ", eq)
            end
            println(io, "")
        else
            println(io, "--- $meth --- FAILED")
            println(io, "")
        end
    end
end
println("Saved: ude_classical_sparse_equations.txt")

## ── Supplementary: Exhaustive SINDy on NN Outputs ──────────────────────────

println("\n=== Supplementary: Exhaustive SINDy on NN Outputs ===")

# Feature library: {1, v, u, v*u, v^2, u^2}
candidate_names = ["1", "v", "u", "v*u", "v^2", "u^2"]
Phi = hcat(
    ones(N),
    v_all,
    u_all,
    v_all .* u_all,
    v_all .^ 2,
    u_all .^ 2,
)  # N × 6

# All single-term and two-term models
single_models = [[i] for i in 1:6]
pair_models   = [[i, j] for i in 1:6 for j in (i+1):6]
all_models    = vcat(single_models, pair_models)

function ols_fit(Phi, y, col_idx)
    Phi_sub = Phi[:, col_idx]
    coeffs  = Phi_sub \ y
    pred    = Phi_sub * coeffs
    mse     = mean((y .- pred) .^ 2)
    return coeffs, mse
end

function exhaustive_sindy(Phi, y, all_models, candidate_names, eq_label)
    results = []
    for idx in all_models
        coeffs, mse = ols_fit(Phi, y, idx)
        terms = join([
            @sprintf("%.6f*%s", coeffs[k], candidate_names[idx[k]])
            for k in 1:length(idx)
        ], " + ")
        push!(results, (idx=idx, coeffs=coeffs, mse=mse, terms=terms))
    end
    sort!(results, by=r -> r.mse)

    println("  Best models for $eq_label:")
    for (rank, r) in enumerate(results[1:min(5, length(results))])
        @printf("    #%d MSE=%.6e : %s\n", rank, r.mse, r.terms)
    end
    return results
end

results_N1 = exhaustive_sindy(Phi, nn_out[1, :], all_models, candidate_names, "N₁(v,u)")
results_N2 = exhaustive_sindy(Phi, nn_out[2, :], all_models, candidate_names, "N₂(v,u)")

# Save exhaustive search results
csv_exh = joinpath(OUTDIR, "ude_classical_exhaustive.csv")
header = reshape(["equation", "rank", "terms", "mse"], 1, :)
write_csv(csv_exh, header)
for (eq_label, results) in [("N1", results_N1), ("N2", results_N2)]
    for (rank, r) in enumerate(results)
        row = reshape([eq_label, rank, replace(r.terms, ',' => ';'), r.mse], 1, :)
        append_csv_row(csv_exh, row)
    end
end
println("Saved: ude_classical_exhaustive.csv")

## ── NN vs Best Symbolic Fit Plot ─────────────────────────────────────────────

best_N1 = results_N1[1]
best_N2 = results_N2[1]

pred_N1 = Phi[:, best_N1.idx] * best_N1.coeffs
pred_N2 = Phi[:, best_N2.idx] * best_N2.coeffs

p_comp = plot(layout=(2, 1), size=(900, 500))
plot!(p_comp, t_all, nn_out[1, :]; subplot=1, label="NN N₁(v,u)", lw=1.5)
plot!(p_comp, t_all, pred_N1; subplot=1, label="Best symbolic: $(best_N1.terms)",
      lw=1.5, ls=:dash)
title!(p_comp, "N₁ equation"; subplot=1)
plot!(p_comp, t_all, nn_out[2, :]; subplot=2, label="NN N₂(v,u)", lw=1.5)
plot!(p_comp, t_all, pred_N2; subplot=2, label="Best symbolic: $(best_N2.terms)",
      lw=1.5, ls=:dash)
title!(p_comp, "N₂ equation"; subplot=2)
xlabel!(p_comp, "Quarter"; subplot=2)
savefig(p_comp, joinpath(OUTDIR, "ude_sindy_comparison.png"))
println("Saved: ude_sindy_comparison.png")

## ── Save Trained Parameters ──────────────────────────────────────────────────

# Build sparse regression equation strings + derivative predictions for saving
sparse_eq_strings_classical = Dict{String, Any}()
sparse_dX_classical = Dict{String, Any}()
for meth in ["ADMM", "STLSQ", "SR3"]
    sr = get(sparse_reg_results_classical, meth, nothing)
    if sr !== nothing
        try
            eqs = ScientificML.ModelingToolkit.equations(sr.system)
            sparse_eq_strings_classical[meth] = [string(eq.rhs) for eq in eqs]
        catch
            sparse_eq_strings_classical[meth] = nothing
        end
        try
            dX_sr = eval_sindy_at_states(sr.res, t_all, X_data)
            dX_sr[1, :] .+= eta1_best .* X_data[1, :]
            dX_sr[2, :] .-= eta2_best .* X_data[2, :]
            sparse_dX_classical[meth] = dX_sr
        catch
            sparse_dX_classical[meth] = nothing
        end
    else
        sparse_eq_strings_classical[meth] = nothing
        sparse_dX_classical[meth] = nothing
    end
end

jld2_path = joinpath(OUTDIR, "ude_classical_params.jld2")
jldsave(jld2_path;
    p_trained,
    eta1 = eta1_best,
    eta2 = eta2_best,
    best_approach = best_label,
    losses_A, losses_B,
    nn_out,
    X_data, t_all,
    dX_tvd,
    exhaustive_N1 = [(idx=r.idx, mse=r.mse, terms=r.terms) for r in results_N1[1:min(5, end)]],
    exhaustive_N2 = [(idx=r.idx, mse=r.mse, terms=r.terms) for r in results_N2[1:min(5, end)]],
    sparse_eq_strings = sparse_eq_strings_classical,
    sparse_dX = sparse_dX_classical,
)
println("Saved: ude_classical_params.jld2")

println("\n✓ Classical specification complete.")

## ═══════════════════════════════════════════════════════════════════════════
## Structuralist Specification: tcu/100, u
## ═══════════════════════════════════════════════════════════════════════════

println("\n" * "="^70)
println("  STRUCTURALIST SPECIFICATION (tcu, u)")
println("="^70)

## ── Structuralist Data Loading ──────────────────────────────────────────────

tcu_raw = [to_float(sheet[i, 3]) for i in 2:size(sheet, 1)]
tcu_full = tcu_raw[valid_core]

mask_tcu   = .!isnan.(tcu_full)
tcu_struct = tcu_full[mask_tcu] ./ 100.0
u_struct   = wage_share[mask_tcu] ./ 100.0
N_struct   = length(tcu_struct)
t_struct   = Float64.(0:N_struct-1)
X_struct   = vcat(tcu_struct', u_struct')   # 2 × N_struct
println("Structuralist spec: N = $N_struct")

# TVDiff derivatives for structuralist
dtcu_tvd = tvdiff(tcu_struct, iter_tvdiff, α_tvdiff; scale="large", dx=1.0)[1:N_struct]
du_s_tvd = tvdiff(u_struct, iter_tvdiff, α_tvdiff; scale="large", dx=1.0)[1:N_struct]
dX_struct_tvd = vcat(dtcu_tvd', du_s_tvd')

## ── Structuralist UDE Setup ─────────────────────────────────────────────────

# Fresh NN for structuralist (same architecture)
nn_struct = Lux.Chain(
    Lux.Dense(2, 15, rbf),
    Lux.Dense(15, 15, rbf),
    Lux.Dense(15, 15, rbf),
    Lux.Dense(15, 2),
)
ps_struct_init, nn_st_struct = Lux.setup(rng_init, nn_struct)
nn_st_struct = Lux.testmode(nn_st_struct)

# Estimate eta for structuralist
eta1_struct_A = median(dtcu_tvd ./ tcu_struct)
eta2_struct_A = -median(du_s_tvd ./ u_struct)
println(@sprintf("Structuralist Approach A — η₁ = %.6f, η₂ = %.6f", eta1_struct_A, eta2_struct_A))

eta1_struct_B = 0.01
eta2_struct_B = 0.01

# Multiple shooting windows for structuralist (N~235)
win_starts_s = collect(1:WIN_STRIDE:N_struct)
win_ranges_s = [(s, min(s + WIN_SIZE - 1, N_struct)) for s in win_starts_s]
win_ranges_s = filter(w -> w[2] - w[1] >= 2, win_ranges_s)
n_wins_s = length(win_ranges_s)
println("Structuralist windows: $n_wins_s of size <=$WIN_SIZE")

function ms_loss_struct(p, extra, ude_fn!)
    t, X = extra
    data_loss = 0.0
    cont_loss = 0.0
    n_ok = 0

    for (i, (i0, i1)) in enumerate(win_ranges_s)
        t_seg = t[i0:i1]
        ic    = X[:, i0]

        prob = ODEProblem(ude_fn!, ic, (t_seg[1], t_seg[end]), p)
        sol  = solve(prob, Tsit5(); abstol=1e-6, reltol=1e-6,
                     saveat=t_seg, maxiters=50_000,
                     sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)))

        if sol.retcode != ReturnCode.Success || length(sol.t) != length(t_seg)
            continue
        end

        Xhat       = Array(sol)
        data_loss += mean((X[:, i0:i1] .- Xhat) .^ 2)
        n_ok      += 1

        if i < n_wins_s
            next_start = win_ranges_s[i + 1][1]
            cont_loss += mean((Xhat[:, end] .- X[:, next_start]) .^ 2)
        end
    end

    n_ok == 0 && return Inf
    return data_loss / n_ok + λ_CONT * cont_loss / max(n_ok - 1, 1) + λ_L2 * sum(abs2, p)
end

## ── Structuralist Training ──────────────────────────────────────────────────

function train_ude_struct(eta1, eta2, label)
    println("\n=== UDE Training — Structuralist $label (η₁=$eta1, η₂=$eta2) ===")

    ude_fn! = build_ude_fn(eta1, eta2, nn_struct, nn_st_struct)
    p0 = ComponentArray(ps_struct_init)
    extra = (t_struct, X_struct)

    losses = Float64[]

    function callback(state, loss)
        push!(losses, loss)
        if length(losses) % 100 == 0
            @printf("  [%s] iter %5d  loss = %.6e\n", label, length(losses), loss)
        end
        return false
    end

    opt_fn = Optimization.OptimizationFunction(
        (p, e) -> ms_loss_struct(p, e, ude_fn!),
        Optimization.AutoZygote())

    p_trained = p0
    try
        println("  Phase 1: Adam(lr=0.01), 3000 iters")
        prob1 = Optimization.OptimizationProblem(opt_fn, p0, extra)
        res1  = Optimization.solve(prob1, OptimizationOptimisers.Adam(0.01);
                                   maxiters=3000, callback=callback)
        p_trained = res1.u

        println("  Phase 2: Adam(lr=0.001), 2000 iters")
        prob2 = Optimization.OptimizationProblem(opt_fn, res1.u, extra)
        res2  = Optimization.solve(prob2, OptimizationOptimisers.Adam(0.001);
                                   maxiters=2000, callback=callback)
        p_trained = res2.u

        println("  Phase 3: BFGS, 1000 iters")
        prob3 = Optimization.OptimizationProblem(opt_fn, res2.u, extra)
        res3  = Optimization.solve(prob3,
                    OptimizationOptimJL.BFGS(linesearch=LineSearches.BackTracking());
                    maxiters=1000, callback=callback)
        p_trained = res3.u
    catch e
        println("  Training stopped: ", e)
    end

    final_loss = isempty(losses) ? Inf : losses[end]
    println(@sprintf("  Final loss (%s): %.6e  (%d iters)", label, final_loss, length(losses)))

    return p_trained, losses
end

p_sA, losses_sA = train_ude_struct(eta1_struct_A, eta2_struct_A, "Approach_A")
p_sB, losses_sB = train_ude_struct(eta1_struct_B, eta2_struct_B, "Approach_B")

best_label_s  = losses_sA[end] < losses_sB[end] ? "A" : "B"
p_trained_s   = best_label_s == "A" ? p_sA : p_sB
losses_s      = best_label_s == "A" ? losses_sA : losses_sB
eta1_best_s   = best_label_s == "A" ? eta1_struct_A : eta1_struct_B
eta2_best_s   = best_label_s == "A" ? eta2_struct_A : eta2_struct_B
println("\nBest approach (structuralist): $best_label_s (final loss: $(losses_s[end]))")

## ── Structuralist Training Loss Curve ───────────────────────────────────────

p_loss_s = plot(size=(800, 400))
plot!(p_loss_s, losses_sA; label="Approach A (data-estimated η)", lw=1.5, yscale=:log10)
plot!(p_loss_s, losses_sB; label="Approach B (η=0.01 baseline)", lw=1.5, yscale=:log10)
xlabel!(p_loss_s, "Iteration")
ylabel!(p_loss_s, "Loss (log scale)")
title!(p_loss_s, "UDE Training Loss — Structuralist Spec")
savefig(p_loss_s, joinpath(OUTDIR, "ude_structuralist_training_loss.png"))
println("Saved: ude_structuralist_training_loss.png")

## ── Evaluate Structuralist UDE ──────────────────────────────────────────────

ude_fn_best_s! = build_ude_fn(eta1_best_s, eta2_best_s, nn_struct, nn_st_struct)

println("\nAttempting structuralist full trajectory rollout...")
Xhat_ude_s = fill(NaN, 2, N_struct)
try
    prob_full_s = ODEProblem(ude_fn_best_s!, X_struct[:, 1], (t_struct[1], t_struct[end]), p_trained_s)
    sol_full_s  = solve(prob_full_s, Tsit5(); abstol=1e-8, reltol=1e-8,
                        saveat=t_struct, maxiters=500_000)
    if sol_full_s.retcode == ReturnCode.Success
        Xhat_ude_s = Array(sol_full_s)
        println("Rollout succeeded.")
    else
        println("Rollout returned: ", sol_full_s.retcode)
    end
catch e
    println("Rollout failed: ", e)
end

# Trajectory fit plot
p_traj_s = plot(layout=(2, 1), size=(900, 500))
plot!(p_traj_s, t_struct, tcu_struct; subplot=1, label="data tcu", lw=1.5)
plot!(p_traj_s, t_struct, Xhat_ude_s[1, :]; subplot=1, label="UDE tcu", lw=1.5, ls=:dash)
title!(p_traj_s, "Capacity utilization (tcu/100)"; subplot=1)
plot!(p_traj_s, t_struct, u_struct; subplot=2, label="data u", lw=1.5)
plot!(p_traj_s, t_struct, Xhat_ude_s[2, :]; subplot=2, label="UDE u", lw=1.5, ls=:dash)
title!(p_traj_s, "Wage share u"; subplot=2)
xlabel!(p_traj_s, "Quarter"; subplot=2)
savefig(p_traj_s, joinpath(OUTDIR, "ude_structuralist_trajectory_fit.png"))
println("Saved: ude_structuralist_trajectory_fit.png")

## ── Structuralist NN Outputs ────────────────────────────────────────────────

nn_out_s = zeros(2, N_struct)
for j in 1:N_struct
    out, _ = nn_struct(X_struct[:, j], p_trained_s, nn_st_struct)
    nn_out_s[:, j] = out
end

csv_path_s = joinpath(OUTDIR, "ude_structuralist_nn_outputs.csv")
header_s = reshape(["t", "tcu", "u", "N1", "N2"], 1, :)
write_csv(csv_path_s, header_s)
for j in 1:N_struct
    row = reshape([t_struct[j], tcu_struct[j], u_struct[j], nn_out_s[1, j], nn_out_s[2, j]], 1, :)
    append_csv_row(csv_path_s, row)
end
println("Saved: ude_structuralist_nn_outputs.csv")

## ── Primary: Sparse Regression on NN Outputs (Structuralist) ────────────────

println("\n=== Primary: Sparse Regression on NN Outputs (Structuralist) ===")

sparse_reg_results_struct = Dict{String, Any}()
for meth in [:ADMM, :STLSQ, :SR3]
    label_m = string(meth)
    println("\n  Method: $label_m")
    try
        res_sr, sys_sr, params_sr = fit_sindy_from_nn(
            X_struct, nn_out_s;
            polyorder = 2, method = meth,
            batchsize = min(20, N_struct - 10),   # smaller for N~235
            rng = ScientificML.StableRNGs.StableRNG(42))

        eqs = ScientificML.ModelingToolkit.equations(sys_sr)
        println("  Discovered equations ($label_m):")
        for (i, eq) in enumerate(eqs)
            println("    eq$i: ", eq)
        end
        sparse_reg_results_struct[label_m] = (res = res_sr, system = sys_sr, params = params_sr)
    catch e
        println("  $label_m failed: ", e)
        sparse_reg_results_struct[label_m] = nothing
    end
end

# Save sparse regression equations
sparse_eq_file_s = joinpath(OUTDIR, "ude_structuralist_sparse_equations.txt")
open(sparse_eq_file_s, "w") do io
    println(io, "Sparse Regression on NN Outputs — Structuralist Spec")
    println(io, "")
    for meth in ["ADMM", "STLSQ", "SR3"]
        sr = get(sparse_reg_results_struct, meth, nothing)
        if sr !== nothing
            eqs = ScientificML.ModelingToolkit.equations(sr.system)
            println(io, "--- $meth ---")
            for (i, eq) in enumerate(eqs)
                println(io, "  eq$i: ", eq)
            end
            println(io, "")
        else
            println(io, "--- $meth --- FAILED")
            println(io, "")
        end
    end
end
println("Saved: ude_structuralist_sparse_equations.txt")

## ── Supplementary: Exhaustive SINDy on NN Outputs (Structuralist) ───────────

println("\n=== Supplementary: Exhaustive SINDy (Structuralist) ===")

candidate_names_s = ["1", "tcu", "u", "tcu*u", "tcu^2", "u^2"]
Phi_s = hcat(
    ones(N_struct),
    tcu_struct,
    u_struct,
    tcu_struct .* u_struct,
    tcu_struct .^ 2,
    u_struct .^ 2,
)  # N_struct × 6

results_s_N1 = exhaustive_sindy(Phi_s, nn_out_s[1, :], all_models, candidate_names_s, "N₁(tcu,u)")
results_s_N2 = exhaustive_sindy(Phi_s, nn_out_s[2, :], all_models, candidate_names_s, "N₂(tcu,u)")

csv_exh_s = joinpath(OUTDIR, "ude_structuralist_exhaustive.csv")
header_exh = reshape(["equation", "rank", "terms", "mse"], 1, :)
write_csv(csv_exh_s, header_exh)
for (eq_label, results) in [("N1", results_s_N1), ("N2", results_s_N2)]
    for (rank, r) in enumerate(results)
        row = reshape([eq_label, rank, replace(r.terms, ',' => ';'), r.mse], 1, :)
        append_csv_row(csv_exh_s, row)
    end
end
println("Saved: ude_structuralist_exhaustive.csv")

## ── NN vs Best Symbolic Fit Plot (Structuralist) ────────────────────────────

best_s_N1 = results_s_N1[1]
best_s_N2 = results_s_N2[1]

pred_s_N1 = Phi_s[:, best_s_N1.idx] * best_s_N1.coeffs
pred_s_N2 = Phi_s[:, best_s_N2.idx] * best_s_N2.coeffs

p_comp_s = plot(layout=(2, 1), size=(900, 500))
plot!(p_comp_s, t_struct, nn_out_s[1, :]; subplot=1, label="NN N₁(tcu,u)", lw=1.5)
plot!(p_comp_s, t_struct, pred_s_N1; subplot=1, label="Best symbolic: $(best_s_N1.terms)",
      lw=1.5, ls=:dash)
title!(p_comp_s, "N₁ equation (structuralist)"; subplot=1)
plot!(p_comp_s, t_struct, nn_out_s[2, :]; subplot=2, label="NN N₂(tcu,u)", lw=1.5)
plot!(p_comp_s, t_struct, pred_s_N2; subplot=2, label="Best symbolic: $(best_s_N2.terms)",
      lw=1.5, ls=:dash)
title!(p_comp_s, "N₂ equation (structuralist)"; subplot=2)
xlabel!(p_comp_s, "Quarter"; subplot=2)
savefig(p_comp_s, joinpath(OUTDIR, "ude_structuralist_sindy_comparison.png"))
println("Saved: ude_structuralist_sindy_comparison.png")

## ── Save Structuralist Parameters ───────────────────────────────────────────

sparse_eq_strings_struct = Dict{String, Any}()
sparse_dX_struct = Dict{String, Any}()
for meth in ["ADMM", "STLSQ", "SR3"]
    sr = get(sparse_reg_results_struct, meth, nothing)
    if sr !== nothing
        try
            eqs = ScientificML.ModelingToolkit.equations(sr.system)
            sparse_eq_strings_struct[meth] = [string(eq.rhs) for eq in eqs]
        catch
            sparse_eq_strings_struct[meth] = nothing
        end
        try
            dX_sr = eval_sindy_at_states(sr.res, t_struct, X_struct)
            dX_sr[1, :] .+= eta1_best_s .* X_struct[1, :]
            dX_sr[2, :] .-= eta2_best_s .* X_struct[2, :]
            sparse_dX_struct[meth] = dX_sr
        catch
            sparse_dX_struct[meth] = nothing
        end
    else
        sparse_eq_strings_struct[meth] = nothing
        sparse_dX_struct[meth] = nothing
    end
end

jld2_path_s = joinpath(OUTDIR, "ude_structuralist_params.jld2")
jldsave(jld2_path_s;
    p_trained = p_trained_s,
    eta1 = eta1_best_s,
    eta2 = eta2_best_s,
    best_approach = best_label_s,
    losses_A = losses_sA, losses_B = losses_sB,
    nn_out = nn_out_s,
    X_data = X_struct, t_all = t_struct,
    dX_tvd = dX_struct_tvd,
    exhaustive_N1 = [(idx=r.idx, mse=r.mse, terms=r.terms) for r in results_s_N1[1:min(5, end)]],
    exhaustive_N2 = [(idx=r.idx, mse=r.mse, terms=r.terms) for r in results_s_N2[1:min(5, end)]],
    sparse_eq_strings = sparse_eq_strings_struct,
    sparse_dX = sparse_dX_struct,
)
println("Saved: ude_structuralist_params.jld2")

println("\n✓ Part 2 complete. All results saved to $OUTDIR")

# experiments/exp_ude_nonstationary.jl
# Stage 2 — UDE on nonstationary Goodwin: noise sweep
# Noise=0 run is cached; only missing noise levels are computed.
#
# Usage:  julia --project=. experiments/exp_ude_nonstationary.jl

include(joinpath(@__DIR__, "..", "src", "ScientificML.jl"))
using .ScientificML

include(joinpath(@__DIR__, "..", "src", "simulate_nonstationary.jl"))

using OrdinaryDiffEq, SciMLSensitivity
using Optimization, OptimizationOptimisers, OptimizationOptimJL, LineSearches
using ComponentArrays, Lux, Zygote
using StableRNGs, LinearAlgebra, Statistics, JLD2

# ─── Data ─────────────────────────────────────────────────────────────────────

x0_true, regime_configs, breakpoints = nonstationary_goodwin_configs()
t_full, X_clean = simulate_nonstationary(x0_true, regime_configs; saveat = 0.5)
N = size(X_clean, 2)
println("Data: $(size(X_clean)) — $N points over [$(t_full[1]), $(t_full[end])]")

# ─── Neural network (shared architecture) ────────────────────────────────────

const ETA1_FIXED = 0.05
const ETA2_FIXED = 0.05

rbf(x) = exp.(-(x .^ 2))

nn = Lux.Chain(
    Lux.Dense(2, 15, rbf),
    Lux.Dense(15, 15, rbf),
    Lux.Dense(15, 15, rbf),
    Lux.Dense(15, 2),
)

# nn_st is stateless (no BatchNorm), safe to share across runs
_, nn_st = Lux.setup(StableRNG(1), nn)

# ─── UDE dynamics ─────────────────────────────────────────────────────────────
# dv/dt = ETA1_FIXED·v + NN₁(v,u)
# du/dt = −ETA2_FIXED·u + NN₂(v,u)

function ude!(du, u, p, t)
    nn_out, _ = nn(u, p, nn_st)
    du[1]     =  ETA1_FIXED * u[1] + nn_out[1]
    du[2]     = -ETA2_FIXED * u[2] + nn_out[2]
end

# ─── Multiple-shooting loss ───────────────────────────────────────────────────

const WIN    = 50     # window size (points)
const λ_CONT = 0.1   # continuity weight
const λ_L2   = 1e-5  # L2 regularisation

let step = WIN - 1
    global WIN_IDXS = collect(1:step:N)
    WIN_IDXS[end] != N && push!(WIN_IDXS, N)
end

function ms_loss(p, (t, X))
    n_wins    = length(WIN_IDXS) - 1
    data_loss = 0.0
    cont_loss = 0.0

    for i in 1:n_wins
        i0, i1 = WIN_IDXS[i], WIN_IDXS[i + 1]
        t_seg   = t[i0:i1]
        ic      = X[:, i0]

        local sol
        try
            prob = ODEProblem(ude!, ic, (t_seg[1], t_seg[end]), p)
            sol  = solve(prob, Tsit5();
                         abstol    = 1e-6, reltol = 1e-6,
                         saveat    = t_seg, maxiters = 50_000,
                         sensealg  = InterpolatingAdjoint(autojacvec = ReverseDiffVJP(true)))
        catch
            return Inf
        end

        (sol.retcode != ReturnCode.Success || length(sol.t) != length(t_seg)) && return Inf

        Xhat       = Array(sol)
        data_loss += mean((X[:, i0:i1] .- Xhat) .^ 2)
        i < n_wins && (cont_loss += mean((Xhat[:, end] .- X[:, WIN_IDXS[i + 1]]) .^ 2))
    end

    return data_loss / n_wins +
           λ_CONT * cont_loss / max(n_wins - 1, 1) +
           λ_L2   * sum(abs2, p)
end

# ─── Feature library for exhaustive SINDy ─────────────────────────────────────

v_all = X_clean[1, :]
u_all = X_clean[2, :]

candidates = [
    ("1",   ones(N)),
    ("v",   v_all),
    ("u",   u_all),
    ("v*u", v_all .* u_all),
    ("v²",  v_all .^ 2),
    ("u²",  u_all .^ 2),
]
feat_names = [c[1] for c in candidates]
Phi        = hcat([c[2] for c in candidates]...)   # N × 6

single_models = [[i]       for i in 1:6]
pair_models   = [[i, j]    for i in 1:6 for j in (i+1):6]
all_models    = vcat(single_models, pair_models)   # 21 total

"OLS fit on subset columns of Phi; returns (coeffs, mse)."
function ols_fit(Phi, y, col_idx)
    Phi_sub = Phi[:, col_idx]
    coeffs  = Phi_sub \ y
    mse     = mean((y .- Phi_sub * coeffs) .^ 2)
    return coeffs, mse
end

"Return (best_col_idx, coeffs, mse) by exhaustive search over all_models."
function exhaustive_sindy(Phi, y, all_models)
    best_idx    = all_models[1]
    best_coeffs, best_mse = ols_fit(Phi, y, best_idx)
    for idx in all_models[2:end]
        c, m = ols_fit(Phi, y, idx)
        if m < best_mse
            best_mse    = m
            best_coeffs = c
            best_idx    = idx
        end
    end
    return best_idx, best_coeffs, best_mse
end

# ─── JLD2 path helper ─────────────────────────────────────────────────────────

jld2_path(noise_label) = joinpath("results", "ude_nonstationary_noise_$(noise_label).jld2")

# ─── Core function ─────────────────────────────────────────────────────────────

function run_ude_sindy(X_data, X_clean, t_full, breakpoints, regime_configs,
                       noise_label; rng_seed = 1111)
    N = size(X_clean, 2)

    # Fresh NN params
    ps0, _ = Lux.setup(StableRNG(rng_seed), nn)
    p0     = ComponentArray(ps0)
    println("NN parameters: $(length(p0))")

    # ── Training ──────────────────────────────────────────────────────────────
    losses = Float64[]
    iter   = Ref(0)

    callback = function (state, l)
        isinf(l) && return false
        iter[] += 1
        push!(losses, l)
        iter[] % 100 == 0 && println("  iter $(iter[])  loss = $(round(l, sigdigits = 5))")
        return false
    end

    extra  = (t_full, X_data)
    opt_fn = Optimization.OptimizationFunction((p, e) -> ms_loss(p, e), Optimization.AutoZygote())

    println("\n── Phase 1: Adam(0.02) × 5000 ──")
    prob1 = Optimization.OptimizationProblem(opt_fn, p0, extra)
    res1  = Optimization.solve(prob1, OptimizationOptimisers.Adam(0.02);
                                maxiters = 5000, callback = callback)

    println("\n── Phase 2: Adam(0.005) × 5000 ──")
    iter[] = 0
    prob2  = Optimization.OptimizationProblem(opt_fn, res1.u, extra)
    res2   = Optimization.solve(prob2, OptimizationOptimisers.Adam(0.005);
                                 maxiters = 5000, callback = callback)

    println("\n── Phase 3: BFGS × 2000 ──")
    p_trained = res2.u
    iter[]    = 0
    try
        prob3     = Optimization.OptimizationProblem(opt_fn, res2.u, extra)
        res3      = Optimization.solve(prob3,
                        OptimizationOptimJL.BFGS(linesearch = LineSearches.BackTracking());
                        maxiters = 2000, callback = callback)
        p_trained = res3.u
        println("BFGS converged: final loss = $(round(ms_loss(p_trained, extra), sigdigits = 5))")
    catch e
        println("BFGS failed ($e) — keeping Phase 2 result")
    end

    println("Final training loss: $(round(ms_loss(p_trained, extra), sigdigits = 5))")

    # ── Full-trajectory UDE rollout ───────────────────────────────────────────

    Xhat_ude = let
        out = nothing
        try
            prob = ODEProblem(ude!, X_clean[:, 1], (t_full[1], t_full[end]), p_trained)
            sol  = solve(prob, Tsit5(); abstol = 1e-6, reltol = 1e-6,
                         saveat = t_full, maxiters = 50_000)
            sol.retcode == ReturnCode.Success && length(sol.t) == N && (out = Array(sol))
        catch
        end

        if isnothing(out)
            println("Full rollout failed — concatenating window segments")
            segs = Vector{Matrix{Float64}}()
            for i in 1:(length(WIN_IDXS) - 1)
                t_s  = t_full[WIN_IDXS[i]:WIN_IDXS[i + 1]]
                prob = ODEProblem(ude!, X_clean[:, WIN_IDXS[i]], (t_s[1], t_s[end]), p_trained)
                sol  = solve(prob, Tsit5(); abstol = 1e-6, reltol = 1e-6,
                             saveat = t_s, maxiters = 50_000)
                push!(segs, Array(sol))
            end
            out = segs[1]
            for k in 2:length(segs)
                out = hcat(out, segs[k][:, 2:end])
            end
        end
        out
    end

    # ── NN output extraction ──────────────────────────────────────────────────

    nn_at_true = zeros(2, N)
    for j in 1:N
        out_j, _          = nn(X_clean[:, j], p_trained, nn_st)
        nn_at_true[:, j]  = out_j
    end

    n_ude     = size(Xhat_ude, 2)
    nn_at_ude = zeros(2, n_ude)
    for j in 1:n_ude
        out_j, _        = nn(Xhat_ude[:, j], p_trained, nn_st)
        nn_at_ude[:, j] = out_j
    end

    println("\nNN outputs at true states:")
    println("  NN1: mean=$(round(mean(nn_at_true[1, :]), sigdigits=4))  std=$(round(std(nn_at_true[1, :]), sigdigits=4))")
    println("  NN2: mean=$(round(mean(nn_at_true[2, :]), sigdigits=4))  std=$(round(std(nn_at_true[2, :]), sigdigits=4))")
    println("NN outputs at UDE-predicted states:")
    println("  NN1: mean=$(round(mean(nn_at_ude[1, :]), sigdigits=4))  std=$(round(std(nn_at_ude[1, :]), sigdigits=4))")
    println("  NN2: mean=$(round(mean(nn_at_ude[2, :]), sigdigits=4))  std=$(round(std(nn_at_ude[2, :]), sigdigits=4))")

    # ── Targeted exhaustive SINDy on NN outputs ──────────────────────────────

    println("\n── Targeted SINDy on NN outputs ──")

    sindy_results = []
    for (k, label) in enumerate(["NN1 (correction to dv/dt)", "NN2 (correction to du/dt)"])
        y            = nn_at_true[k, :]
        idx, coeffs, mse = exhaustive_sindy(Phi, y, all_models)
        names        = feat_names[idx]
        push!(sindy_results, (idx = idx, coeffs = coeffs, mse = mse, names = names))

        println("\n  $label")
        println("  Winning model: $(join(["$(round(coeffs[i], sigdigits=4)) * $(names[i])" for i in eachindex(names)], " + "))")
        println("  MSE = $(round(mse, sigdigits=4))")
    end

    # ── Comparison vs true interaction terms ──────────────────────────────────
    true_vu_coeff = Dict(
        1 => (r1 = -0.10, r3 = -0.12),   # NN1
        2 => (r1 =  0.10, r3 =  0.10),   # NN2
    )
    VU_IDX = findfirst(==("v*u"), feat_names)

    println("\n  True interaction (v*u) coefficients:")
    for k in 1:2
        label = k == 1 ? "NN1" : "NN2"
        res   = sindy_results[k]
        pos   = findfirst(==(VU_IDX), res.idx)
        if !isnothing(pos)
            c_fit = res.coeffs[pos]
            c_r1  = true_vu_coeff[k].r1
            c_r3  = true_vu_coeff[k].r3
            println("  $label: fitted=$(round(c_fit, sigdigits=4))" *
                    "  rel_err_R1=$(round(abs(c_fit - c_r1)/abs(c_r1), sigdigits=3))" *
                    "  rel_err_R3=$(round(abs(c_fit - c_r3)/abs(c_r3), sigdigits=3))")
        else
            println("  $label: v*u not selected by SINDy")
        end
    end

    # ── Per-regime derivative residuals ───────────────────────────────────────

    masks   = get_regime_masks(t_full, breakpoints)
    dX_true = evaluate_true_derivatives(t_full, X_clean, regime_configs, breakpoints)

    res1_sindy, res2_sindy = sindy_results
    dv_sindy = ETA1_FIXED .* v_all .+ Phi[:, res1_sindy.idx] * res1_sindy.coeffs
    du_sindy = .-ETA2_FIXED .* u_all .+ Phi[:, res2_sindy.idx] * res2_sindy.coeffs
    dX_sindy = vcat(dv_sindy', du_sindy')   # 2 × N

    regime_labels_deriv = ["R1 [0,200)", "R2 [200,290)", "Tr [290,300)", "R3 [300,500]"]
    println("\n  Per-regime derivative RMSE (true vs UDE+SINDy reconstructed):")
    for i in 1:4
        m = masks[i]
        !any(m) && continue
        rmse_i = sqrt(mean((dX_true[:, m] .- dX_sindy[:, m]) .^ 2))
        println("  $(regime_labels_deriv[i]):  RMSE = $(round(rmse_i, sigdigits=4))")
    end

    # ── Save ──────────────────────────────────────────────────────────────────

    save_path = jld2_path(noise_label)
    ensure_dir(dirname(save_path))
    jldsave(save_path;
        p_trained, X_clean, Xhat_ude,
        nn_at_true, nn_at_ude, losses,
        sindy_results,
        t          = t_full,
        breakpoints,
        dX_true,
        dX_sindy,
        masks,
        noise_label)
    println("Results saved → $save_path")

    # Also save as _latest for noise=0 (backward compat with viz script)
    if noise_label == "0.0"
        latest = joinpath("results", "ude_nonstationary_latest.jld2")
        cp(save_path, latest; force = true)
        println("  (also copied → $latest)")
    end

    return (; p_trained, losses, sindy_results, dX_sindy, dX_true, masks)
end

# ─── Copy existing noise=0 result ─────────────────────────────────────────────

let src = joinpath("results", "ude_nonstationary_latest.jld2"),
    dst = jld2_path("0.0")
    if isfile(src) && !isfile(dst)
        ensure_dir(dirname(dst))
        cp(src, dst)
        println("Copied existing noise=0 result: $src → $dst")
    end
end

# ─── Noise sweep ──────────────────────────────────────────────────────────────

noise_levels = [0.0, 1e-3, 1e-2, 5e-2]

for noise in noise_levels
    label = string(noise)
    path  = jld2_path(label)

    if isfile(path)
        println("\n═══ noise=$label — SKIPPED ($(path) exists) ═══")
        continue
    end

    println("\n═══════════════════════════════════════════════════════════")
    println("  noise = $label")
    println("═══════════════════════════════════════════════════════════")

    X_data = if noise == 0.0
        X_clean
    else
        add_relative_noise(X_clean, StableRNG(1); noise_magnitude = noise)
    end

    run_ude_sindy(X_data, X_clean, t_full, breakpoints, regime_configs, label)
end

# ─── Combined summary ─────────────────────────────────────────────────────────

println("\n\n══════════════════════════════════════════════════════════════")
println("  COMBINED SUMMARY — all noise levels")
println("══════════════════════════════════════════════════════════════")

VU_IDX_SUMMARY = findfirst(==("v*u"), feat_names)

for noise in noise_levels
    label = string(noise)
    path  = jld2_path(label)
    if !isfile(path)
        println("\n  noise=$label — NOT FOUND")
        continue
    end

    data = load(path)
    sr   = data["sindy_results"]
    ls   = data["losses"]

    # v*u coefficients
    vu_coeffs = Float64[]
    for k in 1:2
        pos = findfirst(==(VU_IDX_SUMMARY), sr[k].idx)
        push!(vu_coeffs, isnothing(pos) ? NaN : sr[k].coeffs[pos])
    end

    # Global derivative RMSE
    dX_t = data["dX_true"]
    dX_s = data["dX_sindy"]
    global_rmse = sqrt(mean((dX_t .- dX_s) .^ 2))

    println("\n  noise=$label:")
    println("    final loss   = $(round(ls[end], sigdigits=5))")
    println("    v*u coeff    = eq1: $(round(vu_coeffs[1], sigdigits=4)),  eq2: $(round(vu_coeffs[2], sigdigits=4))")
    println("    true v*u     = eq1: -0.10/-0.12 (R1/R3),  eq2: +0.10")
    println("    deriv RMSE   = $(round(global_rmse, sigdigits=4))")
    println("    iters        = $(length(ls))")
end

println("\nDone.")

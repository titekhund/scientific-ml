# experiments/exp_ude_stationary.jl
# UDE+SINDy on stationary Goodwin data: noise sweep
# Training code ported directly from the working Colab notebook.
# Primary discovery: fit_sindy_from_nn (ADMM + STLSQ)
# Supplementary:     exhaustive OLS
#
# Usage:  julia --project=. experiments/exp_ude_stationary.jl

include(joinpath(@__DIR__, "..", "src", "ScientificML.jl"))
using .ScientificML

using OrdinaryDiffEq, SciMLSensitivity
using Optimization, OptimizationOptimisers, OptimizationOptimJL, LineSearches
using ComponentArrays, Lux, Zygote
using StableRNGs, LinearAlgebra, Statistics, JLD2, ForwardDiff

# ─── Data (exact Colab ODE and parameters) ───────────────────────────────────
# Colab convention: p_ = [α, β, γ, δ]
#   dv = α*v − β*u*v     →  ETA1 = α = 1.399,  NN1 learns −β*vu = −2.239*vu
#   du = γ*v*u − δ*u     →  ETA2 = δ = 2.43,   NN2 learns +γ*vu = +2.57*vu

const p_ = [1.399, 2.239, 2.57, 2.43]    # [α, β, γ, δ]

function lotka!(du, u, p, t)
    α, β, γ, δ = p
    du[1] = α * u[1] - β * u[2] * u[1]
    du[2] = γ * u[1] * u[2] - δ * u[2]
end

const tspan = (0.0, 10.0)
const u0    = [0.92, 0.65]

prob = ODEProblem(lotka!, u0, tspan, p_)
solution = solve(prob, Vern7(), abstol = 1e-12, reltol = 1e-12, saveat = 0.25)

const X = Array(solution)
const t = solution.t
const N = size(X, 2)

println("Data: $(size(X)) — $N points over [$(t[1]), $(t[end])]")
println("Colab params: α=$(p_[1]), β=$(p_[2]), γ=$(p_[3]), δ=$(p_[4])")

# ─── Neural network (const, defined once outside noise loop) ─────────────────

rbf(x) = exp.(-(x .^ 2))

const U = Lux.Chain(
    Lux.Dense(2, 5, rbf),
    Lux.Dense(5, 5, rbf),
    Lux.Dense(5, 5, rbf),
    Lux.Dense(5, 2),
)

# Initial setup (just to get _st; fresh params created per noise level)
let
    _p, _s = Lux.setup(StableRNGs.StableRNG(1111), U)
    global const _st = _s
end

# ─── UDE dynamics (exact Colab pattern) ──────────────────────────────────────

function ude_dynamics!(du, u, p, t, p_true)
    û = U(u, p, _st)[1]
    du[1] = p_true[1] * u[1] + û[1]
    du[2] = -p_true[4] * u[2] + û[2]
end

nn_dynamics!(du, u, p, t) = ude_dynamics!(du, u, p, t, p_)

# ─── Feature library for exhaustive OLS ──────────────────────────────────────

v_all = X[1, :]
u_all = X[2, :]

candidates = [
    ("1",   ones(N)),
    ("v",   v_all),
    ("u",   u_all),
    ("v*u", v_all .* u_all),
    ("v^2", v_all .^ 2),
    ("u^2", u_all .^ 2),
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

# ─── JLD2 path helper ────────────────────────────────────────────────────────

jld2_path(noise_label) = joinpath("results", "ude_stationary", "ude_stationary_noise_$(noise_label).jld2")

# ─── Core function ────────────────────────────────────────────────────────────

function run_ude_sindy_stationary(Xₙ, noise_label; rng_seed = 1111)

    # ── Fresh NN params ───────────────────────────────────────────────────────
    rng = StableRNGs.StableRNG(rng_seed)
    p_init, _ = Lux.setup(rng, U)
    println("NN parameters: $(length(ComponentVector{Float64}(p_init)))")

    # ── Prediction (exact Colab: remake pattern, Vern7, QuadratureAdjoint) ────
    prob_nn = ODEProblem(nn_dynamics!, Xₙ[:, 1], tspan, p_init)

    function predict(θ, X_ic = Xₙ[:, 1], T = t)
        _prob = remake(prob_nn, u0 = X_ic, tspan = (T[1], T[end]), p = θ)
        Array(solve(_prob, Vern7(), saveat = T,
            abstol = 1e-6, reltol = 1e-6,
            sensealg = ForwardDiffSensitivity()))
    end

    function loss(θ)
        X̂ = predict(θ)
        Statistics.mean(abs2, Xₙ .- X̂)
    end

    # ── Training (exact Colab: Adam 0.05 × 5000, BFGS × 2000) ────────────────
    losses = Float64[]

    callback = function (state, l)
        push!(losses, l)
        if length(losses) % 50 == 0
            println("  iter $(length(losses))  loss = $(round(l, sigdigits = 5))")
        end
        return false
    end

    adtype  = Optimization.AutoZygote()
    optf    = Optimization.OptimizationFunction((x, p) -> loss(x), adtype)
    optprob = Optimization.OptimizationProblem(optf, ComponentVector{Float64}(p_init))

    println("\n── Phase 1: Adam(0.05) x 5000 ──")
    res1 = Optimization.solve(optprob, OptimizationOptimisers.Adam(0.05);
                              callback = callback, maxiters = 5000)

    println("\n── Phase 2: BFGS x 2000 ──")
    p_trained = res1.u
    try
        optprob2  = Optimization.OptimizationProblem(optf, res1.u)
        res2      = Optimization.solve(optprob2,
                        OptimizationOptimJL.BFGS(linesearch = LineSearches.BackTracking());
                        callback = callback, maxiters = 2000)
        p_trained = res2.u
    catch e
        println("BFGS failed ($e) — keeping Phase 1 result")
    end

    final_loss = loss(p_trained)
    println("Final training loss: $(round(final_loss, sigdigits = 5))")

    # ── Full-trajectory UDE rollout ───────────────────────────────────────────

    Xhat_ude = try
        predict(p_trained, X[:, 1], t)
    catch
        println("Full rollout failed — filling with NaN")
        fill(NaN, 2, N)
    end

    # ── NN output extraction ──────────────────────────────────────────────────

    nn_at_true = zeros(2, N)
    for j in 1:N
        nn_at_true[:, j] = U(X[:, j], p_trained, _st)[1]
    end

    nn_at_ude = zeros(2, size(Xhat_ude, 2))
    for j in 1:size(Xhat_ude, 2)
        nn_at_ude[:, j] = U(Xhat_ude[:, j], p_trained, _st)[1]
    end

    println("\nNN outputs at true states:")
    println("  NN1: mean=$(round(mean(nn_at_true[1, :]), sigdigits=4))  std=$(round(std(nn_at_true[1, :]), sigdigits=4))")
    println("  NN2: mean=$(round(mean(nn_at_true[2, :]), sigdigits=4))  std=$(round(std(nn_at_true[2, :]), sigdigits=4))")

    # ── Primary: Sparse regression on NN outputs (ADMM + STLSQ) ──────────────

    println("\n── Primary: Sparse regression on NN outputs ──")

    sparse_reg_results = Dict{String, Any}()
    for meth in [:ADMM, :STLSQ, :SR3]
        label_m = string(meth)
        println("\n  Method: $label_m")
        try
            res_sr, sys_sr, params_sr = fit_sindy_from_nn(
                X, nn_at_true;
                polyorder = 2, method = meth,
                batchsize = 30,
                rng = ScientificML.StableRNGs.StableRNG(42))

            eqs = ScientificML.ModelingToolkit.equations(sys_sr)
            println("  Discovered equations ($label_m):")
            for (i, eq) in enumerate(eqs)
                println("    eq$i: ", eq)
            end
            sparse_reg_results[label_m] = (res = res_sr, system = sys_sr, params = params_sr)
        catch e
            println("  $label_m failed: ", e)
            sparse_reg_results[label_m] = nothing
        end
    end

    # ── Supplementary: Exhaustive OLS on NN outputs ───────────────────────────

    println("\n── Supplementary: Exhaustive OLS on NN outputs ──")

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

    # ── True interaction terms comparison ─────────────────────────────────────
    # UDE: dv = α*v + NN₁,  du = −δ*u + NN₂
    # True: dv = α*v − β*u*v,  du = γ*v*u − δ*u
    # NN1 should learn −β*vu = −2.239*vu, NN2 should learn +γ*vu = +2.57*vu
    VU_IDX = findfirst(==("v*u"), feat_names)

    println("\n  True interaction (v*u) coefficients:")
    println("  Expected: NN1 = $(-p_[2]),  NN2 = $(p_[3])")
    for k in 1:2
        label = k == 1 ? "NN1" : "NN2"
        res   = sindy_results[k]
        pos   = findfirst(==(VU_IDX), res.idx)
        if !isnothing(pos)
            c_fit  = res.coeffs[pos]
            c_true = k == 1 ? -p_[2] : p_[3]
            println("  $label: fitted=$(round(c_fit, sigdigits=4))" *
                    "  rel_err=$(round(abs(c_fit - c_true)/abs(c_true), sigdigits=3))")
        else
            println("  $label: v*u not selected by exhaustive OLS")
        end
    end

    # ── Derivative residuals ──────────────────────────────────────────────────

    # True derivatives via the Colab ODE
    dX_true = zeros(2, N)
    for j in 1:N
        lotka!(view(dX_true, :, j), X[:, j], p_, t[j])
    end

    # Exhaustive OLS reconstructed derivatives
    res1_sindy, res2_sindy = sindy_results
    dv_sindy = p_[1] .* v_all .+ Phi[:, res1_sindy.idx] * res1_sindy.coeffs
    du_sindy = .-p_[4] .* u_all .+ Phi[:, res2_sindy.idx] * res2_sindy.coeffs
    dX_sindy = vcat(dv_sindy', du_sindy')   # 2 × N

    global_rmse_ols = sqrt(mean((dX_true .- dX_sindy) .^ 2))
    println("\n  Global derivative RMSE (exhaustive OLS): $(round(global_rmse_ols, sigdigits=4))")

    # Sparse regression derivative RMSE
    for meth in ["ADMM", "STLSQ", "SR3"]
        sr = get(sparse_reg_results, meth, nothing)
        sr === nothing && continue
        try
            dX_sr = eval_sindy_at_states(sr.res, t, X)
            dX_sr[1, :] .+= p_[1] .* X[1, :]
            dX_sr[2, :] .-= p_[4] .* X[2, :]
            rmse_sr = sqrt(mean((dX_true .- dX_sr) .^ 2))
            println("  Global derivative RMSE ($meth): $(round(rmse_sr, sigdigits=4))")
        catch e
            println("  $meth derivative evaluation failed: ", e)
        end
    end

    # ── Build sparse regression equation strings + derivative predictions ─────

    sparse_eq_strings = Dict{String, Any}()
    sparse_dX = Dict{String, Any}()
    for meth in ["ADMM", "STLSQ", "SR3"]
        sr = get(sparse_reg_results, meth, nothing)
        if sr !== nothing
            try
                eqs = ScientificML.ModelingToolkit.equations(sr.system)
                sparse_eq_strings[meth] = [string(eq.rhs) for eq in eqs]
            catch
                sparse_eq_strings[meth] = nothing
            end
            try
                dX_sr = eval_sindy_at_states(sr.res, t, X)
                dX_sr[1, :] .+= p_[1] .* X[1, :]
                dX_sr[2, :] .-= p_[4] .* X[2, :]
                sparse_dX[meth] = dX_sr
            catch
                sparse_dX[meth] = nothing
            end
        else
            sparse_eq_strings[meth] = nothing
            sparse_dX[meth] = nothing
        end
    end

    # ── Save ──────────────────────────────────────────────────────────────────

    save_path = jld2_path(noise_label)
    ensure_dir(dirname(save_path))
    jldsave(save_path;
        p_trained,
        X_clean = X, Xhat_ude,
        nn_at_true, nn_at_ude = nn_at_ude, losses,
        sindy_results,
        sparse_eq_strings,
        sparse_dX,
        t        = t,
        dX_true,
        dX_sindy,
        noise_label)
    println("Results saved -> $save_path")

    return (; p_trained, losses, sindy_results, sparse_reg_results, dX_sindy, dX_true)
end

# ─── Noise sweep ──────────────────────────────────────────────────────────────

noise_levels = [0.0, 1e-3, 1e-2, 5e-2]

x̄ = Statistics.mean(X, dims = 2)

for noise in noise_levels
    label = string(noise)
    path  = jld2_path(label)

    if isfile(path)
        existing = load(path)
        if haskey(existing, "sparse_dX")
            println("\n=== noise=$label — SKIPPED ($(path) exists with sparse results) ===")
            continue
        else
            println("\n=== noise=$label — RE-RUNNING ($(path) missing sparse results) ===")
        end
    end

    println("\n===============================================================")
    println("  noise = $label")
    println("===============================================================")

    # Colab noise pattern: Xₙ = X .+ (noise_magnitude * x̄) .* randn(rng, ...)
    rng_noise = StableRNGs.StableRNG(1)
    Xₙ = if noise == 0.0
        copy(X)
    else
        X .+ (noise * x̄) .* randn(rng_noise, eltype(X), size(X))
    end

    run_ude_sindy_stationary(Xₙ, label)
end

# ─── Combined summary ────────────────────────────────────────────────────────

println("\n\n==============================================================")
println("  COMBINED SUMMARY — all noise levels")
println("==============================================================")

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

    # v*u coefficient from exhaustive OLS
    VU_IDX_S = findfirst(==("v*u"), feat_names)
    vu_coeffs = Float64[]
    for k in 1:2
        pos = findfirst(==(VU_IDX_S), sr[k].idx)
        push!(vu_coeffs, isnothing(pos) ? NaN : sr[k].coeffs[pos])
    end

    # Global derivative RMSE (exhaustive OLS)
    dX_t = data["dX_true"]
    dX_s = data["dX_sindy"]
    global_rmse = sqrt(mean((dX_t .- dX_s) .^ 2))

    # Sparse regression equations
    sp_eqs = get(data, "sparse_eq_strings", nothing)

    println("\n  noise=$label:")
    println("    final loss       = $(round(ls[end], sigdigits=5))")
    println("    v*u coeff (OLS)  = eq1: $(round(vu_coeffs[1], sigdigits=4)),  eq2: $(round(vu_coeffs[2], sigdigits=4))")
    println("    true v*u         = eq1: $(-p_[2]),  eq2: $(p_[3])")
    println("    deriv RMSE (OLS) = $(round(global_rmse, sigdigits=4))")
    println("    iters            = $(length(ls))")
    if sp_eqs !== nothing
        for meth in ["ADMM", "STLSQ", "SR3"]
            eqs = get(sp_eqs, meth, nothing)
            if eqs !== nothing
                println("    $meth equations:")
                for (i, eq) in enumerate(eqs)
                    println("      eq$i: $eq")
                end
            end
        end
    end
end

println("\nDone.")

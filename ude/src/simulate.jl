# src/simulate.jl
# Simulation + dataset generation utilities

export simulate_clean, add_relative_noise, make_dataset

"Simulate an ODEProblem and return (t, X) where X is states x time."
function simulate_clean(prob;
        solver=Vern7(),
        abstol=1e-12,
        reltol=1e-12,
        saveat=0.25)

    sol = solve(prob, solver; abstol=abstol, reltol=reltol, saveat=saveat)
    t = sol.t
    X = Array(sol)  # size: (n_states, n_time)
    return t, X
end

"Add Gaussian noise scaled by per-state mean magnitude (relative noise)."
function add_relative_noise(X, rng; noise_magnitude=1e-3)
    x̄ = Statistics.mean(X, dims=2)                 # (n_states, 1)
    ϵ  = randn(rng, eltype(X), size(X))             # same size as X
    Xn = X .+ (noise_magnitude .* x̄) .* ϵ
    return Xn
end

"""
Make a dataset from an ODEProblem:
- simulate clean data
- optionally keep only first N points
- optionally add relative noise

Returns (t, X, Xn) where Xn may equal X if noise_magnitude == 0.
"""
function make_dataset(prob, rng;
        saveat=0.25,
        N::Union{Nothing,Int}=nothing,
        noise_magnitude=1e-3,
        solver=Vern7(),
        abstol=1e-12,
        reltol=1e-12)

    t, X = simulate_clean(prob; solver=solver, abstol=abstol, reltol=reltol, saveat=saveat)

    if N !== nothing
        N = min(N, length(t))
        t = t[1:N]
        X = X[:, 1:N]
    end

    Xn = noise_magnitude == 0 ? X : add_relative_noise(X, rng; noise_magnitude=noise_magnitude)
    return t, X, Xn
end

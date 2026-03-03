# src/metrics.jl
# Trajectory-level metrics for state matrices shaped (n_states, n_time)

export mse, rmse, nrmse, mae, r2score, trajectory_metrics

using Statistics
using LinearAlgebra

"Mean squared error."
mse(y, ŷ) = mean(abs2, y .- ŷ)

"Root mean squared error."
rmse(y, ŷ) = sqrt(mse(y, ŷ))

"Mean absolute error."
mae(y, ŷ) = mean(abs, y .- ŷ)

"Normalized RMSE (default normalization uses std(y))."
function nrmse(y, ŷ; norm=:std, ϵ=1e-12)
    denom = if norm === :std
        std(vec(y))
    elseif norm === :range
        maximum(y) - minimum(y)
    elseif norm === :l2
        norm(vec(y))
    else
        throw(ArgumentError("norm must be :std, :range, or :l2"))
    end
    denom = max(denom, ϵ)
    return rmse(y, ŷ) / denom
end

"R² score (coefficient of determination)."
function r2score(y, ŷ; ϵ=1e-12)
    yv = vec(y); ŷv = vec(ŷ)
    ss_res = sum(abs2, yv .- ŷv)
    ss_tot = sum(abs2, yv .- mean(yv))
    ss_tot = max(ss_tot, ϵ)
    return 1 - ss_res / ss_tot
end

"""
Trajectory metrics for matrices shaped (n_states, n_time).
Returns a NamedTuple with aggregate + per-state RMSE/NRMSE.
"""
function trajectory_metrics(X, Xhat; nrmse_norm=:std)
    @assert size(X) == size(Xhat) "X and Xhat must have same size"

    per_state_rmse  = [rmse(view(X,i,:), view(Xhat,i,:)) for i in axes(X,1)]
    per_state_nrmse = [nrmse(view(X,i,:), view(Xhat,i,:); norm=nrmse_norm) for i in axes(X,1)]

    return (
        rmse = rmse(X, Xhat),
        mae  = mae(X, Xhat),
        nrmse = nrmse(X, Xhat; norm=nrmse_norm),
        r2   = r2score(X, Xhat),
        per_state_rmse = per_state_rmse,
        per_state_nrmse = per_state_nrmse
    )
end

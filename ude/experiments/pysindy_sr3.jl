# experiments/pysindy_sr3.jl
# PySINDy SR3 (L0) sparse regression via PyCall.
# Matches the exact settings validated on the stationary Colab experiment,
# with automatic λ sweep for robustness across coefficient scales.
#
# Usage: include("pysindy_sr3.jl") from another script.

using PyCall

# ─── One-time pip install (quiet, no-op if already installed) ─────────────────

const _pysindy_ready = Ref(false)

function _ensure_pysindy()
    _pysindy_ready[] && return
    try
        pyimport("pysindy")
    catch
        println("Installing numpy + pysindy via pip...")
        pip = PyCall.pyprogramname
        try
            run(`$pip -m pip install numpy pysindy --quiet`)
        catch
            # PEP 668: externally-managed Python — force install as last resort
            run(`$pip -m pip install numpy pysindy --quiet --break-system-packages`)
        end
    end
    _pysindy_ready[] = true
end

# ─── Helper: fit one SR3 model at a given λ ──────────────────────────────────

function _fit_one_lambda(ps, SR3_py, library, X_py, xdot_py, Theta, n_feat,
                         lam::Float64, max_iter::Int)
    optimizer = SR3_py(
        regularizer       = "L0",
        reg_weight_lam    = lam,
        max_iter          = max_iter,
        normalize_columns = false,
    )
    model = ps.SINDy(
        optimizer       = optimizer,
        feature_library = library,
    )
    try
        model.fit(X_py, t = 1.0, x_dot = xdot_py, feature_names = ["x", "y"])
    catch
        model.fit(X_py, t = 1.0, x_dot = xdot_py)
    end

    coef_matrix = convert(Matrix{Float64}, model.coefficients())  # 2 × n_feat

    # Per-equation active term count and MSE
    n_active = Int[]
    mse_total = 0.0
    N = size(Theta, 1)
    for k in 1:2
        coeffs = coef_matrix[k, :]
        push!(n_active, count(c -> abs(c) > 0.01, coeffs))
        pred = Theta * coeffs
        mse_total += sum(abs2, xdot_py[:, k] .- pred) / N
    end
    mse_total /= 2  # average over both equations

    return (; coef_matrix, n_active, mse = mse_total, model)
end

# ─── Main function ────────────────────────────────────────────────────────────

"""
    run_pysindy_sr3(X_smooth, U_nn; degree=2, lambdas=[...], max_iter=50000)

Fit PySINDy SR3 (L0, normalize_columns=False) on NN outputs with automatic λ sweep.

Selection criterion:
1. Both equations must have ≥1 active term
2. Among those, pick the sparsest (fewest total active terms)
3. Among ties, pick lowest MSE
4. Fallback: lowest MSE overall if no λ activates both equations

Returns a Dict with:
- `feature_names`, `coeffs_f1`, `coeffs_f2`, `active_f1`, `active_f2`
- `pred_f1`, `pred_f2`: library prediction at each time point
- `selected_lambda`: the chosen regularization weight
"""
function run_pysindy_sr3(X_smooth::Matrix{Float64}, U_nn::Matrix{Float64};
                         degree::Int = 2,
                         lambdas::Vector{Float64} = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0],
                         max_iter::Int = 50000)

    _ensure_pysindy()

    N = size(X_smooth, 2)
    @assert size(U_nn, 2) == N "X_smooth and U_nn must have the same number of columns"
    @assert size(X_smooth, 1) == 2 && size(U_nn, 1) == 2 "Expected 2-row matrices"

    np = pyimport("numpy")
    ps = pyimport("pysindy")
    SR3_py = pyimport("pysindy.optimizers").SR3

    # Transpose to N×2 for PySINDy (rows = samples, cols = features)
    X_py    = np.array(collect(X_smooth'))    # N × 2
    xdot_py = np.array(collect(U_nn'))        # N × 2  (NN outputs = learned derivatives)

    # Build polynomial library (degree=2, no bias/constant term)
    library = ps.PolynomialLibrary(degree = degree, include_bias = false)

    # Fit library once to get Theta matrix and feature names
    library.fit(X_py)
    Theta = convert(Matrix{Float64}, library.transform(X_py))
    n_feat = size(Theta, 2)

    # ── λ sweep ───────────────────────────────────────────────────────────────

    println("\n  λ sweep results:")

    sweep = []
    for lam in lambdas
        r = _fit_one_lambda(ps, SR3_py, library, X_py, xdot_py, Theta, n_feat, lam, max_iter)
        push!(sweep, (lam = lam, result = r))
        Printf_lam = rpad("λ=$(lam):", 12)
        println("  $Printf_lam f1 terms=$(r.n_active[1]), f2 terms=$(r.n_active[2]), MSE=$(round(r.mse, sigdigits=4))")
    end

    # ── Selection ─────────────────────────────────────────────────────────────
    # 1. Both equations must have ≥1 active term
    valid = [(s.lam, s.result) for s in sweep
             if s.result.n_active[1] >= 1 && s.result.n_active[2] >= 1]

    best_lam, best_result = if !isempty(valid)
        # 2. Sparsest total active terms, then 3. lowest MSE
        sort!(valid, by = x -> (sum(x[2].n_active), x[2].mse))
        valid[1]
    else
        # Fallback: lowest MSE overall
        fallback = sort([(s.lam, s.result) for s in sweep], by = x -> x[2].mse)
        fallback[1]
    end

    # Mark selected in the printed table
    println("  Selected λ=$(best_lam)")

    # ── Extract results from winner ───────────────────────────────────────────

    feat_names_py = best_result.model.get_feature_names()
    feat_names = String[string(f) for f in feat_names_py]

    coef_matrix = best_result.coef_matrix

    println("\n============================================================")
    println("  SR3 (L0, normalize_columns=False)")
    println("============================================================")

    results = Dict{String, Any}()
    results["feature_names"] = feat_names
    results["selected_lambda"] = best_lam

    for (k, label) in enumerate(["f1", "f2"])
        coeffs = coef_matrix[k, :]

        # Predictions
        pred = Theta * coeffs

        # Active terms (|coeff| > 0.01)
        active = Tuple{String, Float64}[(feat_names[i], coeffs[i])
                                         for i in 1:n_feat if abs(coeffs[i]) > 0.01]

        # Format active terms for printing
        active_str = join(["('$(name)', '$(c >= 0 ? "+" : "")$(round(c, sigdigits=5))')"
                           for (name, c) in active], ", ")
        println("  $label: [$active_str]")

        results["coeffs_$label"] = coeffs
        results["active_$label"] = active
        results["pred_$label"]   = pred
    end

    return results
end

# ─── Direct mode: PySINDy handles differentiation internally ─────────────────

"""
Helper: fit one SR3 model at a given λ in direct mode (no x_dot).
PySINDy computes derivatives via SmoothedFiniteDifference.
"""
function _fit_one_lambda_direct(ps, SR3_py, library, X_py, dt_val, Theta, n_feat,
                                lam::Float64, max_iter::Int)
    optimizer = SR3_py(
        regularizer       = "L0",
        reg_weight_lam    = lam,
        max_iter          = max_iter,
        normalize_columns = false,
    )
    model = ps.SINDy(
        optimizer       = optimizer,
        feature_library = library,
    )
    try
        model.fit(X_py, t = dt_val, feature_names = ["x", "y"])
    catch
        model.fit(X_py, t = dt_val)
    end

    coef_matrix = convert(Matrix{Float64}, model.coefficients())  # 2 × n_feat

    # Per-equation active term count and MSE (use model predictions)
    n_active = Int[]
    mse_total = 0.0
    N = size(Theta, 1)
    xdot_pred = convert(Matrix{Float64}, model.predict(X_py))  # N × 2
    for k in 1:2
        coeffs = coef_matrix[k, :]
        push!(n_active, count(c -> abs(c) > 0.01, coeffs))
        pred = Theta * coeffs
        mse_total += sum(abs2, xdot_pred[:, k] .- pred) / N
    end
    mse_total /= 2

    return (; coef_matrix, n_active, mse = mse_total, model)
end

"""
    run_pysindy_sr3_direct(X_data, t_vec; degree=2, lambdas=[...], max_iter=50000)

Fit PySINDy SR3 (L0) directly on state data — PySINDy computes derivatives
internally via SmoothedFiniteDifference.

Same selection criterion and return format as `run_pysindy_sr3`.
"""
function run_pysindy_sr3_direct(X_data::Matrix{Float64}, t_vec::Vector{Float64};
                                degree::Int = 2,
                                lambdas::Vector{Float64} = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0],
                                max_iter::Int = 50000)

    _ensure_pysindy()

    N = size(X_data, 2)
    @assert size(X_data, 1) == 2 "Expected 2-row matrix"
    @assert length(t_vec) == N "t_vec length must match columns of X_data"

    dt_val = Float64(t_vec[2] - t_vec[1])

    np = pyimport("numpy")
    ps = pyimport("pysindy")
    SR3_py = pyimport("pysindy.optimizers").SR3

    X_py = np.array(collect(X_data'))    # N × 2

    library = ps.PolynomialLibrary(degree = degree, include_bias = false)

    library.fit(X_py)
    Theta = convert(Matrix{Float64}, library.transform(X_py))
    n_feat = size(Theta, 2)

    # ── λ sweep ───────────────────────────────────────────────────────────────
    println("\n  λ sweep results (direct mode, PySINDy differentiation):")

    sweep = []
    for lam in lambdas
        r = _fit_one_lambda_direct(ps, SR3_py, library, X_py, dt_val, Theta, n_feat, lam, max_iter)
        push!(sweep, (lam = lam, result = r))
        Printf_lam = rpad("λ=$(lam):", 12)
        println("  $Printf_lam f1 terms=$(r.n_active[1]), f2 terms=$(r.n_active[2]), MSE=$(round(r.mse, sigdigits=4))")
    end

    # ── Selection (same logic as run_pysindy_sr3) ─────────────────────────────
    valid = [(s.lam, s.result) for s in sweep
             if s.result.n_active[1] >= 1 && s.result.n_active[2] >= 1]

    best_lam, best_result = if !isempty(valid)
        sort!(valid, by = x -> (sum(x[2].n_active), x[2].mse))
        valid[1]
    else
        fallback = sort([(s.lam, s.result) for s in sweep], by = x -> x[2].mse)
        fallback[1]
    end

    println("  Selected λ=$(best_lam)")

    # ── Extract results ───────────────────────────────────────────────────────
    feat_names_py = best_result.model.get_feature_names()
    feat_names = String[string(f) for f in feat_names_py]

    coef_matrix = best_result.coef_matrix

    println("\n============================================================")
    println("  SR3 Direct (L0, SmoothedFiniteDifference)")
    println("============================================================")

    results = Dict{String, Any}()
    results["feature_names"] = feat_names
    results["selected_lambda"] = best_lam

    for (k, label) in enumerate(["f1", "f2"])
        coeffs = coef_matrix[k, :]

        pred = Theta * coeffs

        active = Tuple{String, Float64}[(feat_names[i], coeffs[i])
                                         for i in 1:n_feat if abs(coeffs[i]) > 0.01]

        active_str = join(["('$(name)', '$(c >= 0 ? "+" : "")$(round(c, sigdigits=5))')"
                           for (name, c) in active], ", ")
        println("  $label: [$active_str]")

        results["coeffs_$label"] = coeffs
        results["active_$label"] = active
        results["pred_$label"]   = pred
    end

    return results
end

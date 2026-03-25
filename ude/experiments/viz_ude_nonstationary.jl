# experiments/viz_ude_nonstationary.jl
# 6 diagnostic plots for Stage 2 UDE+SINDy on nonstationary Goodwin.
# Loads pre-computed results from JLD2 (run exp_ude_nonstationary.jl first).
#
# Usage:  julia --project=. experiments/viz_ude_nonstationary.jl

include(joinpath(@__DIR__, "..", "src", "ScientificML.jl"))
using .ScientificML

include(joinpath(@__DIR__, "..", "src", "simulate_nonstationary.jl"))

using JLD2, Plots, Statistics, LinearAlgebra
gr()

# ─── Load pre-computed results ────────────────────────────────────────────────

data          = load(joinpath(@__DIR__, "..", "results", "ude_nonstationary_latest.jld2"))
X_clean       = data["X_clean"]
Xhat_ude      = data["Xhat_ude"]
nn_at_true    = data["nn_at_true"]
losses        = data["losses"]
sindy_results = data["sindy_results"]
t_full        = data["t"]
breakpoints   = data["breakpoints"]
dX_true       = data["dX_true"]
dX_sindy      = data["dX_sindy"]
masks         = data["masks"]
N             = size(X_clean, 2)

# Reconstruct feature library and helpers needed by Plot 5
const ETA1_FIXED = 0.05
const ETA2_FIXED = 0.05

v_all = X_clean[1, :]
u_all = X_clean[2, :]
candidates    = [("1", ones(N)), ("v", v_all), ("u", u_all),
                 ("v*u", v_all .* u_all), ("v²", v_all .^ 2), ("u²", u_all .^ 2)]
feat_names    = [c[1] for c in candidates]
Phi           = hcat([c[2] for c in candidates]...)
single_models = [[i]    for i in 1:6]
pair_models   = [[i, j] for i in 1:6 for j in (i+1):6]
all_models    = vcat(single_models, pair_models)
VU_IDX        = findfirst(==("v*u"), feat_names)

"OLS fit on subset of Phi columns; returns (coeffs, mse)."
function ols_fit(Phi, y, col_idx)
    Phi_sub = Phi[:, col_idx]
    coeffs  = Phi_sub \ y
    return coeffs, mean((y .- Phi_sub * coeffs) .^ 2)
end

"Exhaustive search over all_models; returns (best_idx, coeffs, mse)."
function exhaustive_sindy(Phi, y, all_models)
    best_idx              = all_models[1]
    best_coeffs, best_mse = ols_fit(Phi, y, best_idx)
    for idx in all_models[2:end]
        c, m = ols_fit(Phi, y, idx)
        if m < best_mse
            best_idx, best_coeffs, best_mse = idx, c, m
        end
    end
    return best_idx, best_coeffs, best_mse
end

outdir = make_run_dir("viz_ude_nonstationary")
println("\nSaving plots to: ", outdir)

# ─── Style (matching viz_nonstationary.jl) ────────────────────────────────────

RLABELS = ["R1 [0,200)", "R2 [200,290)", "Tr [290,300)", "R3 [300,500]"]
RCOLORS = [:steelblue, :darkorange, :mediumpurple, :forestgreen]
BT      = [200.0, 290.0, 300.0]
EQ_R1   = [0.05 / 0.10, 0.05 / 0.10]
EQ_R3   = [0.05 / 0.10, 0.04 / 0.12]

add_bnd!(p) = foreach(t -> vline!(p, [t]; linestyle = :dash, color = :gray60,
                                   linewidth = 1.2, label = ""), BT)

function add_bands!(p, y_lo, y_hi)
    for i in 1:4
        xs = [breakpoints[i], breakpoints[i+1], breakpoints[i+1], breakpoints[i], breakpoints[i]]
        ys = [y_lo, y_lo, y_hi, y_hi, y_lo]
        plot!(p, xs, ys; seriestype = :shape, fillalpha = 0.08,
              color = RCOLORS[i], linewidth = 0, label = "")
    end
end

save_plot(p, name) = (savefig(p, joinpath(outdir, name)); println("  → $name"))

# ─── Stage 1 SINDy baseline: exhaustive OLS on full derivatives ──────────────

println("Fitting Stage 1 SINDy baseline (exhaustive OLS)…")
idx_s1_1, coeffs_s1_1, _ = exhaustive_sindy(Phi, dX_true[1, :], all_models)
idx_s1_2, coeffs_s1_2, _ = exhaustive_sindy(Phi, dX_true[2, :], all_models)
dX_s1 = vcat((Phi[:, idx_s1_1] * coeffs_s1_1)', (Phi[:, idx_s1_2] * coeffs_s1_2)')
println("  Stage 1 terms: eq1=$(feat_names[idx_s1_1]), eq2=$(feat_names[idx_s1_2])")

# ── Plot 1: timeseries_ude.png ─────────────────────────────────────────────────
println("Plot 1: time series…")
let
    function ts_sub(row, ylabel, title)
        y_lo = minimum(X_clean[row, :]) - 0.02
        y_hi = maximum(X_clean[row, :]) + 0.02
        p = plot(; xlabel = "t", ylabel = ylabel, title = title,
                   legend = :topright, size = (700, 280), dpi = 150,
                   leftmargin = 5Plots.mm, bottommargin = 5Plots.mm)
        add_bands!(p, y_lo, y_hi)
        for i in 1:4
            idx = findall(masks[i]); isempty(idx) && continue
            plot!(p, t_full[idx], X_clean[row, idx]; color = RCOLORS[i],
                  linewidth = 1.8, label = (i == 1 ? "True" : ""))
        end
        plot!(p, t_full, Xhat_ude[row, :]; color = :black, linewidth = 1.5,
              linestyle = :dash, label = "UDE")
        add_bnd!(p); ylims!(p, y_lo, y_hi)
        return p
    end
    pv = ts_sub(1, "v  (employment rate)", "v(t): truth vs UDE")
    pu = ts_sub(2, "u  (wage share)",      "u(t): truth vs UDE")
    save_plot(plot(pv, pu; layout = (2, 1), size = (700, 560), dpi = 150), "timeseries_ude.png")
end

# ── Plot 2: phase_ude.png ──────────────────────────────────────────────────────
println("Plot 2: phase portrait…")
let
    p = plot(; xlabel = "v  (employment rate)", ylabel = "u  (wage share)",
               title = "Phase portrait: truth vs UDE", legend = :outertopright,
               size = (620, 500), dpi = 150, leftmargin = 5Plots.mm, bottommargin = 5Plots.mm)
    for i in 1:4
        idx = findall(masks[i]); isempty(idx) && continue
        plot!(p, X_clean[1, idx], X_clean[2, idx]; color = RCOLORS[i],
              linewidth = 1.6, label = RLABELS[i])
    end
    plot!(p, Xhat_ude[1, :], Xhat_ude[2, :]; color = :black, linewidth = 1.4,
          linestyle = :dash, label = "UDE")
    scatter!(p, [EQ_R1[1]], [EQ_R1[2]]; marker = :star5, markersize = 10,
             color = RCOLORS[1], label = "Eq R1")
    scatter!(p, [EQ_R3[1]], [EQ_R3[2]]; marker = :star5, markersize = 10,
             color = RCOLORS[4], label = "Eq R3")
    scatter!(p, [X_clean[1, 1]], [X_clean[2, 1]]; marker = :circle, markersize = 6,
             color = :black, label = "x₀")
    save_plot(p, "phase_ude.png")
end

# ── Plot 3: nn_vs_true_interaction.png ─────────────────────────────────────────
println("Plot 3: NN vs true interaction…")
let
    # True NN target at each t = full ODE derivative minus the fixed mechanistic term
    true_c1 = dX_true[1, :] .- ETA1_FIXED  .* X_clean[1, :]   # −θ₁·v·u in R1/R3
    true_c2 = dX_true[2, :] .+ ETA2_FIXED  .* X_clean[2, :]   # +θ₂·v·u in R1/R3

    function nn_sub(true_c, nn_out, ylabel, title)
        y_lo = min(minimum(true_c), minimum(nn_out)) - 0.005
        y_hi = max(maximum(true_c), maximum(nn_out)) + 0.005
        p = plot(; xlabel = "t", ylabel = ylabel, title = title,
                   legend = :topright, size = (700, 280), dpi = 150,
                   leftmargin = 5Plots.mm, bottommargin = 5Plots.mm)
        add_bands!(p, y_lo, y_hi)
        plot!(p, t_full, true_c; color = :steelblue, linewidth = 1.6, label = "True interaction")
        plot!(p, t_full, nn_out; color = :firebrick, linewidth = 1.4,
              linestyle = :dash, label = "NN output")
        add_bnd!(p); ylims!(p, y_lo, y_hi)
        return p
    end
    p1 = nn_sub(true_c1, nn_at_true[1, :], "dv/dt correction", "NN₁ vs true correction (dv/dt)")
    p2 = nn_sub(true_c2, nn_at_true[2, :], "du/dt correction", "NN₂ vs true correction (du/dt)")
    save_plot(plot(p1, p2; layout = (2, 1), size = (700, 560), dpi = 150),
              "nn_vs_true_interaction.png")
end

# ── Plot 4: deriv_residual_comparison.png ─────────────────────────────────────
println("Plot 4: derivative residual comparison…")
let
    resid_s1 = [norm(dX_true[:, j] - dX_s1[:, j])    for j in eachindex(t_full)]
    resid_s2 = [norm(dX_true[:, j] - dX_sindy[:, j]) for j in eachindex(t_full)]

    y_hi = max(maximum(resid_s1), maximum(resid_s2)) * 1.1

    p = plot(; xlabel = "t", ylabel = "‖ẋ_true − ẋ_model‖",
               title = "Derivative residual: Stage 1 SINDy vs Stage 2 UDE+SINDy",
               legend = :topright, size = (700, 340), dpi = 150,
               leftmargin = 6Plots.mm, bottommargin = 6Plots.mm)
    add_bands!(p, 0.0, y_hi)
    plot!(p, t_full, resid_s1;
        color = :steelblue, linewidth = 1.6, label = "Stage 1 SINDy")
    plot!(p, t_full, resid_s2; color = :firebrick, linewidth = 1.6,
          linestyle = :dash, label = "Stage 2 UDE+SINDy")
    add_bnd!(p); ylims!(p, 0.0, y_hi)
    save_plot(p, "deriv_residual_comparison.png")

    # Per-regime RMSE comparison table
    println("\n  Derivative RMSE comparison:")
    println("    Regime    Stage1_SINDy   Stage2_UDE+SINDy   Ratio")
    rmse_s1_all = Float64[]
    rmse_s2_all = Float64[]
    for (i, lab) in enumerate(RLABELS)
        m = masks[i]; !any(m) && continue
        rs1 = sqrt(mean([resid_s1[j]^2 for j in eachindex(t_full) if m[j]]))
        rs2 = sqrt(mean([resid_s2[j]^2 for j in eachindex(t_full) if m[j]]))
        append!(rmse_s1_all, resid_s1[j]^2 for j in eachindex(t_full) if m[j])
        append!(rmse_s2_all, resid_s2[j]^2 for j in eachindex(t_full) if m[j])
        ratio = rs2 == 0 ? Inf : rs1 / rs2
        println("    $(rpad(lab, 10))$(lpad(round(rs1, sigdigits=4), 12))   $(lpad(round(rs2, sigdigits=4), 16))   $(round(ratio, sigdigits=3))")
    end
    g_s1 = sqrt(mean(rmse_s1_all))
    g_s2 = sqrt(mean(rmse_s2_all))
    g_ratio = g_s2 == 0 ? Inf : g_s1 / g_s2
    println("    $(rpad("Global", 10))$(lpad(round(g_s1, sigdigits=4), 12))   $(lpad(round(g_s2, sigdigits=4), 16))   $(round(g_ratio, sigdigits=3))")
end

# ── Plot 5: coefficient_recovery.png ──────────────────────────────────────────
println("Plot 5: v*u coefficient recovery…")
let
    # Stage 1: exhaustive OLS on full derivative (same library as Stage 2)
    function get_vu(target)
        idx, coeffs, _ = exhaustive_sindy(Phi, target, all_models)
        pos = findfirst(==(VU_IDX), idx)
        isnothing(pos) ? 0.0 : coeffs[pos]
    end
    s1_vu1 = get_vu(dX_true[1, :])
    s1_vu2 = get_vu(dX_true[2, :])

    # Stage 2: from sindy_results (correction-term fit)
    s2_vu1 = let r = sindy_results[1]; pos = findfirst(==(VU_IDX), r.idx)
        isnothing(pos) ? 0.0 : r.coeffs[pos] end
    s2_vu2 = let r = sindy_results[2]; pos = findfirst(==(VU_IDX), r.idx)
        isnothing(pos) ? 0.0 : r.coeffs[pos] end

    bar_labels = ["True R1", "True R3", "Stage 1", "Stage 2"]
    bar_cols   = [:steelblue, :forestgreen, :darkorange, :firebrick]
    vals_eq1   = [-0.10, -0.12, s1_vu1, s2_vu1]
    vals_eq2   = [ 0.10,  0.10, s1_vu2, s2_vu2]

    n_bars = 4; bar_w = 0.18; group_w = n_bars * bar_w + 0.12
    centers = [1.0, 1.0 + group_w]

    p = plot(; xlabel = "Equation", ylabel = "v·u coefficient",
               title = "v*u coefficient recovery",
               legend = :outertopright, size = (660, 400), dpi = 150,
               leftmargin = 6Plots.mm, bottommargin = 6Plots.mm,
               xticks = (centers, ["Eq1  dv/dt", "Eq2  du/dt"]))
    hline!(p, [0.0]; color = :gray60, linewidth = 1.0, linestyle = :dot, label = "")
    for bi in 1:n_bars
        offset = (bi - (n_bars + 1) / 2) * bar_w
        xs = centers .+ offset
        ys = [vals_eq1[bi], vals_eq2[bi]]
        bar!(p, xs, ys; bar_width = bar_w * 0.9, color = bar_cols[bi], label = bar_labels[bi])
    end
    save_plot(p, "coefficient_recovery.png")
end

# ── Plot 6: loss_curve.png ─────────────────────────────────────────────────────
println("Plot 6: training loss curve…")
let
    n_l = length(losses)
    p = plot(1:n_l, losses; xlabel = "Iteration (non-Inf only)",
               ylabel = "Multiple-shooting loss", title = "UDE training loss",
               color = :firebrick, linewidth = 1.4, legend = :topright,
               yscale = :log10,
               size = (700, 320), dpi = 150,
               leftmargin = 6Plots.mm, bottommargin = 6Plots.mm, label = "Loss")
    # Approximate phase boundaries (Adam→Adam at ~5000, Adam→BFGS at ~10000)
    for (x, lab) in zip([5000, 10000], ["Adam→Adam", "Adam→BFGS"])
        x <= n_l && vline!(p, [x]; linestyle = :dash, color = :gray50,
                            linewidth = 1.5, label = lab)
    end
    save_plot(p, "loss_curve.png")
end

println("\nAll plots saved to: ", outdir)

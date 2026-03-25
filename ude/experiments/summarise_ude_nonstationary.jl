# experiments/summarise_ude_nonstationary.jl
#
# Compute per-regime derivative-residual RMSE / NRMSE / R² for BOTH methods:
#   Stage 1 — SR3 SINDy (re-fit here, evaluated at true states)
#   Stage 2 — UDE+SINDy (loaded from JLD2)
# Then write a combined comparison CSV where every row uses the same metric.
#
# Usage:  julia --project=. experiments/summarise_ude_nonstationary.jl
#
# Expected runtime: ~5–15 min (dominated by SINDy fitting, 5 seeds × 4 noise)

include(joinpath(@__DIR__, "..", "src", "ScientificML.jl"))
using .ScientificML

include(joinpath(@__DIR__, "..", "src", "simulate_nonstationary.jl"))

using JLD2, Statistics, LinearAlgebra, DelimitedFiles

# ─── Metric helpers (operate on matrices shaped n_states × n_time) ───────────

deriv_rmse(dX_true, dX_pred)  = sqrt(mean(abs2, dX_true .- dX_pred))

function deriv_nrmse(dX_true, dX_pred; ϵ = 1e-12)
    denom = max(std(vec(dX_true)), ϵ)
    return deriv_rmse(dX_true, dX_pred) / denom
end

function deriv_r2(dX_true, dX_pred; ϵ = 1e-12)
    y  = vec(dX_true)
    ŷ  = vec(dX_pred)
    ss_res = sum(abs2, y .- ŷ)
    ss_tot = max(sum(abs2, y .- mean(y)), ϵ)
    return 1 - ss_res / ss_tot
end

function regime_deriv_metrics(dX_true, dX_pred, masks)
    map(1:4) do i
        m = masks[i]
        !any(m) && return (RMSE = NaN, NRMSE = NaN, R2 = NaN)
        (RMSE  = deriv_rmse(dX_true[:, m],  dX_pred[:, m]),
         NRMSE = deriv_nrmse(dX_true[:, m], dX_pred[:, m]),
         R2    = deriv_r2(dX_true[:, m],    dX_pred[:, m]))
    end
end

# ─── Evaluate a fitted SR3 SINDy model at arbitrary states ───────────────────

## eval_sindy_at_states is exported from ScientificML (src/methods/sindy.jl)

# ─── Equation string helpers (matching exp_nonstationary.jl) ─────────────────

"Extract RHS string from a ModelingToolkit equation, sanitising commas."
function eq_rhs_str(eq)
    s = string(eq.rhs)
    return replace(s, ',' => ';')
end

"Count additive terms in a symbolic equation string."
count_terms_str(s::String) = 1 + count(" + ", s)

# ─── UDE+SINDy equation string from sindy_results NamedTuple ────────────────

const ETA1_FIXED = 0.05
const ETA2_FIXED = 0.05

function ude_equation_str(sindy_result, row::Int)
    mech = row == 1 ? "$(ETA1_FIXED)*v" : "-$(ETA2_FIXED)*u"
    terms = [string(round(sindy_result.coeffs[i], sigdigits = 4),
                    "*", sindy_result.names[i])
             for i in eachindex(sindy_result.names)]
    s = mech * " + " * join(terms, " + ")
    return replace(s, ',' => ';')
end

ude_count_terms(sr) = length(sr.names) + 1   # +1 for the fixed mechanistic term

# ─── Shared column structure ─────────────────────────────────────────────────

HEADER = [
    "seed", "noise", "N_points",
    "global_RMSE", "global_NRMSE", "global_R2",
    "r1_RMSE", "r1_NRMSE", "r1_R2",
    "r2_RMSE", "r2_NRMSE", "r2_R2",
    "tr_RMSE", "tr_NRMSE", "tr_R2",
    "r3_RMSE", "r3_NRMSE", "r3_R2",
    "best_lambda", "n_terms", "eq_v", "eq_u"
]

function metrics_row(seed, noise, N_pts, g_rmse, g_nrmse, g_r2, r_metrics,
                     best_lambda, n_terms, eq_v, eq_u)
    return [
        seed, noise, N_pts,
        g_rmse, g_nrmse, g_r2,
        r_metrics[1].RMSE, r_metrics[1].NRMSE, r_metrics[1].R2,
        r_metrics[2].RMSE, r_metrics[2].NRMSE, r_metrics[2].R2,
        r_metrics[3].RMSE, r_metrics[3].NRMSE, r_metrics[3].R2,
        r_metrics[4].RMSE, r_metrics[4].NRMSE, r_metrics[4].R2,
        best_lambda, n_terms, eq_v, eq_u
    ]
end

# ─── Output directory ────────────────────────────────────────────────────────

outdir = make_run_dir("deriv_comparison")
println("Output directory: ", outdir)

sindy_summary_path = joinpath(outdir, "summary_sindy.csv")
ude_summary_path   = joinpath(outdir, "summary_ude_sindy.csv")
combined_path      = joinpath(outdir, "summary_combined.csv")

# ═════════════════════════════════════════════════════════════════════════════
# STAGE 1 — SINDy derivative residuals (re-fit from scratch)
# ═════════════════════════════════════════════════════════════════════════════

println("\n══ Stage 1: SINDy derivative residuals ══")

# Config matching exp_nonstationary.jl exactly
seeds        = 1:5
noise_levels = [0.0, 1e-3, 1e-2, 5e-2]
saveat       = 0.5
polyorder    = 3
nu           = 1.0
λs           = exp10.(-6:0.5:0)
prox         = ScientificML.DataDrivenSparse.SoftThreshold()

x0_true, regime_configs, breakpoints = nonstationary_goodwin_configs()

write_csv(sindy_summary_path, reshape(HEADER, 1, :))

for seed in seeds
    rng = ScientificML.StableRNGs.StableRNG(seed)

    t_full, X_clean = simulate_nonstationary(x0_true, regime_configs; saveat = saveat)
    masks   = get_regime_masks(t_full, breakpoints)
    dX_true = evaluate_true_derivatives(t_full, X_clean, regime_configs, breakpoints)
    N_pts   = length(t_full)

    println("\n  seed=$seed  ($N_pts points)")

    for noise in noise_levels
        Xn = (noise == 0.0) ? copy(X_clean) :
             add_relative_noise(X_clean, rng; noise_magnitude = noise)

        # Defaults (remain NaN on failure)
        g_rmse = NaN; g_nrmse = NaN; g_r2 = NaN
        r_metrics   = [(RMSE=NaN, NRMSE=NaN, R2=NaN) for _ in 1:4]
        n_terms     = NaN
        eq_v        = "failed"
        eq_u        = "failed"

        try
            res, system, _ = fit_sindy_sr3(
                t_full, Xn;
                polyorder = polyorder, λs = λs, nu = nu, proximal = prox, rng = rng
            )

            # Evaluate discovered SINDy ODE at true states → derivative prediction
            dX_sindy = eval_sindy_at_states(res, t_full, X_clean)

            g_rmse    = deriv_rmse(dX_true, dX_sindy)
            g_nrmse   = deriv_nrmse(dX_true, dX_sindy)
            g_r2      = deriv_r2(dX_true, dX_sindy)
            r_metrics = regime_deriv_metrics(dX_true, dX_sindy, masks)

            eqs     = ScientificML.ModelingToolkit.equations(system)
            eq_v    = eq_rhs_str(eqs[1])
            eq_u    = eq_rhs_str(eqs[2])
            n_terms = count_terms_str(eq_v) + count_terms_str(eq_u)
        catch e
            println("    ! SINDy failed (seed=$seed, noise=$noise): ", e)
        end

        row = metrics_row(seed, noise, N_pts, g_rmse, g_nrmse, g_r2, r_metrics,
                          NaN, n_terms, eq_v, eq_u)
        append_csv_row(sindy_summary_path, reshape(row, 1, :))

        r2_str = isnan(g_r2) ? "NaN" : string(round(g_r2, sigdigits = 4))
        println("    noise=$noise  deriv_RMSE=$(round(g_rmse, sigdigits=4))  R²=$r2_str")
    end
end

println("\nSINDy summary → ", sindy_summary_path)

# ═════════════════════════════════════════════════════════════════════════════
# STAGE 2 — UDE+SINDy derivative residuals (from JLD2)
# ═════════════════════════════════════════════════════════════════════════════

println("\n══ Stage 2: UDE+SINDy derivative residuals ══")

jld2_dir = joinpath(@__DIR__, "..", "results")
noise_files = filter(readdir(jld2_dir; join = true)) do f
    occursin(r"ude_nonstationary_noise_.*\.jld2$", basename(f))
end
sort!(noise_files)

isempty(noise_files) && error("No ude_nonstationary_noise_*.jld2 found in $jld2_dir")
println("Found $(length(noise_files)) JLD2 files")

write_csv(ude_summary_path, reshape(HEADER, 1, :))

for path in noise_files
    data  = load(path)
    label = if haskey(data, "noise_label")
        data["noise_label"]
    else
        m = match(r"noise_([\d.e+-]+)\.jld2$", basename(path))
        m === nothing ? error("Cannot parse noise from filename: $path") : m[1]
    end
    noise = parse(Float64, label)

    dX_true  = data["dX_true"]
    dX_sindy = data["dX_sindy"]
    masks    = data["masks"]
    N_pts    = size(dX_true, 2)
    sr       = data["sindy_results"]

    g_rmse    = deriv_rmse(dX_true, dX_sindy)
    g_nrmse   = deriv_nrmse(dX_true, dX_sindy)
    g_r2      = deriv_r2(dX_true, dX_sindy)
    r_metrics = regime_deriv_metrics(dX_true, dX_sindy, masks)

    eq_v    = ude_equation_str(sr[1], 1)
    eq_u    = ude_equation_str(sr[2], 2)
    n_terms = ude_count_terms(sr[1]) + ude_count_terms(sr[2])

    row = metrics_row(1, noise, N_pts, g_rmse, g_nrmse, g_r2, r_metrics,
                      NaN, n_terms, eq_v, eq_u)
    append_csv_row(ude_summary_path, reshape(row, 1, :))

    println("  noise=$label  deriv_RMSE=$(round(g_rmse, sigdigits=4))  " *
            "deriv_R²=$(round(g_r2, sigdigits=4))")
end

println("\nUDE+SINDy summary → ", ude_summary_path)

# ═════════════════════════════════════════════════════════════════════════════
# COMBINED — stack both with a method column
# ═════════════════════════════════════════════════════════════════════════════

println("\n══ Building combined CSV ══")

sindy_data, sindy_hdr = readdlm(sindy_summary_path, ',', header = true)
ude_data,   ude_hdr   = readdlm(ude_summary_path,   ',', header = true)

sindy_hdr = vec(sindy_hdr)
ude_hdr   = vec(ude_hdr)
@assert sindy_hdr == ude_hdr "Column headers differ"

combined_header = vcat(["method"], sindy_hdr)

n_sindy = size(sindy_data, 1)
n_ude   = size(ude_data, 1)
ncols   = length(sindy_hdr)

combined = Matrix{Any}(undef, n_sindy + n_ude, ncols + 1)
for r in 1:n_sindy
    combined[r, 1] = "sindy"
    for c in 1:ncols
        combined[r, c + 1] = sindy_data[r, c]
    end
end
for r in 1:n_ude
    combined[n_sindy + r, 1] = "ude_sindy"
    for c in 1:ncols
        combined[n_sindy + r, c + 1] = ude_data[r, c]
    end
end

write_csv(combined_path, vcat(reshape(combined_header, 1, :), combined))

println("Combined CSV → ", combined_path)
println("  $n_sindy sindy + $n_ude ude_sindy = $(n_sindy + n_ude) total rows")
println("\nDone.")

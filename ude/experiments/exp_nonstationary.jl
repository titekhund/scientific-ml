# experiments/exp_nonstationary.jl
#
# Stage 1 baseline: single global SINDy (SR3) on nonstationary Goodwin data.
#
# Simulates a 3-regime (R1 → R2-forced → Transition → R3) Goodwin system,
# fits one global SINDy model on the full trajectory, and evaluates per-regime.
#
# Grid: seeds × noise_levels
# Output: results/exp_nonstationary/<timestamp>/summary.csv
#
# Run with:  julia --project=. experiments/exp_nonstationary.jl

include(joinpath(@__DIR__, "..", "src", "ScientificML.jl"))
using .ScientificML

include(joinpath(@__DIR__, "..", "src", "simulate_nonstationary.jl"))

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

expname      = "exp_nonstationary"
seeds        = 1:5
noise_levels = [0.0, 1e-3, 1e-2, 5e-2]
saveat       = 0.5   # dt = 0.5 → 1001 points over [0, 500]

# SINDy (SR3) settings — same as exp_noise_length baseline
polyorder = 3
nu        = 1.0
λs        = exp10.(-6:0.5:0)
prox      = ScientificML.DataDrivenSparse.SoftThreshold()

# Regime config (fixed across all runs — parameters match Python reference)
x0_true, regime_configs, breakpoints = nonstationary_goodwin_configs()

# Equilibrium points for reference
eq_r1 = (v = 0.05 / 0.10, u = 0.05 / 0.10)   # v*=0.5, u*=0.5
eq_r3 = (v = 0.05 / 0.10, u = 0.04 / 0.12)   # v*=0.5, u*≈0.333

regime_labels = ["R1", "R2", "Tr", "R3"]

# ─────────────────────────────────────────────────────────────────────────────
# Output
# ─────────────────────────────────────────────────────────────────────────────

outdir       = make_run_dir(expname)
summary_path = joinpath(outdir, "summary.csv")
println("Output directory: ", outdir)

header = [
    "seed", "noise", "N_points",
    "global_RMSE", "global_NRMSE", "global_R2",
    "r1_RMSE", "r1_NRMSE", "r1_R2",
    "r2_RMSE", "r2_NRMSE", "r2_R2",
    "tr_RMSE", "tr_NRMSE", "tr_R2",
    "r3_RMSE", "r3_NRMSE", "r3_R2",
    "best_lambda", "n_terms", "eq_v", "eq_u"
]
write_csv(summary_path, reshape(header, 1, :))

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

"Extract RHS string from a ModelingToolkit equation, sanitising commas."
function eq_rhs_str(eq)
    s = string(eq.rhs)
    return replace(s, ',' => ';')   # guard against CSV delimiter collisions
end

"""
Count additive terms in a symbolic equation string.
Counts ' + ' occurrences and adds 1 for the final term.
This is a conservative lower bound — negative leading terms read as one token.
"""
function count_terms_str(s::String)
    return 1 + count(" + ", s)
end

# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────

for seed in seeds
    rng = ScientificML.StableRNGs.StableRNG(seed)

    # Simulate once per seed — noise is added separately for each noise level.
    # The clean trajectory is the same regardless of noise.
    t_full, X_clean = simulate_nonstationary(x0_true, regime_configs; saveat = saveat)
    masks = get_regime_masks(t_full, breakpoints)

    println("\n═══ seed = $seed  ($(length(t_full)) time points) ═══")

    for noise in noise_levels

        # Add noise to clean trajectory
        Xn = (noise == 0.0) ? copy(X_clean) :
             add_relative_noise(X_clean, rng; noise_magnitude = noise)

        # ── Default values (populated on success, remain NaN on failure) ────
        global_RMSE = NaN; global_NRMSE = NaN; global_R2 = NaN
        r_metrics   = [(RMSE=NaN, NRMSE=NaN, R2=NaN) for _ in 1:4]
        best_lambda = NaN
        n_terms     = NaN
        eq_v        = "failed"
        eq_u        = "failed"

        try
            # Fit a single SINDy model on the full noisy trajectory
            res, system, _ = fit_sindy_sr3(
                t_full, Xn;
                polyorder = polyorder, λs = λs, nu = nu, proximal = prox, rng = rng
            )

            # Roll out from the clean initial condition for fair comparison
            Xhat = predict_sindy(res, X_clean[:, 1], t_full)

            # Global metrics (SINDy rollout vs clean ground truth)
            gm           = trajectory_metrics(X_clean, Xhat)
            global_RMSE  = gm.rmse
            global_NRMSE = gm.nrmse
            global_R2    = gm.r2

            # Per-regime metrics
            r_metrics = map(1:4) do i
                if !any(masks[i])
                    return (RMSE=NaN, NRMSE=NaN, R2=NaN)
                end
                m = trajectory_metrics(X_clean[:, masks[i]], Xhat[:, masks[i]])
                (RMSE=m.rmse, NRMSE=m.nrmse, R2=m.r2)
            end

            # Equation strings (LHS ~ RHS; we keep only RHS)
            eqs  = ScientificML.ModelingToolkit.equations(system)
            eq_v = eq_rhs_str(eqs[1])
            eq_u = eq_rhs_str(eqs[2])

            # Term count across both equations
            n_terms = count_terms_str(eq_v) + count_terms_str(eq_u)

        catch e
            println("  ! SINDy failed (seed=$seed, noise=$noise): ", e)
        end

        row = [
            seed, noise, length(t_full),
            global_RMSE, global_NRMSE, global_R2,
            r_metrics[1].RMSE, r_metrics[1].NRMSE, r_metrics[1].R2,
            r_metrics[2].RMSE, r_metrics[2].NRMSE, r_metrics[2].R2,
            r_metrics[3].RMSE, r_metrics[3].NRMSE, r_metrics[3].R2,
            r_metrics[4].RMSE, r_metrics[4].NRMSE, r_metrics[4].R2,
            best_lambda, n_terms, eq_v, eq_u
        ]
        append_csv_row(summary_path, reshape(row, 1, :))

        # Progress log
        r2_str = isnan(global_R2) ? "NaN" : string(round(global_R2, sigdigits=4))
        println("  noise=$noise  global_RMSE=$(round(global_RMSE, sigdigits=4))  " *
                "R²=$r2_str")
        for i in 1:4
            rm = r_metrics[i].RMSE
            println("    $(regime_labels[i])  RMSE=$(round(rm, sigdigits=4))")
        end
    end
end

println("\nFinished.  Summary → ", summary_path)

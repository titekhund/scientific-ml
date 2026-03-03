# experiments/exp_noise_length.jl
#
# Experiment: How noise level and data length (N points) affect performance.
# For now, this script generates datasets and records baseline metrics:
#   - error between noisy observations Xn and clean trajectory X (RMSE/NRMSE/R²).
# Later you can plug in SINDy / UDE-SINDy to compute model-fit metrics too.

include(joinpath(@__DIR__, "..", "src", "ScientificML.jl"))
using .ScientificML

# -------------------------
# Config
# -------------------------
expname = "exp_noise_length"

# Noise levels (relative-to-mean Gaussian noise, like your earlier script)
noise_levels = [0.0, 1e-4, 1e-3, 1e-2, 5e-2]

# Different dataset lengths (number of time points kept after simulation)
# Note: with your current defaults (tspan=(0,10), saveat=0.25) you only have 41 points total.
# So N will be clipped to available points automatically.
N_values = [20, 41, 80, 200]

# Replicates
seeds = 1:5

# Simulation settings
saveat = 0.25

# Choose system: Goodwin (current systems.jl)
prob = make_goodwin_problem()

# -------------------------
# Output folder
# -------------------------
outdir = make_run_dir(expname)  # results/exp_noise_length/<timestamp>/
println("Saving to: ", outdir)

# Summary rows will be appended here
summary_path = joinpath(outdir, "summary.csv")

# Write CSV header
header = ["seed", "noise", "N", "T", "dt", "rmse", "nrmse", "r2", "rmse_v", "rmse_u", "nrmse_v", "nrmse_u"]
write_csv(summary_path, reshape(header, 1, :))

# -------------------------
# Main loop
# -------------------------
for seed in seeds
    rng = ScientificML.StableRNGs.StableRNG(seed)

    for noise in noise_levels
        for N in N_values
            # Generate dataset
            t, X, Xn = make_dataset(prob, rng; saveat=saveat, N=N, noise_magnitude=noise)

            # Baseline metrics: how far noisy observations are from clean trajectory
            m = trajectory_metrics(X, Xn)

            # Per-state metrics (2 states): [v, u] for Goodwin
            rmse_v, rmse_u = m.per_state_rmse
            nrmse_v, nrmse_u = m.per_state_nrmse

            T = t[end] - t[1]
            dt = length(t) > 1 ? (t[2] - t[1]) : NaN

            row = [seed, noise, length(t), T, dt, m.rmse, m.nrmse, m.r2, rmse_v, rmse_u, nrmse_v, nrmse_u]
            write_csv(summary_path, row')  # append as 1-row matrix by writing a row

            # Optional: save the actual dataset for this run (comment out if you want fewer files)
            run_tag = "seed$(seed)_noise$(noise)_N$(length(t))"
            write_csv(joinpath(outdir, "t_" * run_tag * ".csv"), t)
            write_csv(joinpath(outdir, "X_clean_" * run_tag * ".csv"), X')
            write_csv(joinpath(outdir, "X_noisy_" * run_tag * ".csv"), Xn')

            println("Done: seed=$(seed) noise=$(noise) N=$(length(t)) rmse=$(round(m.rmse, digits=6))")
        end
    end
end

println("Finished. Summary saved to: ", summary_path)

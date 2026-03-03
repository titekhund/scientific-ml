# experiments/exp_noise_length.jl
#
# Experiment: How noise level and data length (N points) affect performance.
# Generates datasets and records:
#   (1) baseline observation error: Xn vs X (RMSE/NRMSE/R²)
#   (2) SR3-SINDy rollout error: X̂ vs X (RMSE/NRMSE/R²)

include(joinpath(@__DIR__, "..", "src", "ScientificML.jl"))
using .ScientificML

# -------------------------
# Config
# -------------------------
expname = "exp_noise_length"

# Noise levels (relative-to-mean Gaussian noise)
noise_levels = [0.0, 1e-4, 1e-3, 1e-2, 5e-2]

# Dataset lengths (number of time points kept after simulation)
# With tspan=(0,10) and saveat=0.25 you have only 41 points max; larger N will be clipped.
N_values = [20, 41, 80, 200]

# Replicates
seeds = 1:5

# Simulation settings
saveat = 0.25

# SINDy (SR3) settings
polyorder = 3
nu = 1.0
λs = exp10.(-6:0.5:0)         # stronger thresholds than (-8:-2) -> more sparsity
prox = ScientificML.DataDrivenSparse.SoftThreshold()
prox_name = prox isa ScientificML.DataDrivenSparse.HardThreshold ? "hard" : "soft"

# Choose system: Goodwin
prob = make_goodwin_problem()

# -------------------------
# Output folder
# -------------------------
outdir = make_run_dir(expname)  # results/exp_noise_length/<timestamp>/
println("Saving to: ", outdir)

summary_path = joinpath(outdir, "summary.csv")

# Write CSV header ONCE
header = [
  "seed","noise","N","T","dt",
  "rmse_obs","nrmse_obs","r2_obs",
  "rmse_v_obs","rmse_u_obs","nrmse_v_obs","nrmse_u_obs",
  "sindy_rmse","sindy_nrmse","sindy_r2",
  "sindy_rmse_v","sindy_rmse_u","sindy_nrmse_v","sindy_nrmse_u",
  "polyorder","nu","lambda_min","lambda_max","prox"
]
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

            # Baseline metrics: noisy observations vs clean trajectory
            m_obs = trajectory_metrics(X, Xn)
            rmse_v_obs, rmse_u_obs = m_obs.per_state_rmse
            nrmse_v_obs, nrmse_u_obs = m_obs.per_state_nrmse

            # SINDy rollout metrics (defaults to NaN if anything fails)
            sindy_rmse = NaN; sindy_nrmse = NaN; sindy_r2 = NaN
            sindy_rmse_v = NaN; sindy_rmse_u = NaN
            sindy_nrmse_v = NaN; sindy_nrmse_u = NaN

            try
                res, system, _ = fit_sindy_sr3(t, Xn; polyorder=polyorder, λs=λs, nu=nu, proximal=prox, rng=rng)
                Xhat = predict_sindy(res, Xn[:, 1], t)

                m_sindy = trajectory_metrics(X, Xhat)
                sindy_rmse  = m_sindy.rmse
                sindy_nrmse = m_sindy.nrmse
                sindy_r2    = m_sindy.r2

                sindy_rmse_v, sindy_rmse_u = m_sindy.per_state_rmse
                sindy_nrmse_v, sindy_nrmse_u = m_sindy.per_state_nrmse
            catch e
                println("SINDy failed for seed=$(seed) noise=$(noise) N=$(length(t)) : ", e)
            end

            # Time grid info
            T  = t[end] - t[1]
            dt = length(t) > 1 ? (t[2] - t[1]) : NaN

            row = [
              seed, noise, length(t), T, dt,
              m_obs.rmse, m_obs.nrmse, m_obs.r2,
              rmse_v_obs, rmse_u_obs, nrmse_v_obs, nrmse_u_obs,
              sindy_rmse, sindy_nrmse, sindy_r2,
              sindy_rmse_v, sindy_rmse_u, sindy_nrmse_v, sindy_nrmse_u,
              polyorder, nu, minimum(λs), maximum(λs), prox_name
            ]

            # APPEND (do not overwrite)
            append_csv_row(summary_path, reshape(row, 1, :))

            println("Done: seed=$(seed) noise=$(noise) N=$(length(t)) obs_rmse=$(round(m_obs.rmse, digits=6)) sindy_rmse=$(round(sindy_rmse, digits=6))")
        end
    end
end

println("Finished. Summary saved to: ", summary_path)

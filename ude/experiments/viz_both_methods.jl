#!/usr/bin/env julia
#= ──────────────────────────────────────────────────────────────────────
   viz_both_methods.jl
   2×3 panel figure comparing Direct SR3 vs UDE+SR3 at noise=0.
   Columns: time series, phase portrait, derivative residual norm.
   ────────────────────────────────────────────────────────────────────── =#

using JLD2, Plots, Statistics, OrdinaryDiffEq
gr()

const FIGDIR = "results/figures"
mkpath(FIGDIR)

const BOUNDS     = [200.0, 290.0, 300.0]
const REG_NAMES  = ["R1", "R2", "Tr", "R3"]
const EQ_POINTS  = ([0.5, 0.5], [0.5, 1/3])  # R1 eq, R3 eq

# Feature name → function for rollout ODE
const TERM_MAP = Dict{String, Function}(
    "v"    => (v, u) -> v,
    "u"    => (v, u) -> u,
    "v^2"  => (v, u) -> v^2,
    "v u"  => (v, u) -> v * u,
    "u^2"  => (v, u) -> u^2,
)

function add_regime_lines!(p)
    for b in BOUNDS
        vline!(p, [b]; color=:grey, linestyle=:dash, alpha=0.5, label=false)
    end
end

# ── Load data ────────────────────────────────────────────────────────
@info "Loading data..."

# UDE+SR3
d_ude = JLD2.load("results/ude_nonstationary_noise_0.0.jld2")
t     = d_ude["t"]
Xc    = d_ude["X_clean"]
dXt   = d_ude["dX_true"]
bp    = d_ude["breakpoints"]
masks = d_ude["masks"]
N     = length(t)

# UDE+SR3 discovered derivatives and rollout coefficients
dX_ude     = d_ude["dX_pysindy"]
ude_af1    = d_ude["pysindy_active_f1"]  # Vector of (name, coeff)
ude_af2    = d_ude["pysindy_active_f2"]
ude_c_f1   = d_ude["pysindy_coeffs_f1"]
ude_c_f2   = d_ude["pysindy_coeffs_f2"]

# Direct SR3 — seed=1, noise=0.0
d_dir = JLD2.load("results/direct_sr3_nonstationary/results.jld2")
dir_results = d_dir["results_all"]
dir_r = dir_results[(0.0, 1)]
dX_dir   = dir_r["dX_discovered"]
dir_af1  = dir_r["active_f1"]
dir_af2  = dir_r["active_f2"]
dir_c_f1 = dir_r["coeffs_f1"]
dir_c_f2 = dir_r["coeffs_f2"]
dir_feat = dir_r["feat_names"]

# ── Build rollout ODEs ───────────────────────────────────────────────

# Map PySINDy feature names to TERM_MAP keys
# PySINDy may use "v u" or "v^2" etc — normalize
function normalize_feat(name::String)
    s = strip(name)
    # handle potential variations
    s = replace(s, "  " => " ")
    return s
end

function build_rollout_rhs(active_f1, active_f2; eta1=0.0, eta2=0.0)
    # For UDE+SR3: eta1=0.05, eta2=0.05 (mechanistic part)
    # For Direct SR3: eta1=0.0, eta2=0.0 (linear terms included in coefficients)
    function rhs!(dx, x, p, t)
        v, u = x
        dx[1] = eta1 * v
        dx[2] = -eta2 * u
        for (name, coeff) in active_f1
            fn = normalize_feat(name)
            if haskey(TERM_MAP, fn)
                dx[1] += coeff * TERM_MAP[fn](v, u)
            end
        end
        for (name, coeff) in active_f2
            fn = normalize_feat(name)
            if haskey(TERM_MAP, fn)
                dx[2] += coeff * TERM_MAP[fn](v, u)
            end
        end
    end
    return rhs!
end

function safe_rollout(rhs!, x0, tspan, t_save)
    prob = ODEProblem(rhs!, x0, tspan)
    sol = solve(prob, Tsit5(); saveat=t_save, abstol=1e-9, reltol=1e-9,
                maxiters=1_000_000,
                isoutofdomain=(x, p, t) -> any(abs.(x) .> 1e6))
    t_sol = sol.t
    X_sol = Array(sol)
    # Pad with NaN if diverged early
    if length(t_sol) < length(t_save)
        n_missing = length(t_save) - length(t_sol)
        X_sol = hcat(X_sol, fill(NaN, 2, n_missing))
        t_sol = vcat(t_sol, t_save[length(t_sol)+1:end])
    end
    return t_sol, X_sol
end

# Direct SR3 — full system already in coefficients
dir_rhs! = build_rollout_rhs(dir_af1, dir_af2; eta1=0.0, eta2=0.0)
t_dir, X_dir = safe_rollout(dir_rhs!, Xc[:, 1], (t[1], t[end]), t)

# UDE+SR3 — add mechanistic part
# Map pysindy names (x, y, x y) → (v, u, v u) for TERM_MAP
ude_af1_mapped = [(replace(replace(n, "x" => "v"), "y" => "u"), c) for (n, c) in ude_af1]
ude_af2_mapped = [(replace(replace(n, "x" => "v"), "y" => "u"), c) for (n, c) in ude_af2]
ude_rhs! = build_rollout_rhs(ude_af1_mapped, ude_af2_mapped; eta1=0.05, eta2=0.05)
t_ude_roll, X_ude = safe_rollout(ude_rhs!, Xc[:, 1], (t[1], t[end]), t)

# ── Derivative residual norms ────────────────────────────────────────
resid_dir = sqrt.(sum((dXt .- dX_dir).^2, dims=1))[1, :]
resid_ude = sqrt.(sum((dXt .- dX_ude).^2, dims=1))[1, :]

# ── Build the 2×3 figure ─────────────────────────────────────────────
@info "Building figure..."

# Common settings
common = (grid_alpha=0.2, guidefontsize=9, legendfontsize=7, tickfontsize=7)

# ── Row 1: Direct SR3 ───────────────────────────────────────────────
# (1,1) Time series
p11 = plot(t, Xc[1, :]; lw=1.2, color=:black, label="True v", title="Direct SINDy — Time series", common...)
plot!(p11, t, Xc[2, :]; lw=1.2, color=:black, linestyle=:dash, label="True u")
plot!(p11, t_dir, X_dir[1, :]; lw=1.2, color=:red, label="SR3 v")
plot!(p11, t_dir, X_dir[2, :]; lw=1.2, color=:red, linestyle=:dash, label="SR3 u")
add_regime_lines!(p11)
ylabel!(p11, "State")
xlabel!(p11, "t")

# (1,2) Phase portrait
p12 = plot(Xc[1, :], Xc[2, :]; lw=1.2, color=:black, label="True", title="Direct SINDy — Phase portrait", common...)
plot!(p12, X_dir[1, :], X_dir[2, :]; lw=1.2, color=:red, label="SR3")
scatter!(p12, [EQ_POINTS[1][1]], [EQ_POINTS[1][2]]; marker=:circle, ms=8, color=:blue, label="R1 eq")
scatter!(p12, [EQ_POINTS[2][1]], [EQ_POINTS[2][2]]; marker=:circle, ms=8, color=:green, label="R3 eq")
xlabel!(p12, "v")
ylabel!(p12, "u")

# (1,3) Derivative residual
p13 = plot(t, resid_dir; lw=1.0, color=:red, label="||dX_true − dX_SR3||",
           title="Direct SINDy — Derivative residual", common...)
add_regime_lines!(p13)
xlabel!(p13, "t")
ylabel!(p13, "Residual norm")

# ── Row 2: UDE+SR3 ──────────────────────────────────────────────────
# (2,1) Time series
p21 = plot(t, Xc[1, :]; lw=1.2, color=:black, label="True v", title="UDE+SINDy — Time series", common...)
plot!(p21, t, Xc[2, :]; lw=1.2, color=:black, linestyle=:dash, label="True u")
plot!(p21, t_ude_roll, X_ude[1, :]; lw=1.2, color=:red, label="UDE+SR3 v")
plot!(p21, t_ude_roll, X_ude[2, :]; lw=1.2, color=:red, linestyle=:dash, label="UDE+SR3 u")
add_regime_lines!(p21)
ylabel!(p21, "State")
xlabel!(p21, "t")

# (2,2) Phase portrait
p22 = plot(Xc[1, :], Xc[2, :]; lw=1.2, color=:black, label="True", title="UDE+SINDy — Phase portrait", common...)
plot!(p22, X_ude[1, :], X_ude[2, :]; lw=1.2, color=:red, label="UDE+SR3")
scatter!(p22, [EQ_POINTS[1][1]], [EQ_POINTS[1][2]]; marker=:circle, ms=8, color=:blue, label="R1 eq")
scatter!(p22, [EQ_POINTS[2][1]], [EQ_POINTS[2][2]]; marker=:circle, ms=8, color=:green, label="R3 eq")
xlabel!(p22, "v")
ylabel!(p22, "u")

# (2,3) Derivative residual
p23 = plot(t, resid_ude; lw=1.0, color=:red, label="||dX_true − dX_UDE||",
           title="UDE+SINDy — Derivative residual", common...)
add_regime_lines!(p23)
xlabel!(p23, "t")
ylabel!(p23, "Residual norm")

# ── Combine ──────────────────────────────────────────────────────────
fig = plot(p11, p12, p13, p21, p22, p23;
           layout=(2, 3), size=(1800, 1000), dpi=200,
           left_margin=5Plots.mm, bottom_margin=5Plots.mm)

outpath = joinpath(FIGDIR, "both_methods_nonstationary.png")
savefig(fig, outpath)
@info "Saved $outpath"

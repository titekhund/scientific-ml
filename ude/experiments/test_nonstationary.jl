# experiments/test_nonstationary.jl
#
# Integration tests for the nonstationary Goodwin pipeline.
# Runs 7 checks, prints PASS/FAIL for each, and fixes any issues found.
#
# Run with:  julia --project=. experiments/test_nonstationary.jl
#
# Expected runtime: ~2–5 min (Test 5 fits SINDy on 1001 points)

include(joinpath(@__DIR__, "..", "src", "ScientificML.jl"))
using .ScientificML

include(joinpath(@__DIR__, "..", "src", "simulate_nonstationary.jl"))

using LinearAlgebra
using Statistics

# ─────────────────────────────────────────────────────────────────────────────
# Test harness
# ─────────────────────────────────────────────────────────────────────────────

const RESULTS = Pair{String, Bool}[]

function check(name::String, cond::Bool, detail::String = "")
    push!(RESULTS, name => cond)
    marker = cond ? "PASS ✓" : "FAIL ✗"
    msg    = isempty(detail) ? "" : "  ($detail)"
    println("    [$marker] $name$msg")
    return cond
end

# ─────────────────────────────────────────────────────────────────────────────
# Shared setup
# ─────────────────────────────────────────────────────────────────────────────

x0_true, regime_configs, breakpoints = nonstationary_goodwin_configs()
SAVEAT = 0.5

# ═════════════════════════════════════════════════════════════════════════════
println("\n" * "═"^60)
println("Test 1 — Parameter ordering and equilibria")
println("═"^60)
# ═════════════════════════════════════════════════════════════════════════════
#
# Ground truth: systems.jl line 13 reads
#   eta1, theta1, eta2, theta2 = p
# so p = [η1, θ1, η2, θ2].
# The user's earlier description of a γ/δ swap was INCORRECT —
# systems.jl and simulate_nonstationary.jl both use the SAME ordering.
# No parameter swap exists; the code is correct as written.

p_r1 = regime_configs[1].p     # autonomous R1
p_r2 = regime_configs[2].p     # forced R2 (base params = R1)
p_tr = regime_configs[3].p     # damped transition (base params = R3)
p_r3 = regime_configs[4].p     # autonomous R3

# Expected values (matching Python reference and systems.jl convention)
p_r1_expected = [0.05, 0.10, 0.05, 0.10]   # [η1, θ1, η2, θ2]
p_r3_expected = [0.04, 0.12, 0.05, 0.10]

println("  R1 params: $(p_r1)  (expected: $p_r1_expected)")
println("  R3 params: $(p_r3)  (expected: $p_r3_expected)")

check("R1 parameter vector",  p_r1 == p_r1_expected,
      "got $(p_r1)")
check("R2 base params == R1", p_r2[1:4] == p_r1_expected,
      "got $(p_r2[1:4])")
check("R3 parameter vector",  p_r3 == p_r3_expected,
      "got $(p_r3)")
check("Transition base params == R3", p_tr[1:4] == p_r3_expected,
      "got $(p_tr[1:4])")

# Equilibria: v* = η2/θ2 = p[3]/p[4],  u* = η1/θ1 = p[1]/p[2]
eq_r1_v = p_r1[3] / p_r1[4]   # η2/θ2 = 0.5
eq_r1_u = p_r1[1] / p_r1[2]   # η1/θ1 = 0.5
eq_r3_v = p_r3[3] / p_r3[4]   # η2/θ2 = 0.5
eq_r3_u = p_r3[1] / p_r3[2]   # η1/θ1 ≈ 0.3333

println("  R1 equilibrium: v*=$(round(eq_r1_v, digits=4))  u*=$(round(eq_r1_u, digits=4))")
println("  R3 equilibrium: v*=$(round(eq_r3_v, digits=4))  u*=$(round(eq_r3_u, digits=6))")
println("  Transition target stored in p[6:7]: v*=$(p_tr[6])  u*=$(round(p_tr[7], digits=6))")

check("R1 equilibrium v* = 0.5",          abs(eq_r1_v - 0.5) < 1e-10)
check("R1 equilibrium u* = 0.5",          abs(eq_r1_u - 0.5) < 1e-10)
check("R3 equilibrium v* = 0.5",          abs(eq_r3_v - 0.5) < 1e-10)
check("R3 equilibrium u* ≈ 1/3",          abs(eq_r3_u - 1/3) < 1e-9)
check("Transition target v* = R3 eq v*",  abs(p_tr[6] - eq_r3_v) < 1e-10)
check("Transition target u* = R3 eq u*",  abs(p_tr[7] - eq_r3_u) < 1e-10)

# ═════════════════════════════════════════════════════════════════════════════
println("\n" * "═"^60)
println("Test 2 — Dynamics consistency: goodwin! ≡ goodwin_autonomous!")
println("═"^60)
# ═════════════════════════════════════════════════════════════════════════════

tspan_r1 = regime_configs[1].tspan   # (0.0, 200.0)

prob_ref   = ODEProblem(goodwin!,           x0_true, tspan_r1, p_r1)
prob_auto  = ODEProblem(goodwin_autonomous!, x0_true, tspan_r1, p_r1)

sol_ref  = solve(prob_ref,  Vern7(); abstol=1e-12, reltol=1e-12, saveat=SAVEAT)
sol_auto = solve(prob_auto, Vern7(); abstol=1e-12, reltol=1e-12, saveat=SAVEAT)

X_ref  = Array(sol_ref)
X_auto = Array(sol_auto)

max_diff = maximum(abs.(X_ref .- X_auto))
println("  max|goodwin! − goodwin_autonomous!| = $(max_diff)")

check("Same trajectory (tol 1e-10)",   max_diff < 1e-10, "max_diff=$max_diff")
check("Same number of time points",    length(sol_ref.t) == length(sol_auto.t),
      "$(length(sol_ref.t)) vs $(length(sol_auto.t))")

# ═════════════════════════════════════════════════════════════════════════════
println("\n" * "═"^60)
println("Test 3 — Full simulation sanity")
println("═"^60)
# ═════════════════════════════════════════════════════════════════════════════
#
# saveat=0.5 over [0,500] should give exactly 1001 points:
#   R1  [0,200]:  401 pts
#   R2  (200,290]: 180 pts (stripped boundary)
#   Tr  (290,300]:  20 pts
#   R3  (300,500]: 400 pts
#   Total = 1001

t_full, X_clean = simulate_nonstationary(x0_true, regime_configs; saveat=SAVEAT)
N = length(t_full)

println("  shape:       $(size(X_clean))  (expected (2, 1001))")
println("  time range:  [$(t_full[1]), $(t_full[end])]  (expected [0.0, 500.0])")
println("  N points:    $N  (expected 1001)")
println("  dt (median): $(median(diff(t_full)))")

check("Shape is (2, N)",          size(X_clean, 1) == 2)
check("N = 1001 points",          N == 1001, "got $N")
check("t[1] = 0.0",               t_full[1] == 0.0)
check("t[end] = 500.0",           t_full[end] == 500.0)
check("Uniform dt = 0.5",         all(abs.(diff(t_full) .- 0.5) .< 1e-10))

# Continuity at regime boundaries: the state should be continuous.
# In our design, segments share no duplicated boundary points in X_clean,
# but the dynamics are continuous by construction (each segment's x0 =
# previous segment's final state). Check that no large jumps occur.
#
# The gap after stripping is between t=boundary and t=boundary+0.5.
# We check the actual X difference at each boundary index.
boundary_times = breakpoints[2:end-1]   # [200.0, 290.0, 300.0]

all_continuous = true
for tb in boundary_times
    idx = findfirst(t_full .≈ tb)
    if idx === nothing
        println("  WARNING: boundary t=$tb not found on saveat grid!")
        all_continuous = false
        continue
    end
    if idx < N
        Δx = norm(X_clean[:, idx+1] - X_clean[:, idx])
        println("  Δx at t=$tb → t=$(round(t_full[idx+1], digits=1)): $(round(Δx, sigdigits=4))")
        if Δx > 0.1
            all_continuous = false
        end
    end
end
check("No large discontinuities at boundaries (Δx < 0.1)", all_continuous)

# ═════════════════════════════════════════════════════════════════════════════
println("\n" * "═"^60)
println("Test 4 — Regime orbit check (states orbit near equilibria)")
println("═"^60)
# ═════════════════════════════════════════════════════════════════════════════
#
# The Goodwin system is a conservative oscillator: over enough cycles the
# time-average of each state converges to the equilibrium. With x0=[0.7,0.5]
# (close to R1 equilibrium), the mean over R1 should be within a loose band.

masks = get_regime_masks(t_full, breakpoints)
regime_names = ["R1", "R2", "Tr", "R3"]

for i in 1:4
    idx = findall(masks[i])
    n_pts = length(idx)
    if n_pts == 0
        println("  $(regime_names[i]): no points!")
        continue
    end
    mean_v = mean(X_clean[1, idx])
    mean_u = mean(X_clean[2, idx])
    println("  $(regime_names[i]) ($(n_pts) pts): " *
            "mean v=$(round(mean_v, digits=4))  mean u=$(round(mean_u, digits=4))")
end

# R1: equilibrium (0.5, 0.5), expect mean states to be loosely close
mean_v_r1 = mean(X_clean[1, findall(masks[1])])
mean_u_r1 = mean(X_clean[2, findall(masks[1])])
mean_v_r3 = mean(X_clean[1, findall(masks[4])])
mean_u_r3 = mean(X_clean[2, findall(masks[4])])

# Loose tolerances — not a tight test, just a sanity check
check("R1 mean v ≈ 0.5 (± 0.15)", abs(mean_v_r1 - 0.5) < 0.15)
check("R1 mean u ≈ 0.5 (± 0.15)", abs(mean_u_r1 - 0.5) < 0.15)
check("R3 mean v ≈ 0.5 (± 0.15)", abs(mean_v_r3 - 0.5) < 0.15)
check("R3 mean u ≈ 1/3 (± 0.15)", abs(mean_u_r3 - 1/3) < 0.15)

# ═════════════════════════════════════════════════════════════════════════════
println("\n" * "═"^60)
println("Test 5 — SINDy fit + rollout")
println("═"^60)
# ═════════════════════════════════════════════════════════════════════════════
#
# Separates the fit step from the rollout step so that:
#   (a) equations are printed even when the rollout diverges, and
#   (b) NaN/Inf in the rollout is detected across the full array, not just
#       the first N columns.
# Uses a reduced λ grid for speed; production runs use the full 13-value grid.

println("  Fitting SINDy on $(N)-point clean trajectory…  (may take ~1–3 min)")
rng_test    = ScientificML.StableRNGs.StableRNG(42)
λs_test     = exp10.(-6:1.0:0)   # 7 values (faster than the full 13-value grid)
prox        = ScientificML.DataDrivenSparse.SoftThreshold()

fit_ok      = false
rollout_ok  = false
Xhat_test   = nothing
n_hat_int   = 0
has_nans    = false
res_test    = nothing
sys_test    = nothing

# ── Step A: fit ──────────────────────────────────────────────────────────────
try
    res_test, sys_test, _ = fit_sindy_sr3(
        t_full, X_clean;
        polyorder = 3, λs = λs_test, nu = 1.0, proximal = prox, rng = rng_test
    )
    fit_ok = true
    println("  Fit succeeded.  retcode = $(res_test.retcode)")
catch e
    println("  Fit failed: ", e)
end

# ── Print discovered equations and retcode (independent of rollout) ──────────
if fit_ok && sys_test !== nothing
    try
        eqs = ScientificML.ModelingToolkit.equations(sys_test)
        println("  eq_v: $(eqs[1].rhs)")
        println("  eq_u: $(eqs[2].rhs)")
    catch
        println("  (could not extract equation strings)")
    end
    # Best lambda: not directly exposed in the public DataDrivenDiffEq API;
    # log what we can access from the result metadata.
    try
        println("  result alg: $(res_test.alg)")
    catch
        println("  best_lambda: (not accessible from result)")
    end
end

# ── Step B: rollout ──────────────────────────────────────────────────────────
if fit_ok
    try
        Xhat_test  = predict_sindy(res_test, X_clean[:, 1], t_full)
        n_hat_int  = size(Xhat_test, 2)
        has_nans   = any(isnan, Xhat_test) || any(isinf, Xhat_test)
        rollout_ok = true
        nan_cols   = has_nans ? count(j -> any(isnan, Xhat_test[:, j]) || any(isinf, Xhat_test[:, j]),
                                      1:n_hat_int) : 0
        println("  Rollout shape: $(size(Xhat_test))  NaN/Inf: $has_nans" *
                (has_nans ? "  ($nan_cols/$n_hat_int columns affected)" : ""))
    catch e
        println("  Rollout failed (ODE diverged or solver error): ", typeof(e), " — ", e)
    end
end

check("SINDy fit completed",               fit_ok)
check("Rollout succeeded (no exception)",  rollout_ok)
check("Rollout length = $N",
      rollout_ok && n_hat_int == N,
      rollout_ok ? "got $n_hat_int" : "rollout did not complete")
check("Rollout has 2 state rows",          rollout_ok && size(Xhat_test, 1) == 2)
check("Rollout is finite (no NaN/Inf)",
      rollout_ok && !has_nans,
      has_nans ? "model diverged on 500-unit horizon (expected at high noise or poor fit)" :
                 "clean")

# ═════════════════════════════════════════════════════════════════════════════
println("\n" * "═"^60)
println("Test 6 — Regime masks (exhaustive + exclusive)")
println("═"^60)
# ═════════════════════════════════════════════════════════════════════════════

n_regime = length(masks)

# Exhaustive: every time index is in exactly one mask
covered   = zeros(Int, N)
for m in masks
    covered .+= Int.(m)
end

check("All $N points covered exactly once", all(covered .== 1),
      "min=$(minimum(covered))  max=$(maximum(covered))")
check("Number of regimes = 4", n_regime == 4)

# Sizes match expectations for saveat=0.5 over [0,500]
expected_sizes = [400, 180, 20, 401]
sizes = sum.(masks)
println("  Regime sizes: $(sizes)  (expected: $expected_sizes)")
check("R1 size = 400",  sizes[1] == expected_sizes[1])
check("R2 size = 180",  sizes[2] == expected_sizes[2])
check("Tr size = 20",   sizes[3] == expected_sizes[3])
check("R3 size = 401",  sizes[4] == expected_sizes[4])

# Boundary assignments
t200_idx = findfirst(t_full .≈ 200.0)
t290_idx = findfirst(t_full .≈ 290.0)
t300_idx = findfirst(t_full .≈ 300.0)

println("  t=200 in R2 mask: $(t200_idx !== nothing && masks[2][t200_idx])")
println("  t=290 in Tr mask: $(t290_idx !== nothing && masks[3][t290_idx])")
println("  t=300 in R3 mask: $(t300_idx !== nothing && masks[4][t300_idx])")

check("t=200.0 assigned to R2",
      t200_idx !== nothing && masks[2][t200_idx])
check("t=290.0 assigned to Tr",
      t290_idx !== nothing && masks[3][t290_idx])
check("t=300.0 assigned to R3",
      t300_idx !== nothing && masks[4][t300_idx])

# ═════════════════════════════════════════════════════════════════════════════
println("\n" * "═"^60)
println("Test 7 — evaluate_true_derivatives")
println("═"^60)
# ═════════════════════════════════════════════════════════════════════════════

dX = evaluate_true_derivatives(t_full, X_clean, regime_configs, breakpoints)

check("dX shape matches X_clean", size(dX) == size(X_clean))

# Spot check at t=0: x=[0.7, 0.5], R1 params [0.05, 0.10, 0.05, 0.10]
#   dv/dt = (η1 - θ1*u)*v = (0.05 - 0.10*0.5)*0.7 = 0.0 * 0.7 = 0.0
#   du/dt = (-η2 + θ2*v)*u = (-0.05 + 0.10*0.7)*0.5 = 0.02 * 0.5 = 0.01
dv0_expected = 0.0
du0_expected = 0.01
dv0_actual   = dX[1, 1]
du0_actual   = dX[2, 1]

println("  At t=0:  dv/dt = $(dv0_actual)  (expected $dv0_expected)")
println("  At t=0:  du/dt = $(du0_actual)  (expected $du0_expected)")

check("dv/dt at t=0 ≈ 0.0",  abs(dv0_actual - dv0_expected) < 1e-9)
check("du/dt at t=0 ≈ 0.01", abs(du0_actual - du0_expected) < 1e-9)

# Spot check at t=300: first point of R3, uses R3 params.
# x at t=300 comes from the end of the transition regime.
# Just verify output is finite and non-zero (state-dependent).
idx300 = findfirst(t_full .≈ 300.0)
if idx300 !== nothing
    dv300 = dX[1, idx300]
    du300 = dX[2, idx300]
    println("  At t=300: dv/dt = $(round(dv300, sigdigits=4))" *
            "  du/dt = $(round(du300, sigdigits=4))  (R3 autonomous)")
    check("dX at t=300 is finite", isfinite(dv300) && isfinite(du300))
else
    println("  t=300 not found on grid")
end

# ═════════════════════════════════════════════════════════════════════════════
println("\n" * "═"^60)
println("SUMMARY")
println("═"^60)
# ═════════════════════════════════════════════════════════════════════════════

n_pass = count(last, RESULTS)
n_fail = count(!last, RESULTS)
n_total = length(RESULTS)

println("  $n_pass / $n_total passed")

if n_fail > 0
    println("\n  Failed checks:")
    for (name, ok) in RESULTS
        ok || println("    ✗  $name")
    end
else
    println("  All checks passed.")
end

println()

# Exit with non-zero code if any check failed (useful for CI)
if n_fail > 0
    exit(1)
end

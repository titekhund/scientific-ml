# src/simulate_nonstationary.jl
#
# Nonstationary multi-regime Goodwin model.
# Self-contained (not part of ScientificML module).
# Include directly in experiment/viz scripts after loading ScientificML.
#
# Parameter convention (matches systems.jl and Python reference):
#   dv/dt = (η1 - θ1·u)·v
#   du/dt = (-η2 + θ2·v)·u
# Equilibrium: v* = η2/θ2,  u* = η1/θ1

using OrdinaryDiffEq

# ─────────────────────────────────────────────────────────────────────────────
# Regime dynamics
# ─────────────────────────────────────────────────────────────────────────────

"Autonomous Goodwin.  p = [η1, θ1, η2, θ2]"
function goodwin_autonomous!(dx, x, p, t)
    eta1, theta1, eta2, theta2 = p
    v, u = x
    dx[1] = (eta1 - theta1 * u) * v
    dx[2] = (-eta2 + theta2 * v) * u
    return nothing
end

"""
Sinusoidally forced Goodwin (R2).
  η1(t) = η1 + amp·sin(freq·t)
  η2(t) = η2 + amp·cos(freq·t)
p = [η1, θ1, η2, θ2, amp, freq]
"""
function goodwin_forced!(dx, x, p, t)
    eta1, theta1, eta2, theta2, amp, freq = p
    v, u = x
    eta1_t = eta1 + amp * sin(freq * t)
    eta2_t = eta2 + amp * cos(freq * t)
    dx[1] = (eta1_t - theta1 * u) * v
    dx[2] = (-eta2_t + theta2 * v) * u
    return nothing
end

"""
Damped transition: Goodwin + exponential restoring force toward (v*, u*).
  dv/dt = (η1 - θ1·u)·v - ε·(v - v*)
  du/dt = (-η2 + θ2·v)·u - ε·(u - u*)
p = [η1, θ1, η2, θ2, ε, v*, u*]
"""
function goodwin_damped!(dx, x, p, t)
    eta1, theta1, eta2, theta2, eps_damp, v_star, u_star = p
    v, u = x
    dx[1] = (eta1 - theta1 * u) * v - eps_damp * (v - v_star)
    dx[2] = (-eta2 + theta2 * v) * u - eps_damp * (u - u_star)
    return nothing
end

# ─────────────────────────────────────────────────────────────────────────────
# Default 3-regime + transition configuration
# ─────────────────────────────────────────────────────────────────────────────

"""
    nonstationary_goodwin_configs(; kwargs...) -> (x0, configs, breakpoints)

Build the default 4-segment Goodwin configuration:
  R1  [0, 200)   — autonomous, R1 params
  R2  [200, 290) — sinusoidally forced, R1 base params
  Tr  [290, 300) — damped transition toward R3 equilibrium
  R3  [300, 500] — autonomous, R3 params

Returns
  - `x0`          : initial condition [v0, u0]
  - `configs`     : Vector of NamedTuples (tspan, f!, p)
  - `breakpoints` : [t0, t1, t2, t3, t_end]  (for regime masking)
"""
function nonstationary_goodwin_configs(;
        # Time grid
        t0   = 0.0,  t1  = 200.0, t2  = 290.0, t3  = 300.0, t_end = 500.0,
        # Initial condition
        x0   = [0.7, 0.5],
        # R1 / R2 base parameters
        eta1_r1 = 0.05, theta1_r1 = 0.10, eta2_r1 = 0.05, theta2_r1 = 0.10,
        # R3 parameters
        eta1_r3 = 0.04, theta1_r3 = 0.12, eta2_r3 = 0.05, theta2_r3 = 0.10,
        # R2 forcing
        amp      = 0.02,
        freq     = 0.2,
        # Transition damping coefficient
        eps_damp = 0.02
    )

    # R3 equilibrium: v* = η2/θ2 = 0.5,  u* = η1/θ1 ≈ 0.333
    v_star_r3 = eta2_r3 / theta2_r3
    u_star_r3 = eta1_r3 / theta1_r3

    configs = [
        # R1 — autonomous Goodwin, R1 parameters
        (tspan = (t0, t1),
         f!    = goodwin_autonomous!,
         p     = Float64[eta1_r1, theta1_r1, eta2_r1, theta2_r1]),

        # R2 — sinusoidally forced, same base params as R1
        (tspan = (t1, t2),
         f!    = goodwin_forced!,
         p     = Float64[eta1_r1, theta1_r1, eta2_r1, theta2_r1, amp, freq]),

        # Transition — damped toward R3 equilibrium using R3 base params
        (tspan = (t2, t3),
         f!    = goodwin_damped!,
         p     = Float64[eta1_r3, theta1_r3, eta2_r3, theta2_r3,
                         eps_damp, v_star_r3, u_star_r3]),

        # R3 — autonomous Goodwin, R3 parameters
        (tspan = (t3, t_end),
         f!    = goodwin_autonomous!,
         p     = Float64[eta1_r3, theta1_r3, eta2_r3, theta2_r3]),
    ]

    breakpoints = [t0, t1, t2, t3, t_end]

    return float.(x0), configs, breakpoints
end

# ─────────────────────────────────────────────────────────────────────────────
# Chained simulator
# ─────────────────────────────────────────────────────────────────────────────

"""
    simulate_nonstationary(x0, regime_configs; saveat, solver, abstol, reltol)

Simulate a multi-regime ODE by chaining sub-problems.
Each element of `regime_configs` is a NamedTuple with fields:
  - `tspan` : (t_start, t_end)
  - `f!`    : in-place dynamics `f!(dx, x, p, t)`
  - `p`     : parameter vector

The end-state of each segment seeds the next as its initial condition.
Returns `(t_full, X_full)` with `X_full` shaped `(n_states × n_time)`.
Duplicate boundary time points between regimes are removed.
"""
function simulate_nonstationary(x0, regime_configs;
        saveat = 0.5,
        solver = Vern7(),
        abstol = 1e-9,
        reltol = 1e-9
    )

    t_segs = Vector{Vector{Float64}}()
    X_segs = Vector{Matrix{Float64}}()

    x_cur = copy(float.(x0))

    for cfg in regime_configs
        prob = ODEProblem(cfg.f!, x_cur, cfg.tspan, cfg.p)
        sol  = solve(prob, solver; abstol = abstol, reltol = reltol, saveat = saveat)
        push!(t_segs, sol.t)
        push!(X_segs, Array(sol))
        x_cur = X_segs[end][:, end]   # hand off to next regime
    end

    # Concatenate, stripping the duplicated boundary point from each continuation
    t_full = t_segs[1]
    X_full = X_segs[1]
    for k in 2:length(t_segs)
        t_full = vcat(t_full, t_segs[k][2:end])
        X_full = hcat(X_full, X_segs[k][:, 2:end])
    end

    return t_full, X_full
end

# ─────────────────────────────────────────────────────────────────────────────
# Regime masks
# ─────────────────────────────────────────────────────────────────────────────

"""
    get_regime_masks(t, breakpoints) -> Vector{BitVector}

Return one BitVector per interval defined by `breakpoints`.
Intervals are half-open `[bk[i], bk[i+1])`, except the last which is
closed `[bk[end-1], bk[end]]`.

Example:
    masks = get_regime_masks(t, [0.0, 200.0, 290.0, 300.0, 500.0])
    # masks[1] → R1,  masks[2] → R2,  masks[3] → Transition,  masks[4] → R3
"""
function get_regime_masks(t, breakpoints)
    n = length(breakpoints) - 1
    masks = Vector{BitVector}(undef, n)
    for i in 1:n
        lo = breakpoints[i]
        hi = breakpoints[i + 1]
        if i == n
            masks[i] = (t .>= lo) .& (t .<= hi)   # closed final interval
        else
            masks[i] = (t .>= lo) .& (t .< hi)    # half-open
        end
    end
    return masks
end

# ─────────────────────────────────────────────────────────────────────────────
# True derivative evaluator (for derivative residual diagnostics)
# ─────────────────────────────────────────────────────────────────────────────

"""
    evaluate_true_derivatives(t, X, regime_configs, breakpoints) -> dX

Evaluate the known ground-truth ODE at each time point in the trajectory.
Returns `dX` shaped `(n_states × n_time)`.
Useful for computing derivative residuals against the discovered SINDy model.
"""
function evaluate_true_derivatives(t, X, regime_configs, breakpoints)
    masks = get_regime_masks(t, breakpoints)
    dX    = zeros(eltype(X), size(X))
    dx_buf = zeros(eltype(X), size(X, 1))

    for i in eachindex(regime_configs)
        cfg = regime_configs[i]
        for j in findall(masks[i])
            cfg.f!(dx_buf, X[:, j], cfg.p, t[j])
            dX[:, j] .= dx_buf
        end
    end
    return dX
end

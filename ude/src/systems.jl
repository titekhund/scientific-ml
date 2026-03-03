export goodwin!, goodwin_defaults, make_goodwin_problem

# -------------------------
# Goodwin (reparameterized) (2D)
# state x = [v, u]
#   v = employment rate (prey)
#   u = wage share (predator)
# p = [eta1, theta1, eta2, theta2]
#   v̇ = (eta1 - theta1*u) * v
#   u̇ = (-eta2 + theta2*v) * u
# -------------------------
function goodwin!(dx, x, p, t)
    eta1, theta1, eta2, theta2 = p
    v, u = x
    dx[1] = (eta1 - theta1 * u) * v
    dx[2] = (-eta2 + theta2 * v) * u
    return nothing
end

function goodwin_defaults()
    tspan = (0.0, 10.0)                 
    x0    = [0.92, 0.65]                # [v0, u0] (same as Lotka u0, given in the original code)
    ptrue = [1.399, 2.239, 2.57, 2.43]  # [eta1, theta1, eta2, theta2]
    return (tspan=tspan, x0=x0, ptrue=ptrue)
end

function make_goodwin_problem(; tspan=nothing, x0=nothing, ptrue=nothing)
    d = goodwin_defaults()
    tspan === nothing && (tspan = d.tspan)
    x0    === nothing && (x0    = d.x0)
    ptrue === nothing && (ptrue = d.ptrue)
    return ODEProblem(goodwin!, x0, tspan, ptrue)
end

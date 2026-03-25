# src/methods/sindy.jl
# Pure SINDy (SR3) pipeline
# Included inside `module ScientificML` — no nested module needed.

export sindy_poly_basis, fit_sindy_sr3, predict_sindy, eval_sindy_at_states

using DataDrivenDiffEq
using DataDrivenSparse
using ModelingToolkit
using StableRNGs

"""
Build a polynomial SINDy Basis for n-dimensional state x.
polyorder = highest total polynomial degree included.
"""
function sindy_poly_basis(n::Int; polyorder::Int=3)
    @variables x[1:n]
    xv = collect(x)
    h  = polynomial_basis(xv, polyorder)
    return Basis(h, xv)
end

"""
Fit SINDy using SR3 from time series X (n_states × n_time) at times ts.
- Derivatives estimated via ContinuousDataDrivenProblem(X, ts, GaussianKernel())
- SR3(λs, ν, proximal): λs is the threshold grid (Pareto search lives inside SR3)
- DataDrivenCommonOptions handles rounding, batching, etc. (not λ)
"""
function fit_sindy_sr3(ts, X;
        polyorder::Int = 3,
        λs             = exp10.(-10:0.5:0),
        nu::Real       = 1.0,
        proximal       = SoftThreshold(),   # or HardThreshold()
        batchsize::Int = 0,
        rng            = StableRNGs.StableRNG(1),
        digits::Int    = 3
    )

    n     = size(X, 1)
    basis = sindy_poly_basis(n; polyorder=polyorder)
    prob  = ContinuousDataDrivenProblem(X, ts, GaussianKernel())

    # λ grid belongs in SR3, not in DataDrivenCommonOptions
    opt = SR3(λs, nu, proximal)

    options = if batchsize > 0
        sampler = DataProcessing(
            split     = 0.8,
            shuffle   = true,
            batchsize = batchsize,
            rng       = rng
        )
        DataDrivenCommonOptions(digits=digits, data_processing=sampler)
    else
        DataDrivenCommonOptions(digits=digits)
    end

    res    = solve(prob, basis, opt; options=options)
    system = get_basis(res)

    param_map = get_parameter_map(system)
    params    = isempty(param_map) ? nothing : param_map

    return res, system, params
end


"""
Roll out (simulate) a discovered SINDy model and return Xhat (n_states × n_time).

Inputs:
- res: result returned by fit_sindy_sr3 (recommended), OR a ModelingToolkit ODESystem
- x0: initial condition vector (length = n_states)
- ts: time grid vector

Returns:
- Xhat: Array with shape (n_states, length(ts))
"""
function predict_sindy(res, x0, ts;
        solver = Vern7(),
        abstol = 1e-9,
        reltol = 1e-9
    )

    # Allow passing either the solve result or the system itself
    system = hasmethod(get_basis, Tuple{typeof(res)}) ? get_basis(res) : res

    # Map (p₁=>..., p₂=>..., ...) from the fitted system
    pmap = Dict(get_parameter_map(system))

    # Convert to numeric vector in the correct parameter order
    pvec = ModelingToolkit.varmap_to_vars(pmap, ModelingToolkit.parameters(system))

    tspan = (ts[1], ts[end])
    prob = ODEProblem(system, x0, tspan, pvec)
    sol   = solve(prob, solver; abstol=abstol, reltol=reltol, saveat=ts)
    return Array(sol)
end

"""
Evaluate the discovered SINDy RHS at each column of X.
Returns dX shaped (n_states, size(X,2)).

Extracts symbolic equations and fitted parameters from the Basis,
substitutes parameter values, then compiles pure-Julia callables via
`Symbolics.build_function`.  No ODE solve or finite differences —
the model is evaluated pointwise at the supplied states, matching
how `evaluate_true_derivatives` works for the ground truth.
"""
function eval_sindy_at_states(res, t, X)
    system = hasmethod(get_basis, Tuple{typeof(res)}) ? get_basis(res) : res

    Sym = ModelingToolkit.Symbolics

    eqs   = ModelingToolkit.equations(system)
    pmap  = get_parameter_map(system)
    svars = DataDrivenDiffEq.states(system)   # [x[1], x[2]]

    # Substitute fitted parameter values → expressions in state vars only
    param_dict = Dict(pmap)
    rhs_num    = [Sym.substitute(eq.rhs, param_dict) for eq in eqs]

    # Compile to fast Julia lambdas: f(x1, x2) -> Float64
    fs = [Sym.build_function(expr, svars...; expression = Val{false})
          for expr in rhs_num]

    n, T = size(X)
    length(fs) == n || error("SINDy discovered $(length(fs)) equations but state has $n dimensions")
    dX = zeros(n, T)
    for j in 1:T
        for k in 1:n
            dX[k, j] = fs[k](X[1, j], X[2, j])
        end
    end
    return dX
end

# src/methods/sindy.jl
# Pure SINDy (SR3) pipeline
# Included inside `module ScientificML` — no nested module needed.

export sindy_poly_basis, fit_sindy_sr3

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

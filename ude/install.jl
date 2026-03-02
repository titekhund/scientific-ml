using Pkg

# Activate the Julia environment in the current repo root
Pkg.activate(@__DIR__)

t_install = @elapsed Pkg.add([
    "OrdinaryDiffEq",
    "ModelingToolkit",
    "DataDrivenDiffEq",
    "SciMLSensitivity",
    "DataDrivenSparse",
    "Optimization",
    "OptimizationOptimisers",
    "OptimizationOptimJL",
    "LineSearches",
    "ComponentArrays",
    "Lux",
    "Zygote",
    "Plots",
    "StableRNGs",
])

Pkg.precompile()
println("Package install: $(round(t_install/60, digits=1)) minutes")

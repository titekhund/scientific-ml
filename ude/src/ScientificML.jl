module ScientificML

# =========================
# Core SciML + SINDy + UDE imports
# =========================
using OrdinaryDiffEq
using ModelingToolkit
using DataDrivenDiffEq
using SciMLSensitivity
using DataDrivenSparse

using Optimization
using OptimizationOptimisers
using OptimizationOptimJL
using LineSearches

using LinearAlgebra
using Statistics
using DelimitedFiles

using ComponentArrays
using Lux
using Zygote
using StableRNGs
using Plots

# =========================
# project files
# =========================
include("systems.jl")
include("simulate.jl")

#include(joinpath("methods", "sindy.jl"))
#include(joinpath("methods", "ude.jl"))
#include(joinpath("methods", "ude_sindy.jl")) 

#include("io.jl")      # for CSV saving
include("metrics.jl") # for comparing methods

end # module

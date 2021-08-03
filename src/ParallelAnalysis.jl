module ParallelAnalysis

using MultivariateStats, StatsFuns, Statistics
import Statistics: cov
import MultivariateStats: loadings
import Base: show
using StatsBase: sample
# Polychoric
using Trapz: trapz, @trapz
using Optim: optimize, Brent
using LinearAlgebra: LowerTriangular, diagind, Symmetric
using QuadGK: quadgk
using Distributions, DataFrames
# Plot recipes
using RecipesBase

include("Polychoric.jl")
export polycor, cov, loadings

include("RandomMatrix.jl")
export random_matrix, random_sample

include("FA.jl")
export fa

include("IRT.jl")
export generate_response, v2vv, heuristicIRT

include("PA.jl")
export parallel

export show

end


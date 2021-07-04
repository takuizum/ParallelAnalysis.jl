module ParallelAnalysis

using MultivariateStats, StatsFuns, Statistics
import Statistics: cov
import MultivariateStats: loadings
import Base: show
# Polychoric
using Trapz: trapz
using Optim: optimize, Brent
using LinearAlgebra: LowerTriangular, diagind
using QuadGK: quadgk
using Distributions, DataFrames
# Plot recipes
using RecipesBase

include("Polychoric.jl")
export polycor, cov, loadings

include("RandomMatrix.jl")
export random_matrix

include("FA.jl")
export fa

include("IRT.jl")
export generate_response, v2vv

include("PA.jl")
export parallel

export show

end


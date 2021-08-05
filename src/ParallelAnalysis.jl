module ParallelAnalysis

using MultivariateStats, StatsFuns, Statistics, ProgressMeter
import Statistics: cov, mean, std
import MultivariateStats: loadings
import Base: show
using StatsBase: sample
# Polychoric
using Trapz: trapz#, @trapz
using Optim: optimize, Brent
using LinearAlgebra: LowerTriangular, diagind, Symmetric, Diagonal
using QuadGK: quadgk
using Distributions, DataFrames
# Plot recipes
using RecipesBase

include("Polychoric.jl")
export polycor, cov, loadings, communalities

include("RandomMatrix.jl")
export random_matrix, random_sample

include("FA.jl")
export fa, factorscores, Bartllet, BayesMean

include("IRT.jl")
export generate_response, v2vv, heuristicIRT

include("PA.jl")
export parallel

include("MPA.jl")
export modified_parallel

export show

end


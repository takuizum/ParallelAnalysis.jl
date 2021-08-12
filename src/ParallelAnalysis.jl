module ParallelAnalysis

using MultivariateStats, StatsFuns, Statistics, ProgressMeter
import Statistics: cov
import MultivariateStats: loadings
import Base: show
using StatsBase: sample
# Polychoric
using Trapz: trapz, @trapz
using Cuba: cuhre
using QuadGK: quadgk
using Optim: optimize, Brent, NelderMead, Fminbox
using LinearAlgebra # : LowerTriangular, diagind, Symmetric
using Distributions, DataFrames
using Roots: find_zero
using ForwardDiff
import Distributions: cdf, pdf
import Statistics: quantile
# Plot recipes
using RecipesBase

include("Polychoric.jl")
export polycor, cov, loadings, communalities

include("RandomMatrix.jl")
export random_matrix, random_sample

include("FA.jl")
export fa

include("IRT.jl")
export generate_response, v2vv, heuristicIRT

include("PA.jl")
export parallel

include("SkewNormalPolychoric.jl")
export snpolycor

include("AlphaSkewNormal.jl")
export asnpolycor

export show
export cdf, pdf, quantile

end


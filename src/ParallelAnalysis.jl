module ParallelAnalysis

using MultivariateStats, StatsFuns, Statistics
# Polychor
using Trapz: trapz
using Optim: optimize, Brent
using LinearAlgebra: LowerTriangular, diagind
using QuadGK: quadgk
using Distributions, DataFrames

include("Polychoric.jl")
export polycor


end


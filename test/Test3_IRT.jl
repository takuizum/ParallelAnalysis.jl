using ParallelAnalysis
using Random, Distributions, LinearAlgebra

Random.seed!(1234)
a = rand(Uniform(0.5, 2.0), 30)
b = [sort(rand(Uniform(-3, 3), 4); rev = false) for i in 1:30]
θ = rand(Normal(0, 1), 3000)

resp = generate_response(θ, a, b)
heuIRT = @time heuristicIRT(resp; method = :em)
@code_warntype heuristicIRT(resp)
heuIRT.a
heuIRT.d
heuIRT.b


test = polycor(resp)
ParallelAnalysis.replace_diagonal!(test)
eigvals(test)
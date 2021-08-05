using Random, Distributions, LinearAlgebra
using StatsPlots

Random.seed!(1234)
a = rand(Uniform(0.5, 2.0), 30)
b = [sort(rand(Uniform(-3, 3), 4); rev = false) for i in 1:30]
θ = rand(Normal(0, 1), 3000)

resp = generate_response(θ, a, b)
heuIRT = @time heuristicIRT(resp; method = :em)
@code_warntype heuristicIRT(resp; method = :em)
heuIRT.a
heuIRT.d
heuIRT.b

scatter(a, heuIRT.a)
scatter(vcat(b...), vcat(heuIRT.b...))

new_resp = generate_response(heuIRT, BayesMean())
@code_warntype generate_response(heuIRT, BayesMean())


mpa_sim = modified_parallel(resp, 100, BayesMean())

mpa_sim.real
mpa_sim.simulated
mpa_sim.simulated_bounds

plot(mpa_sim)
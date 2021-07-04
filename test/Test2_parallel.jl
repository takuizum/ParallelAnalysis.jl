using ParallelAnalysis
using Random, Distributions
using StatsPlots

Random.seed!(1234)
a = rand(Uniform(0.5, 2.0), 30)
b = [sort(rand(Uniform(-3, 3), 4); rev = false) for i in 1:30]
θ = rand(Normal(0, 1), 3000)

resp = generate_response(θ, a, b)

fit1 = @profview fa(resp; method = :Polychoric)
fit2 = fa(resp; method = :Pearson)

par_fit = parallel(resp, 20, x -> fa(x; method = :Polychoric))

plot(par_fit)
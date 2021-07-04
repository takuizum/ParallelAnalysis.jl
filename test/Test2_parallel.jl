using ParallelAnalysis
using Random, Distributions, LinearAlgebra
using StatsPlots

Random.seed!(1234)
a = rand(Uniform(0.5, 2.0), 30)
b = [sort(rand(Uniform(-3, 3), 4); rev = false) for i in 1:30]
θ = rand(Normal(0, 1), 3000)

resp = generate_response(θ, a, b)

fit1 = fa(resp; method = :Polychoric)
fit2 = fa(resp; method = :Pearson)
@code_warntype fa(resp; method = :Polychoric)

cov(fit1) |> diagind
test = cov(fit1)
test[diagind(test)]
maximum(test[collect(offdiag(test))])

par_fit = parallel(resp, 10000, x -> fa(x; method = :Polychoric))


plot(par_fit)
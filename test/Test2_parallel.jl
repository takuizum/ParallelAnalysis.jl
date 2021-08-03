using Distributions: LinearAlgebra
using ParallelAnalysis
using Random, Distributions, LinearAlgebra
using StatsPlots

Random.seed!(1234)
a = rand(Uniform(0.5, 2.0), 30)
b = [sort(rand(Uniform(-3, 3), 4); rev = false) for i in 1:30]
θ = rand(Normal(0, 1), 3000)

resp = generate_response(θ, a, b)

@time fit1 = fa(resp; method = :em)
@time fit2 = fa(resp; cor_method = :Pearson, method = :cm)
@code_warntype fa(resp; method = :em)
@profview fa(resp; method = :em)

cov(fit1) |> diagind
test = cov(fit1)
test[diagind(test)]
maximum(test[collect(offdiag(test))])

par_fit = parallel(resp, 20, x -> fa(x; cor_method = :Polychoric))
@code_warntype parallel(resp, 10, x -> fa(x; cor_method = :Polychoric))
@profview parallel(resp, 10, x -> fa(x; cor_method = :Polychoric))
@code_warntype parallel(resp, 100, x -> fa(x; cor_method = :Pearson))

plot(par_fit)

par_fit = parallel(X, 20, x -> fa(x; cor_method = :Polychoric))
plot(par_fit)
par_fit.bounds
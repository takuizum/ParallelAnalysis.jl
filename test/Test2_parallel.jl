using StatsPlots, Distributions, MultivariateStats, LinearAlgebra

@testset "fa" begin
    Random.seed!(1234)
    a = rand(Uniform(0.5, 2.0), 30)
    b = [sort(rand(Uniform(-3, 3), 4); rev = false) for i in 1:30]
    θ = rand(Normal(0, 1), 3000)
    resp = generate_response(θ, a, b)
    
    fit1 = fa(resp; method = :em)
    @test all(loadings(fit1) .≤ 1.0)
    fit2 = fa(resp; method = :cm)
    @test all(loadings(fit2) .≤ 1.0)
    fit3 = fa(resp; cor_method = :Pearson, method = :em)
    @test all(loadings(fit3) .≤ 1.0)
    # @code_warntype factorscores(fit1, Bartllet())
    # @code_warntype factorscores(fit1, BayesMean())
    # @code_warntype ParallelAnalysis.fsm1(fit1)
    # @code_warntype ParallelAnalysis.fsm2(fit1)
    # @code_warntype fa(resp; method = :em)
    # @profview fa(resp; method = :em)
end

@testset "parallel" begin
    Random.seed!(1234)
    a = rand(Uniform(0.5, 2.0), 30)
    b = [sort(rand(Uniform(-3, 3), 4); rev = false) for i in 1:30]
    θ = rand(Normal(0, 1), 3000)
    resp = generate_response(θ, a, b)
    
    par_fit1 = parallel(resp, 10, x -> fa(x; cor_method = :Polychoric))
    @test ParallelAnalysis.findnfactors(par_fit1.FA.real, par_fit1.FA.resampled) == 1
    @test ParallelAnalysis.findnfactors(par_fit1.PCA.real, par_fit1.PCA.resampled) == 1
    par_fit2 = parallel(resp, 10, x -> fa(x; cor_method = :Pearson))
    @test ParallelAnalysis.findnfactors(par_fit2.FA.real, par_fit2.FA.resampled) == 1
    @test ParallelAnalysis.findnfactors(par_fit2.PCA.real, par_fit2.PCA.resampled) == 1
    # @code_warntype parallel(resp, 10, x -> fa(x; cor_method = :Polychoric))
    # @profview parallel(resp, 10, x -> fa(x; cor_method = :Polychoric))
    # @code_warntype parallel(resp, 100, x -> fa(x; cor_method = :Pearson))

    plot(par_fit1.PCA)
    plot(par_fit1.FA)
    plot(par_fit1)
    plot(par_fit1, markershape = :none)
end

using Distributions, LinearAlgebra
# Support function

@testset "polycor" begin
    function discretize(x, th)
        sum(th .< x)
    end
    # Gen sim data.
    Random.seed!(1234)
    raw = rand(MvNormal([0, 0], [1 .5; .5 1]), 10000)'
    τ1 = sort!(rand(Uniform(-2, 2), 3))
    τ2 = sort!(rand(Uniform(-2, 2), 4))
    obs = [discretize.(raw[:, 1], Ref(τ1)) discretize.(raw[:, 2], Ref(τ2)) ]
    mat = polycor(obs)
    @test mat isa Matrix
    ctb = ParallelAnalysis.contab(obs[:, 1], obs[:, 2])
    @test isapprox(sum(ctb.X, dims = 2), ctb.mx)
    @test isapprox(sum(ctb.X, dims = 1)[:], ctb.my)
    plc = ParallelAnalysis.polyc(ctb)
    @test 0.45 ≤ plc.ρ ≤ 0.55
    loss = ParallelAnalysis.loss(ctb, plc.τ₁, plc.τ₂, MvNormal([0, 0], [1.0 0.5; 0.5 1.0]))
    @test loss isa Real
end


@testset "Polychoric - matrix checking" begin
    
    Random.seed!(1234)
    a = rand(Uniform(0.5, 2.0), 30)
    b = [sort(rand(Uniform(-3, 3), 1); rev = false) for i in 1:30]
    θ = rand(Normal(0, 1), 3000)
    resp = generate_response(θ, a, b)

    M = polycor(resp)

    @test isposdef(M)

end

@testset "Polychoric - simulation" begin

    # Ordered categorical variables
    μ, σ = ParallelAnalysis.simulate_polychoric(100, 0.5)
    @test μ - 2σ ≤ 0.5 ≤ μ + 2σ 
    μ, σ = ParallelAnalysis.simulate_polychoric(100, 0.9)
    @test μ - 2σ ≤ 0.9 ≤ μ + 2σ 
    μ, σ = ParallelAnalysis.simulate_polychoric(100, 0.2)
    @test μ - 2σ ≤ 0.2 ≤ μ + 2σ 

    # Dichoromous variables
    μ, σ = ParallelAnalysis.simulate_polychoric(100, 0.5; cats = (1,1))
    @test μ - 2σ ≤ 0.5 ≤ μ + 2σ 
    μ, σ = ParallelAnalysis.simulate_polychoric(100, 0.9; cats = (1,1))
    @test μ - 2σ ≤ 0.9 ≤ μ + 2σ 
    μ, σ = ParallelAnalysis.simulate_polychoric(100, 0.2; cats = (1,1))
    @test μ - 2σ ≤ 0.2 ≤ μ + 2σ 

end

# # Check type stability
# test = ParallelAnalysis.contab(obs[:, 1], obs[:, 2])
# isapprox(sum(test.X, dims = 2), test.mx)
# sum(test.X, dims = 1)[:] == test.my
# @time test2 = ParallelAnalysis.polyc(test)
# ParallelAnalysis.loss(test, test2.τ₁, test2.τ₂, MvNormal([0, 0], [1.0 0.5; 0.5 1.0]))
# @profview polycor(obs)

# @code_warntype ParallelAnalysis.contab(obs[:, 1], obs[:, 2])
# @code_warntype ParallelAnalysis.polyc(ParallelAnalysis.contab(obs[:, 1], obs[:, 2]))
# @code_warntype ParallelAnalysis.loss(test, test2.τ₁, test2.τ₂, MvNormal([0, 0], [1.0 0.5; 0.5 1.0]))
# @code_warntype polycor(obs)
# @code_warntype ParallelAnalysis.ppdf(MvNormal([0, 0], [1 0.5; 0.5 1]),[0.0, 0.0])
# @code_warntype ParallelAnalysis.H(-3, -3, 3, 3, MvNormal([0, 0], [1 0.5; 0.5 1]))
# @profview ParallelAnalysis.H(-3, -3, 3, 3, MvNormal([0, 0], [1 0.5; 0.5 1]))


# # Containing missing response
# o1 = vcat(fill(missing, 10000), obs[:, 1])
# o2 = vcat(obs[:, 2], fill(missing, 10000))
# ParallelAnalysis.contab(o1, o2).X

# ParallelAnalysis.contab(obs[:, 1], obs[:, 2])
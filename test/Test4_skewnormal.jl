using Distributions, LinearAlgebra

# @testset "snpolycor" begin
    function discretize(x, th)
        sum(th .< x)
    end
    # Gen sim data.
    Random.seed!(1234)
    raw = rand(MvNormal([0, 0], [1 .8; .8 1]), 10000)'
    τ1 = sort!(rand(Uniform(-2, 2), 1))
    τ2 = sort!(rand(Uniform(-2, 2), 1))
    obs = [discretize.(raw[:, 1], Ref(τ1)) discretize.(raw[:, 2], Ref(τ2)) ]
    mat = @time snpolycor(obs)
    mat = @time polycor(obs)
    @test mat isa Matrix
    ctb = ParallelAnalysis.contab(obs[:, 1], obs[:, 2])
    plc = ParallelAnalysis.snpolyc(ctb)
    @test 0.45 ≤ plc.ρ ≤ 0.55
# end


# @code_warntype snpolycor(obs)
# @profview snpolycor(obs)
# @code_warntype ParallelAnalysis.snpolyc(ctb)
# snd = SkewNormal(0.0, 1.0, ParallelAnalysis.marginalparameters(ParallelAnalysis.BivariateSkewNormal([.0, .0], [1.0 0.5; 0.5 1.0], [0.1, 0.1]), 1))

@testset "SkewNormalPolychoric - matrix checking" begin
    
    Random.seed!(1234)
    a = rand(Uniform(0.5, 2.0), 30)
    b = [sort(rand(Uniform(-3, 3), 3); rev = false) for i in 1:30]
    θ = rand(Normal(0, 1), 3000)
    resp = generate_response(θ, a, b)

    M = snpolycor(resp)

    @test isposdef(M)

end
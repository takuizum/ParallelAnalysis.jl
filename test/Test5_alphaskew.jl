using Distributions, LinearAlgebra

# AlphaSkewNormal

using StatsPlots

d = ParallelAnalysis.AlphaSkewNormal([0.0, 0.0], [1.0 0.0; 0.0 1.0], [0.0, 0.0])
@code_warntype pdf(d, [0.0, 0.0])
contourf([pdf(ParallelAnalysis.AlphaSkewNormal([0.0, 0.0], [1.0 0.0; 0.0 1.0], [0.0, 0.0]), [x, y]) for x in -4:0.1:4, y in -4:0.1:4]')
contourf([pdf(ParallelAnalysis.AlphaSkewNormal([0.0, 0.0], [1.0 0.8; 0.8 1.0], [10.0, 10.0]), [x, y]) for x in -4:0.1:4, y in -4:0.1:4]')

plot(x -> ParallelAnalysis.mpdf(ParallelAnalysis.AlphaSkewNormal([0.0, 0.0], [1.0 0.1; 0.1 1.0], [10.0, 10.0]), x, 1))
cdf(d, 0.1, 1)
quantile(d, 1.0, 1)
quantile(d, 0.5, 1)
quantile(d, 0.2, 1)
quantile(d, 0.0, 1)
@time cdf(d, [-8.0, -8.0])
@time cdf(d, [8.0, 8.0])
@time cdf(d, [-0.11086775718163666, 1.3060639401191874])
@time cdf(d, [0.00084447681067934, 1.3060639401191874])
@time ParallelAnalysis.H(0.0, -4.0, 0.0, -4.0, d)
@time ParallelAnalysis.H(Inf, 1, Inf, 3, d)
@time ParallelAnalysis.H(1, -Inf, 0, -Inf, d) # NaN
@time ParallelAnalysis.H(4, -3, 1, 0, d)

# @testset "snpolycor" begin
    function discretize(x, th)
        sum(th .< x)
    end
    # Gen sim data.
    Random.seed!(1234)
    raw = rand(MvNormal([0, 0], [1 .3; .3 1]), 10000)'
    τ1 = sort!(rand(Uniform(-2, 2), 3))
    τ2 = sort!(rand(Uniform(-2, 2), 4))
    obs = [discretize.(raw[:, 1], Ref(τ1)) discretize.(raw[:, 2], Ref(τ2)) ]
    mat = @time asnpolycor(obs)
    mat = @time polycor(obs)
    @test mat isa Matrix
    ctb = ParallelAnalysis.contab(obs[:, 1], obs[:, 2])
    plc = ParallelAnalysis.asnpolyc(ctb)
    @test 0.45 ≤ plc.ρ ≤ 0.55
# end
cumx = [0.0; cumsum(ctb.mx)[1:end-1]; 1.0]
cumy = [0.0; cumsum(ctb.my)[1:end-1]; 1.0]
d = ParallelAnalysis.AlphaSkewNormal([0.0, 0.0], [1.0 0.0; 0.0 1.0], [0.0, 0.0])
ParallelAnalysis.loss(ctb, cumx, cumy, d)
@code_warntype ParallelAnalysis.loss(ctb, cumx, cumy, d)
using Optim
@code_warntype optimize(θ -> ParallelAnalysis.loss(ctb, cumx, cumy, ParallelAnalysis.AlphaSkewNormal([.0, .0], [1.0 θ[1]; θ[1] 1.0], [θ[2], θ[3]])), [-1.0, -Inf, -Inf], [1.0, Inf, Inf], [0.0, 0.0, 0.0], Fminbox(NelderMead()))
@time opt = optimize(θ -> ParallelAnalysis.loss(ctb, cumx, cumy, ParallelAnalysis.AlphaSkewNormal([.0, .0], [1.0 θ[1]; θ[1] 1.0], [θ[2], θ[3]])), [-1.0, -Inf, -Inf], [1.0, Inf, Inf], [0.0, 0.0, 0.0], Fminbox(NelderMead()))
@time opt1 = optimize(θ -> ParallelAnalysis.loss(ctb, cumx, cumy, ParallelAnalysis.AlphaSkewNormal([.0, .0], [1.0 θ; θ 1.0], [0.0, 0.0])), -1.0, 1.0, Brent())
@time opt2 = optimize(θ -> ParallelAnalysis.loss(ctb, cumx, cumy, ParallelAnalysis.AlphaSkewNormal([.0, .0], [1.0 opt1.minimizer[1]; opt1.minimizer[1] 1.0], θ)), [-Inf, -Inf], [Inf, Inf], [0.0, 0.0], Fminbox(NelderMead()))
@code_warntype ParallelAnalysis.asnpolyc(ctb)
@time ParallelAnalysis.asnpolyc(ctb)

using StatsPlots
plot([ParallelAnalysis.loss(ctb, cumx, cumy, ParallelAnalysis.AlphaSkewNormal([.0, .0], [1.0 x; x 1.0], [0.0, 0.0])) for x in -0.7:0.01:0.7])
contourf([ParallelAnalysis.loss(ctb, cumx, cumy, ParallelAnalysis.AlphaSkewNormal([.0, .0], [1.0 .5; .5 1.0], [x, y])) for x in -4:1.:4, y in -4:1.:4])
[ParallelAnalysis.loss(ctb, cumx, cumy, ParallelAnalysis.AlphaSkewNormal([.0, .0], [1.0 .2; .2 1.0], [x, y])) for x in -4:1.:4, y in -4:1.:4]
ParallelAnalysis.loss(ctb, cumx, cumy, ParallelAnalysis.AlphaSkewNormal([.0, .0], [1.0 .2; .2 1.0], [4., 4.]))


@testset "SkewNormalPolychoric - matrix checking" begin
    
    Random.seed!(1234)
    a = rand(Uniform(0.5, 2.0), 30)
    b = [sort(rand(Uniform(-3, 3), 3); rev = false) for i in 1:30]
    θ = rand(Normal(0, 1), 3000)
    resp = generate_response(θ, a, b)

    M = snpolycor(resp)

    @test isposdef(M)

end
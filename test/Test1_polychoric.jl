using Distributions, ParallelAnalysis
# Support function
function discretize(x, th)
    sum(th .< x)
end
# Gen sim data.
raw = rand(MvNormal([0, 0], [1 .5; .5 1]), 10000)'
τ1 = sort!(rand(Uniform(-2, 2), 3))
τ2 = sort!(rand(Uniform(-2, 2), 4))
obs = [discretize.(raw[:, 1], Ref(τ1)) discretize.(raw[:, 2], Ref(τ2)) ]

@testset "polycor" begin
    @test polycor(obs) isa Matrix
end


# Check type stability
@code_warntype ParallelAnalysis.contab(obs[:, 1], obs[:, 2])
@code_warntype ParallelAnalysis.polyc(ParallelAnalysis.contab(obs[:, 1], obs[:, 2]))
@code_warntype ParallelAnalysis.loss(test, test2.τ₁, test2.τ₂, MvNormal([0, 0], [1.0 0.5; 0.5 1.0]))
@code_warntype polycor(obs)

test = ParallelAnalysis.contab(obs[:, 1], obs[:, 2])
test2 = ParallelAnalysis.polyc(test)
ParallelAnalysis.loss(test, test2.τ₁, test2.τ₂, MvNormal([0, 0], [1.0 0.5; 0.5 1.0]))
polycor(obs)

# Containing missing response
o1 = vcat(fill(missing, 10000), obs[:, 1])
o2 = vcat(obs[:, 2], fill(missing, 10000))
ParallelAnalysis.contab(o1, o2).X

ParallelAnalysis.contab(obs[:, 1], obs[:, 2])
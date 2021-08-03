using ParallelAnalysis
using Test, Random

@testset "ParallelAnalysis.jl" begin
    include("Test1_polychoric.jl")
    include("Test2_parallel.jl")
end

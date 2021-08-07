
abstract type ParallelType end
struct Parallel{T1<:AbstractVector, T2<:AbstractVector, T3 <: Symbol, T4<: Real} <: ParallelType
    real::T1
    simulated::T1
    simulated_bounds::T2
    resampled::T1
    resampled_bounds::T2
    iter::T4
    reduction_method::T3
    correlation_method::T3
end

struct ParallelSets{T <: ParallelType}
    FA::T
    PCA::T
end

"""
    parallel(data, niter, f = fa)
Parallel Analysis.

# Arguments
- `data`
- `niter` is a size of simulation that generate (normal) random data matrix, calculate R, correlation matrix and its eigen values.
- `f` Function for dimension reduction. Default is `fa`.

# Values
- `real` Eigen values from read data.
- `simulated`, `resampled` Mean of eigen values from simulated data.
- `simulated_bound`, `resampled_bounds` 2.5th and 97.5th perceintile rank values of simulated eigen values.
- `iter` Number of iterations (of the simulation).
- `reduncion_method`
- `correlation_method`


# Example
```julia
julia> using ParallelAnalysis, Random, StatsPlots
julia> begin Random.seed!(1234)
           a = rand(Uniform(0.5, 2.0), 30)
           b = [sort(rand(Uniform(-3, 3), 4); rev = false) for i in 1:30]
           θ = rand(Normal(0, 1), 3000)
           resp = generate_response(θ, a, b)
       end;

julia> par_fit1 = parallel(resp, 10, x -> fa(x; cor_method = :Polychoric))
Progress: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| Time: 0:00:02
Parallel Analysis:
    Suggested that the number of factors is 1 (based on resampling)
    Suggested that the number of componets is 1 (based on resampling)

julia> plot(par_fit1) # visualize a result of parallel analysis

julia> plot(par_fit1, markershape = :none) # Don't show noisy makers.



```

"""
function parallel(data, niter, f = fa)
    n, j = size(data)
    fit = f(data)
    # println("Analyze read data.")
    if fit.cor == :Pearson
        V = cor(Matrix(data))
    elseif fit.cor == :Polychoric
        V = polycor(data)
    end
    eig_real_pca = sort!(eigvals(Symmetric(V)); rev = true)
    V[diagind(V)] = communalities(fit)
    eig_real_fa = sort!(eigvals(Symmetric(V)); rev = true)

    # println("Prepare matrix.")
    eig_sim_fa = Matrix{eltype(eig_real_fa)}(undef, niter, length(eig_real_fa))
    eig_sim_pca = Matrix{eltype(eig_real_pca)}(undef, niter, length(eig_real_pca))
    rsm_sim_fa = similar(eig_sim_fa)
    rsm_sim_pca = similar(eig_sim_pca)
    
    prg = Progress(niter)
    # println("Start simulation!")
    Threads.@threads for i in 1:niter
    
        W = random_sample(data)
        Z = randn(n, j)
        
        # resample data
        rsm_fit = f(W)
        M = rsm_fit.mat
        rsm_sim_pca[i, :] = sort!(eigvals(Symmetric(M)); rev = true)
        M[diagind(M)] = communalities(rsm_fit)
        rsm_sim_fa[i, :] = sort!(eigvals(Symmetric(M)); rev = true)

        # simulated data
        sim_fit = fa_pearson(Z)
        R = sim_fit.mat
        eig_sim_pca[i, :] = sort!(eigvals(Symmetric(R)); rev = true)
        R[diagind(R)] = communalities(sim_fit)
        eig_sim_fa[i, :] = sort!(eigvals(Symmetric(R)); rev = true)

        next!(prg)
    end

    # summarize
    eig_sim_fa_mean = mean(eig_sim_fa, dims = 1)[:]
    eig_sim_fa_bounds = map(x -> quantile(x, [0.025, 0.975]), eachcol(eig_sim_fa))
    rsm_sim_fa_mean = mean(rsm_sim_fa, dims = 1)[:]
    rsm_sim_fa_bounds = map(x -> quantile(x, [0.025, 0.975]), eachcol(rsm_sim_fa))

    eig_sim_pca_mean = mean(eig_sim_pca, dims = 1)[:]
    eig_sim_pca_bounds = map(x -> quantile(x, [0.025, 0.975]), eachcol(eig_sim_pca))
    rsm_sim_pca_mean = mean(rsm_sim_pca, dims = 1)[:]
    rsm_sim_pca_bounds = map(x -> quantile(x, [0.025, 0.975]), eachcol(rsm_sim_pca))

    ps = ParallelSets(
        Parallel(eig_real_fa, eig_sim_fa_mean, eig_sim_fa_bounds, rsm_sim_fa_mean, rsm_sim_fa_bounds, niter, :FA, Symbol(fit.cor)),
        Parallel(eig_real_pca, eig_sim_pca_mean, eig_sim_pca_bounds, rsm_sim_pca_mean, rsm_sim_pca_bounds, niter, :PCA, Symbol(fit.cor))
    )

    return ps
end

function findnfactors(x, y)
    for i in axes(x, 1)
        comp = x[i:end] .> y[i:end]
        findfirst(comp) != 1 && return i - 1
    end
    return 0
end

function Base.show(io::IO, x::Parallel)
    println(io, "Parallel Analysis: \nDimension reduction: $(x.reduction_method)")
    println(io, "Correlation method: $(x.correlation_method)")
    println(io, "Simulation size: $(x.iter)")
    println(io, "Suggested the number of factors is $(findnfactors(x.real, x.resampled)) (based on resampling)")
end

function Base.show(io::IO, x::ParallelSets)
    println(io, "Parallel Analysis:")
    println(io, "   Suggested that the number of factors is $(findnfactors(x.FA.real, x.FA.resampled)) (based on resampling)")
    println(io, "   Suggested that the number of componets is $(findnfactors(x.PCA.real, x.PCA.resampled)) (based on resampling)")
end

@recipe function f(pa::Parallel)
    
    @series begin
        # simulated mean
        label --> "$(pa.reduction_method) - Simulated mean"
        linecolor --> :red
        fillcolor --> :red
        linestyle --> :dash
        bds = hcat(pa.simulated_bounds...)
        l = bds[1, :] .- pa.simulated
        u = pa.simulated .- bds[2, :]
        ribbon := (l, u)
        pa.simulated
    end

    @series begin
        # simulated mean
        label --> "$(pa.reduction_method) - Resampled mean"
        linecolor --> :blue
        fillcolor --> :blue
        linestyle --> :dash
        bds = hcat(pa.resampled_bounds...)
        l = bds[1, :] .- pa.resampled
        u = pa.resampled .- bds[2, :]
        ribbon := (l, u)
        pa.resampled
    end

    @series begin
        label := ""
        linecolor --> :black
        linestyle --> :dot
        seriestype := :hline
        [1]
    end
    
    label --> "$(pa.reduction_method) - Real data"
    linecolor --> :black
    yguide --> "Eigen values"
    xguide --> "The number of components"
    title --> "Parallel Analysis over $(pa.iter) simulation."
    ylims --> (-Inf, Inf)
    pa.real
end

@recipe function f(x::ParallelSets)
    @series begin
        markershape --> :dtriangle
        markeralpha --> 0.5
        markercolor --> :white
        x.PCA 
    end
    @series begin
        markershape --> :utriangle
        markeralpha --> 0.5
        markercolor --> :white
        x.FA
    end
    nothing
end
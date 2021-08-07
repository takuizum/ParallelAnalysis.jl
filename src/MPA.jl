# MPA

# Modified Parallel Analysis
# 1) 

abstract type ModifiedParallelType end

struct ModifiedParallelSets

end
struct ModifiedParallel{T1<:AbstractVector, T2<:AbstractVector, T3 <: Real} <: ModifiedParallelType
    real::T1
    simulated::T1
    simulated_bounds::T2
    iter::T3
end

"""
    modified_parallel(data, niter, fsm::FactorScoreMethod)
Modified Parallel Analysis (MPA). MPA, which was introduced by [Drasgow and Lissak, 1983](10.1037/0021-9010.68.3.363), is not a method based on simulation.
After their work, [Finch and Monahan (2008)](10.1080/08957340801926102) proposed a parametric bootstrap variant.
`modified_parallel` is implemented based on Finch and Monahan's work.

# Arguments

- `data` 
- `niter`
- `fsm` is a method for latent trait scoring.

# Values

- `real` Eigen values from read data.
- `simulated` Mean of eigen values from simulated data.
- `simulated_bound` 2.5th and 97.5th perceintile rank values of simulated eigen values.
- `iter` Number of iterations (of the simulation).

# Example
```julia

julia> modified_parallel(data, 100, BayesMean())

Progress: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| Time: 0:00:29
Modified Parallel Analysis:
Simulation size: 100
Suggested the number of factors is 1.

```
"""
function modified_parallel(data, niter, fsm::FactorScoreMethod)

    # IRT
    heufit = heuristicIRT(data)

    # FA
    fit = fa(data)
    V = polycor(data)
    V[diagind(V)] = communalities(fit)
    eig_real_fa = sort!(eigvals(Symmetric(V)); rev = true)
    rsm_sim_fa = Matrix{eltype(eig_real_fa)}(undef, niter, length(eig_real_fa))

    prg = Progress(niter)
    Threads.@threads for i in 1:niter

        # Sample from IRT model
        W = generate_response(heufit, fsm)
        rsm_fit = fa(W)
        M = rsm_fit.mat
        M[diagind(M)] = communalities(rsm_fit)
        rsm_sim_fa[i, :] = sort!(eigvals(Symmetric(M)); rev = true)

        next!(prg)

    end
    
    rsm_sim_fa_mean = mean(rsm_sim_fa, dims = 1)[:]
    rsm_sim_fa_bounds = map(x -> quantile(x, [0.025, 0.975]), eachcol(rsm_sim_fa))

    return ModifiedParallel(
        eig_real_fa, rsm_sim_fa_mean, rsm_sim_fa_bounds, niter
    )

end

function Base.show(io::IO, x::ModifiedParallel)
    println(io, "Modified Parallel Analysis:")
    println(io, "Simulation size: $(x.iter)")
    println(io, "Suggested the number of factors is $(findnfactors(x.real, x.simulated)).")
end

@recipe function f(mpa::ModifiedParallel)
    
    @series begin
        # simulated mean
        label --> "Simulated mean"
        linecolor --> :red
        fillcolor --> :red
        linestyle --> :dash
        bds = hcat(mpa.simulated_bounds...)
        l = bds[1, :] .- mpa.simulated
        u = mpa.simulated .- bds[2, :]
        ribbon := (l, u)
        mpa.simulated
    end

    @series begin
        label := ""
        linecolor --> :black
        linestyle --> :dot
        seriestype := :hline
        [1]
    end
    
    label --> "Real data"
    linecolor --> :black
    yguide --> "Eigen values"
    xguide --> "The number of components"
    title --> "Parallel Analysis over $(mpa.iter) simulation."
    ylims --> (-Inf, Inf)
    mpa.real
end

struct Parallel
    real
    simulated
    bounds
    iter
    reduction_method
    correlation_method
end

"""
    parallel(data, niter, f = fa, g = Statistics.cor; args...)
Parallel Analysis.

# Arguments
- `data`
- `niter` is a size of simulation that generate (normal) random data matrix, calculate R, correlation matrix and its eigen values.
- `cor` Function for dimension reduction. Default is `fa`.

# Values
- `real` Eigen values from read data.
- `simulated` Mean of eigen values from simulated data.
- `bound` 5th and 95th perceintile rank values of simulated eigen values.
- `iter` Number of iterations (of the simulation).
- `reduncion_method`
- `correlation_method`

"""
function parallel(data, niter, f = fa)
    n, j = size(data)
    fit = f(data)
    if fit.cor == :Pearson
        V = cor(data)
    elseif fit.cor == :Polychoric
        V = polycor(data)
    end
    V[diagind(V)] = communarities(fit)
    eig_real = sort!(eigvals(Symmetric(V)); rev = true)
    eig_sim = Matrix{eltype(eig_real)}(undef, niter, length(eig_real))
    Threads.@threads for i in 1:niter
        if fit.cor == :Pearson
            M = random_matrix(n, j)
        elseif fit.cor == :Polychoric
            M = polycor(random_sample(data))
        end
        sim_fit = f(data)
        M[diagind(M)] = communarities(sim_fit)
        eig_sim[i, :] = sort!(eigvals(Symmetric(M)); rev = true)
    end
    eig_sim_mean = mean(eig_sim, dims = 1)[:]
    eig_sim_bounds = map(x -> quantile(x, [0.5, 0.95]), eachcol(eig_sim))
    return Parallel(eig_real, eig_sim_mean, eig_sim_bounds, niter, Symbol(f), Symbol(fit.cor))
end


function Base.show(io::IO, x::Parallel)
    println(io, "Parallel Analysis: \nDimension reduction: $(x.reduction_method)")
    println(io, "Correlation method: $(x.correlation_method)")
    println(io, "Simulation size: $(x.iter)")
    println(io, "Suggested the number of factors is $(count(x.real .> x.simulated))")
end


@recipe function f(pa::Parallel)
    
    @series begin
        # simulated mean
        label --> "Simulated mean"
        linecolor --> :red
        linestyle --> :dash
        bds = hcat(pa.bounds...)
        l = bds[1, :] .- pa.simulated
        u = bds[2, :] .- pa.simulated
        ribbon := (l, u)
        pa.simulated
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
    ylims --> (0, Inf)
    pa.real
end
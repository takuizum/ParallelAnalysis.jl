
struct fa
    data
    fit
    cor
    loadings
    cov
    projection
    nfactors
    dimension
end

"""
    fa(M, nfactors::Int64 = 1; args...)
Factor analysis wrapper.

`M` is a dataframe to be fitted. `nfactors` is the number of the latent traits.
`args` are arguments for FA model. See, `MultivariateStats.fit(FactorAnalysis, X; args...)`(https://multivariatestatsjl.readthedocs.io/en/latest/fa.html).
`maxoutdim` was preferentially fixed by `nfactors` and `mean` was fixed by 0. User can modified only `method`, `tol`, `tot` and `η`.
"""

function fa(M; method = :Polychoric, nfactors = 1, args...)
    if method == :Polychoric
        fa_polychoric(M, nfactors; args...)
    elseif method == :Pearson
        fa_pearson(M, nfactors; args...)
    else
        @warn "`method` should be choosen from (:Pearson, :Polychoric)."
        return nothing
    end
end

function fa_polychoric(M, nfactors = 1; args...)
    S = convert(Matrix{Float64}, polycor(M))
    n = size(S, 1)
    ft = fit(FactorAnalysis, S; mean = fill(0.0, n), maxoutdim = nfactors, args...)
    fa(M, ft, :Polychoric, loadings(ft), cov(ft), projection(ft), nfactors, n)
end
function fa_pearson(M, nfactors = 1; args...)
    S = convert(Matrix{Float64}, cor(M))
    n = size(S, 1)
    ft = fit(FactorAnalysis, S; mean = fill(0.0, n), maxoutdim = nfactors, args...)
    fa(M, ft, :Pearson, loadings(ft), cov(ft), projection(ft), nfactors, n)
end

"""
    cov(x::fa)
Extract covariance matrix from FA models.
"""
function cov(x::fa)
    x.cov
end

"""
    loadings(x::fa)
Extract factor loadings from FA models.
"""
function loadings(x::fa)
    x.loadings
end

function show(io::IO, x::fa)
    println(io, "Factor Analysis $(x.cor)")
end
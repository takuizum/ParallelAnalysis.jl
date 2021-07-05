
struct FA{T1<:Real, F1<:FactorAnalysis}
    data
    fit::F1
    cor
    loadings::AbstractArray{T1}
    cov::AbstractArray{T1}
    projection::AbstractArray{T1}
    nfactors
    dimension
end

"""
    fa(M, nfactors::Int64 = 1; args...)
Factor analysis wrapper.

`M` is a dataframe to be fitted. `nfactors` is the number of the latent traits.
`cor_method` is method for calculating the correlation matrix.
`args` are arguments for FA model. See, `MultivariateStats.fit(FactorAnalysis, X; args...)`(https://multivariatestatsjl.readthedocs.io/en/latest/fa.html).
`maxoutdim` was preferentially fixed by `nfactors` and `mean` was fixed by 0. User can modified only `method`, `tol`, `tot` and `η`.
"""

function fa(M; cor_method = :Polychoric, nfactors = 1, args...)
    if cor_method == :Polychoric
        fa_polychoric(M, nfactors; args...)
    elseif cor_method == :Pearson
        fa_pearson(M, nfactors; args...)
    else
        throw(ArgumentError("`method` should be choosen from (:Pearson, :Polychoric)."))
    end
end

function fa_polychoric(M, nfactors = 1; args...)
    S = convert(Matrix{Float64}, polycor(M))
    n = size(S, 1)
    ft = fit(FactorAnalysis, S; mean = fill(0.0, n), maxoutdim = nfactors, args...)
    FA(M, ft, :Polychoric, loadings(ft), cov(ft), projection(ft), nfactors, n)
end
function fa_pearson(M, nfactors = 1; args...)
    S = convert(Matrix{Float64}, cor(M))
    n = size(S, 1)
    ft = fit(FactorAnalysis, S; mean = fill(0.0, n), maxoutdim = nfactors, args...)
    FA(M, ft, :Pearson, loadings(ft), cov(ft), projection(ft), nfactors, n)
end

"""
    cov(x::FA)
Extract covariance matrix from FA models.
"""
function cov(x::FA)
    x.cov
end

"""
    loadings(x::FA)
Extract factor loadings from FA models.
"""
function loadings(x::FA)
    x.loadings
end

function show(io::IO, x::FA)
    println(io, "Factor Analysis $(x.cor)")
end
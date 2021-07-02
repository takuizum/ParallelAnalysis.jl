
struct fa
    data
    fit
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
function fa(M, nfactors = 1; args...)
    S = convert(Matrix{Float64}, cor(M))
    n = size(S, 1)
    ft = fit(FactorAnalysis, S; mean = fill(0.0, n), maxoutdim = nfactors, args...)
    fa(M, ft, loadings(ft), cov(ft), projection(ft), nfactors, n)
end


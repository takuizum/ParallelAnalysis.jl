

"""
    random_matrix(n, m)
Generates a correlation matrix from a `n` by `m` random matrix.
`n` is the number of culumns. `m` is the number of rows.

# Example
```julia
julia> random_matrix(10_000, 5)
5×5 Matrix{Float64}:
  1.0          0.00446344   0.00358672  -0.00662204   0.00685964
  0.00446344   1.0         -0.00526056   0.0165235    0.00962451
  0.00358672  -0.00526056   1.0          0.00227261  -0.0133972
 -0.00662204   0.0165235    0.00227261   1.0         -9.9025e-5
  0.00685964   0.00962451  -0.0133972   -9.9025e-5    1.0
```
"""
function random_matrix(n, m)
    M = randn(n, m)
    R = cor(M)
    return R
end
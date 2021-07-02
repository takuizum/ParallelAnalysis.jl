

"""
    random_matrix(n, m)
Generates a random matrix.
`n` is the number of culumns. `m` is the number of rows.
"""
function random_matrix(n, m)
    M = randn(n, m)
    R = cor(M)
    return R
end


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


"""


# Example
```julia
julia> using StatsBase
julia> X = hcat(map(i -> sample([1,2,3,4,5], 100), 1:10)...)
100×10 Matrix{Int64}:
 3  1  5  1  2  5  3  3  1  3
 2  1  1  1  2  4  4  4  3  1
 5  4  3  5  2  5  3  3  3  1
 3  4  4  1  3  1  2  3  1  3
 2  3  1  4  1  2  2  1  5  3
 5  2  3  1  4  1  4  3  5  3
 2  2  4  3  4  2  4  3  1  3
 ⋮              ⋮           
 2  5  3  1  1  4  1  4  5  5
 3  2  4  3  4  5  5  1  1  4
 5  1  5  5  2  1  2  1  5  1
 1  5  1  4  3  4  2  4  1  1
 1  2  1  1  2  1  1  1  2  5
 1  1  2  5  5  5  1  1  5  3

julia> random_sample(X)

```
"""
function random_sample(x)
    n = size(x, 1)
    hcat(map(i -> sample_atleast2(i, n; replace = true), eachcol(x))...)
    [sample(i, n; replace = true) for i in eachcol(x)]
end


"""
	sample_atleast2(i, n; args...)
Sample `n` size vector from a set `i`.

# Example
```julia
julia> using StatsBase, Statistics
julia> x = vcat(fill(1, 100), fill(0, 1));
julia> var(sample(x, 101; replace = true))
julia> var(sample_atleast2(x, 101; replace = true))

```
"""
function sample_atleast2(i, n; args...)
	length(unique(i)) == 1 && throw(ArgumentError("Unique sample set contains only one element."))
	iter = 1
	y = sample(i, n; args...)
	while iter ≤ 100
		length(unique(y)) ≥ 2 && break
		iter += 1
		y = sample(i, n; args...)
	end
	iter ≥ 100 && throw(ArgumentError("Over 100 resampling, the condition (at least 2 elements) was not satisfied. Data might has too small variance."))
	return y
end

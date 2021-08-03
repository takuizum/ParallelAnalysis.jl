
struct ConTab{T1<:AbstractMatrix, T2<:AbstractVector}
    X::T1
    mx::T2
    my::T2
end

function contab(x, y; normalize = true, correction = 0.5, verbose = true)
    t = [count((i .=== x) .& (j .=== y)) for i in sort(unique(skipmissing(x))), j in sort(unique(skipmissing(y)))]
    t = Matrix{Float64}(t)
    t[t .== 0] .= correction / sum(t)
    if normalize
        t = t ./ sum(t)
        if any(t .< eps()) 
            @warn "Some normalized frequency are smaller than `eps()`. Use larger `correction` value."
        end
        if any(t .< 0.01) && verbose
            @warn "Probabilities of some elements in the contigency tabel, $(findall(x -> x < 0.01, t)) are less than 0.01. The estimation of polychoric correlation may be unstable. For `snpolyc`, use a smaller number of `step` to solve it."
        end
    end
    return ConTab(t, sum(t; dims = 2)[:], sum(t; dims = 1)[:])
end

function contab(t; normalize = true, correction = 0.5, verbose = true)
    t = Matrix(t)
    t[t .== 0] .= correction / sum(t)
    if normalize
        t = t ./ sum(t)
        if any(t .< eps()) 
            @warn "Some normalized frequency are smaller than `eps()`. Use more large `correction` value."
        end
        if any(t .< 0.01) && verbose
            @warn "Some elements in the contigency tabel, $(findall(x -> x < 0.01, t)). The estimation of polychoric correlation may be unstable. For `snpolyc`, use a larger number of `step` to solve it."
        end
    end
    return contab(t, sum(t; dims = 2)[:], sum(t; dims = 1)[:])
end


# Calculate all mass, not density.

"""
    ppdf(f, max, min, step)

Compute the volume of the rectangular area ∈ [min, max] by using trapezoidal rule.
"""
function ppdf(f, max, min, step)
    x = range(min[1], stop = max[1], step = step)
    y = range(min[2], stop = max[2], step = step)
    return trapz((x, y), [f([i, j]) for i in x, j in y])
    # return hcubature(f, max, min)[1]
end

"""
Compute, more efficiently than using the trapezoidal rule, the volume of the bivariate normal distributions.
"""
function ppdf(f, m)
    g(h, k, θ) = exp(-0.5 * (h^2 + k^2 - 2*h*k*θ)/(1-θ^2)) / sqrt(1-θ^2 )
    Φ(x) = cdf(Normal(0, 1), x)
    return quadgk(t->g(m[1], m[2], t), 0, f.Σ[1,2])[1] / 2π + Φ(m[1])*Φ(m[2])
    # v = range(0, f.Σ[1,2], length = 20) 
    # V = @trapz v x g(m[1], m[2], x)
    # return V / 2π + Φ(m[1])*Φ(m[2])
end

# Numerical integration with sophisticated 1dim version
function H(a, b, c, d, mvd::MvNormal)
    A1 = a == -Inf || c == -Inf ?  0.0 : ppdf(mvd, [a, c])
    A2 = a == -Inf ? 0.0 : ppdf(mvd, [a, d])
    A3 = c == -Inf ? 0.0 : ppdf(mvd, [b, c])
    A4 = d == Inf && b == Inf ? 1.0 : ppdf(mvd, [b, d])
    return A1 - A2 - A3 + A4
end

# Loss function
function loss(t::ConTab, ξx, ξy, d::MultivariateDistribution)
    I, J = size(t.X)
    h = zeros(Float64, I, J)
    @fastmath for i in 1:I, j in 1:J
        if i == I && j == J && sum(h) < 1.0
            h[i, j] = 1.0 - sum(h)
        elseif i == I
            h[i, j] = t.my[j] - sum(h[1:end-1, j])[1]
        elseif j == J
            h[i, j] = t.mx[i] - sum(h[i, 1:end-1])[1]
        else
            h[i, j] = H(ξx[i], ξx[i+1], ξy[j], ξy[j+1], d)
        end
    end
    if any(h .< 0.0)
        h = abs.(h)
    end
    h = h ./ sum(h)
    return - sum(t.X .* log.(h))
end

function polyc(t::ConTab)
    # Fix marginal freq
    cumx = [0.0; cumsum(t.mx)[1:end-1]; 1.0]
    cumy = [0.0; cumsum(t.my)[1:end-1]; 1.0]   
    ξx = quantile.(Normal(0, 1), cumx)
    ξy = quantile.(Normal(0, 1), cumy)
    opt = optimize(ρ -> loss(t, ξx, ξy, MvNormal([0, 0], [1.0 ρ; ρ 1.0])), -1.0, 1.0, Brent())
    return (ρ = opt.minimizer, τ₁ = ξx, τ₂ = ξy)
end

function polyc(x, y; verbose = true)
    tab = contab(x, y; verbose = verbose)
    # Fix marginal freq
    polyc(tab)
end

"""
    polycor(X::Union{AbstractDataFrame, AbstractMatrix}; lower_tri = false)
Estimate polychoric correlation matrix from X.

# Arguments
- `X` Matrix contains categorical vectors.
- `lower_tri` logical.

# Examples
```julia

using Distributions
# Support function
function discretize(x, th)
    sum(th .< x)
end
# Gen sim data.
raw = rand(MvNormal([0, 0], [1 .5; .5 1]), 10000)'
τ1 = sort!(rand(Uniform(-2, 2), 3))
τ2 = sort!(rand(Uniform(-2, 2), 4))
obs = [discretize.(raw[:, 1], Ref(τ1)) discretize.(raw[:, 2], Ref(τ2)) ]

polycor(obs)
```
"""
function polycor(X)
    J = size(X, 2)
    r = Matrix{AbstractFloat}(undef, J, J)
    r[diagind(r)] .= 1.0
    for i in 1:J
        x = @view X[:, i]
        for j in i:J
            y = @view X[:, j]
            r[j, i] = polyc(x, y; verbose = false).ρ
            r[i, j] = r[j, i]
        end
    end
    return r
end

function replace_diagonal!(M)
    offdiagind(A) = (ι for ι in CartesianIndices(A) if ι[1] ≠ ι[2])
    L = maximum(M[collect(offdiagind(M))])
    M[diagind(M)] .= L
end


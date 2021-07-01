module polycor

using Trapz, QuadGK, Roots, Distributions, Optim, DataFrames, LinearAlgebra, Statistics

struct ConTab
    X
    mx
    my
end

function ConTab(x, y; normalize = true, correction = 0.5, verbose = true)
    t = [count((i .== x) .& (j .== y)) for i in sort(unique(x)), j in sort(unique(y))]
    t = Matrix{Float64}(t)
    t[t .== 0] .= correction / sum(t)
    if normalize
        t = t ./ sum(t)
        if any(t .< eps()) 
            @warn "Some normalized frequency are smaller than `eps()`. Use more large `correction` value."
        end
        if any(t .< 0.01) && verbose
            @warn "Some elements in the contigency tabel, $(findall(x -> x < 0.01, t)). The estimation of polychoric correlation may be unstable. For `snpolyc`, use a small number of `step` to solve it."
        end
    end
    return ConTab(t, sum(t; dims = 2)[:], sum(t; dims = 1)[:])
end

function ConTab(t; normalize = true, correction = 0.5, verbose = true)
    t = Matrix(t)
    t[t .== 0] .= correction / sum(t)
    if normalize
        t = t ./ sum(t)
        if any(t .< eps()) 
            @warn "Some normalized frequency are smaller than `eps()`. Use more large `correction` value."
        end
        if any(t .< 0.01) && verbose
            @warn "Some elements in the contigency tabel, $(findall(x -> x < 0.01, t)). The estimation of polychoric correlation may be unstable. For `snpolyc`, use a large number of `step` to solve it."
        end
    end
    return ConTab(t, sum(t; dims = 2)[:], sum(t; dims = 1)[:])
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
Compute, more efficiently than one using trapezoidal rule, the volume of bivariate normal distributions.
"""
function ppdf(f, m)
    g(h, k, θ) = exp(-0.5 * (h^2 + k^2 - 2*h*k*θ)/(1-θ^2)) / sqrt(1-θ^2 )
    Φ(x) = cdf(Normal(0, 1), x)
    return quadgk(t->g(m[1], m[2], t), 0, f.Σ[1,2])[1] / 2π + Φ(m[1])*Φ(m[2])
end

# Numerical integration with sophisticated 1dim version
function H(a, b, c, d, mvd::MvNormal)
    A1 = a == -Inf || c == -Inf ?  0.0 : ppdf(mvd, [a, c])
    A2 = a == -Inf ? 0.0 : ppdf(mvd, [a, d])
    A3 = c == -Inf ? 0.0 : ppdf(mvd, [b, c])
    A4 = d == Inf && b == Inf ? 1.0 : ppdf(mvd, [b, d])
    return A1 - A2 - A3 + A4
end

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


"""

Estimate polychoric correlation based on the contigency table and the bivariate distribution.

# Arguments

`x` and `y` are score vectors that consits of the dicrete values with several categories.

observed

`t` which type is `ConTab`

# Optional arguments

`step` The step size of the quadreture. This option control the accuracy of the numerical integration based on the trapezoidal rule.

If there is relatively small frequency category, to avoid an error in the evaluation of the category probability, use a small number of `step`.
"""
function polyc(t::ConTab)
    # Fix marginal freq
    cumx = [0.0; cumsum(t.mx)[1:end-1]; 1.0]
    cumy = [0.0; cumsum(t.my)[1:end-1]; 1.0]   
    ξx = quantile.(Normal(0, 1), cumx)
    ξy = quantile.(Normal(0, 1), cumy)
    opt = optimize(ρ -> loss(t, ξx, ξy, MvNormal([0, 0], [1.0 ρ; ρ 1.0])), -1.0, 1.0, Brent())
    return (ρ = opt.minimizer, τ₁ = ξx, τ₂ = ξy)
end

"""

Estimate polychoric correlation based on the contigency table and the bivariate distribution.
"""
function polyc(x, y; verbose = true)
    tab = ConTab(x, y; verbose = verbose)
    # Fix marginal freq
    polyc(tab)
end

"""
    cor(X::Union{AbstractDataFrame, AbstractMatrix}; lower_tri = false)
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

polycor.cor(obs)
```
"""
function cor(X::Union{AbstractDataFrame, AbstractMatrix}; lower_tri = false)
    println("Compute polychoric correlations.")
    J = size(X, 2)
    r = Matrix{Union{Missing, Float64}}(undef, J, J)
    r[diagind(r)] .= 1.0
    for i in 1:J
        x = @view X[:, i]
        for j in i:J
            y = @view X[:, j]
            r[j, i] = polyc(x, y; verbose = false).ρ
            if !lower_tri
                r[i, j] = r[j, i]
            end
        end
    end
    if lower_tri
        return LowerTriangular(r)
    else
        return r
    end
end


# SkewedNormal
# 
# See https://github.com/JuliaStats/Distributions.jl/blob/master/src/univariate/continuous/skewnormal.jl#L52 , 
# and 
# https://github.com/JuliaStats/Distributions.jl/pull/1104
# Distributions' utilities for skew normal, except for cdf, has already been porvided.
# Install up from version 0.23.10
# add Distributions@0.23.10

# Internal functions
# Too slow
Distributions.cdf(dist::SkewNormal, x::Real) = quadgk(t->pdf(dist,t), -Inf, x)[1]
function Statistics.quantile(dist::SkewNormal, β::Float64) 
    if β == 0.0
        return -Inf
    elseif β == 1.0
        return Inf
    else
        find_zero(x -> cdf(dist, x) - β, dist.ξ)
    end
end

# Trivariate Normal integration
function trivariateintegral(d::MvNormal, c₁, c₂, c₃)
    Φ = Distributions.normcdf
    A₁ = Distributions.normpdf(c₁) / Φ(c₁)
    μ₂₁ = -d.Σ[1, 2] * A₁
    B₁ = A₁*(c₁ + A₁)
    σ₂₁ = sqrt(1 - d.Σ[1, 2]^2 * B₁)
    c₂₁ = (c₂ - μ₂₁) / σ₂₁
    μ₃₁ = -d.Σ[1, 3] * A₁
    σ₃₁ = sqrt(1 - d.Σ[1, 3]^2 * B₁)
    c₃₁ = (c₃ - μ₃₁) / σ₃₁
    A₂₁ = Distributions.normpdf(c₂₁) / Φ(c₂₁)
    B₂₁ = A₂₁*(c₂₁ + A₂₁)
    r₂₃₁ = (d.Σ[2, 3] - d.Σ[1, 2]*d.Σ[1, 3]*B₁) / (√(1-d.Σ[1, 2]^2*B₁) * √(1-d.Σ[1, 3]^2*B₁))
    c₃₂ = (c₃₁ + r₂₃₁*A₂₁) / √(1-r₂₃₁^2*B₂₁)
    return Φ(c₃₂) * Φ(c₂₁) * Φ(c₁)
end


# BivariateSkewedNormal

# function bivariateskewnormpdf(x₁, x₂, ω, α₁, α₂)
#     2pdf(MvNormal([0, 0], [1.0 ω; ω 1.0]), [x₁, x₂]) * Distributions.normcdf(x₁ * α₁ + x₂ * α₂)
# end

struct BivariateSkewNormal <: ContinuousMultivariateDistribution
    μ
    Σ
    α
    function BivariateSkewNormal(μ, Σ, α)
        if !(length(μ) == length(α) == size(Σ, 1) == size(Σ, 2))
            @error "The size of coordinates for each parameters do not match."
        else
            new(μ, Σ, α)
        end
    end
end

function Distributions.pdf(d::BivariateSkewNormal, x::AbstractArray{T,1}) where T
    2*pdf(MvNormal(d.μ, d.Σ), x) * Distributions.normcdf(d.α'x)
end

function Distributions.cdf(d::BivariateSkewNormal, c::AbstractArray{T,1}) where T
    Ψ = d.Σ
    α = d.α
    Σ = [
        1+α'Ψ*α  -α'Ψ
        -Ψ*α      Ψ
    ]
    2trivariateintegral(MvNormal(Σ), 0.0, c[1], c[2])
end

# Numerical integration for bivariate skew normal
function H(a, b, c, d, mvd::BivariateSkewNormal)
    A1 = a == -Inf || c == -Inf ?  0.0 : cdf(mvd, [a, c])
    A2 = a == -Inf ? 0.0 : cdf(mvd, [a, d])
    A3 = c == -Inf ? 0.0 : cdf(mvd, [b, c])
    A4 = d == Inf && b == Inf ? 1.0 : cdf(mvd, [b, d])
    # if (A1 - A2 - A3 + A4) < 0.0
    #     @warn "Volumes of a rectangles are $([A1, A2, A3, A4]) at $([a, b, c, d]) on\n$(mvd) is negative."
    # end
    return A1 - A2 - A3 + A4
end

function marginalparameters(d::BivariateSkewNormal, i)
    c = Dict(1 => 2, 2 => 1)
    Ω = d.Σ[c[i],c[i]] - d.Σ[c[i],i] / d.Σ[i,i] * d.Σ[i,c[i]] # 1- ω²
    (d.α[i] + d.Σ[i,c[i]]/d.Σ[i,i]*d.α[c[i]]) / √(1+d.α[c[i]]^2* Ω)
end

function snpolyc(t::ConTab)
    # non iterative, psedo-MLE
    cumx = [0.0; cumsum(t.mx)[1:end-1]; 1.0]
    cumy = [0.0; cumsum(t.my)[1:end-1]; 1.0]
    # ξx = quantile.(SkewNormal(0.0, 1.0, 0.0), cumx)
    # ξy = quantile.(SkewNormal(0.0, 1.0, 0.0), cumy)
    # # Heuristic version
    # opt = optimize(θ -> loss(t, ξx, ξy, BivariateSkewNormal([0, 0], [1.0 θ[1]; θ[1] 1.0], [θ[2], θ[3]])), [-1.0, -Inf, -Inf], [1.0, Inf, Inf], [0.0, 0.0, 0.0], Fminbox(NelderMead()))
    # Complex model
    opt = optimize(θ -> loss2(t, cumx, cumy, BivariateSkewNormal([0, 0], [1.0 θ[1]; θ[1] 1.0], [θ[2], θ[3]])), [-1.0, -Inf, -Inf], [1.0, Inf, Inf], [0.0, 0.0, 0.0], Fminbox(NelderMead()))
    # calculate parameters on marginal distributions
    θ′ = opt.minimizer
    α₁ = marginalparameters(BivariateSkewNormal([0, 0], [1.0 θ′[1]; θ′[1] 1.0], [θ′[2], θ′[3]]), 1)
    α₂ = marginalparameters(BivariateSkewNormal([0, 0], [1.0 θ′[1]; θ′[1] 1.0], [θ′[2], θ′[3]]), 2)
    ξx = quantile.(SkewNormal(0.0, 1.0, α₁), cumx)
    ξy = quantile.(SkewNormal(0.0, 1.0, α₂), cumy)
    ω = θ′[1]
    δ₁ = α₁ / sqrt(1 + α₁^2)
    δ₂ = α₂ / sqrt(1 + α₂^2)
    𝜓 = (ω - δ₁*δ₂) / sqrt((1 - δ₁) * (1 - δ₂))
    ρ = (ω - 1/(2π) * δ₁*δ₂) / sqrt((1 - 2*δ₁^2/π) * (1 - 2*δ₂^2/π))
    return (ρ = ρ, r = 2sin(ρ * π / 6), τ₁ = ξx, τ₂ = ξy, α₁ = α₁, α₂ = α₂)
end

function snpolyc(x, y)
    tab = ConTab(x, y)
    snpolyc(tab)
end


function loss2(t::ConTab, cumx, cumy, d::MultivariateDistribution)
    ξx = quantile.(SkewNormal(0.0, 1.0, marginalparameters(d, 1)), cumx)
    ξy = quantile.(SkewNormal(0.0, 1.0, marginalparameters(d, 2)), cumy)
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
        # h[i, j] = H(ξx[i], ξx[i+1], ξy[j], ξy[j+1], d, step)
    end
    if any(h .< 0.0)
        # @warn("Fail to optimize parameters. Return, on the way, the volume of the rectangles of SkewNormal")
        # @show h, d
        h = abs.(h)
    end
    h = h ./ sum(h)
    return - sum(t.X .* log.(h))
    # return - prod(t.X .^ h)
end


end

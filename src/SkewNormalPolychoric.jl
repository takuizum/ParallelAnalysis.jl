## NOT INCLUDED FILE
#


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
#
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


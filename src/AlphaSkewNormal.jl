using Distributions

abstract type AbstractAlphaSkewNormal <: ContinuousMultivariateDistribution end

struct AlphaSkewNormal{T1<:AbstractVector, T2<:AbstractMatrix} <: AbstractAlphaSkewNormal
    μ::T1
    Σ::T2
    α::T1
end

function Distributions.pdf(d::AbstractAlphaSkewNormal, z::AbstractArray{T, 1}) where T
    K = 2 + sum(d.α .^2) + 2prod(d.α) * d.Σ[1,2]
    s = 1 + (1 - d.α'z)
    ϕ = pdf(MvNormal(d.μ, d.Σ), z)
    return s / K * ϕ
end

function semifinite_integrand(d, x, y, c1, c2)
    pdf(d, [c1 + x/(1-x), c2 + y/(1-y)]) / (1-x)^2 / (1-y)^2
end
function Distributions.cdf(d::AbstractAlphaSkewNormal, c::AbstractArray{T, 1}) where T
    # hypercubeに座標変換して積分を計算する。
    cuhre( (x, f) -> f[1] = semifinite_integrand(d, x[1], x[2], c...) ).integral[1]
end

function mpdf(d, x, i)
    i == 1 ? j = 2 : j = 1
    K = 2 + sum(d.α .^2) + 2prod(d.α) * d.Σ[1,2]
    s = 1 + d.α[j]^2 * (1 - d.Σ[1,2]^2) + *(1 - (d.α[i] + d.α[j] * d.Σ[1,2])*(x - d.μ[i])/d.Σ[i,i])^2
    ϕ = pdf(Normal(d.μ[i], d.Σ[i,i]), x)
    return s / K * ϕ
end

function Distributions.cdf(d::AbstractAlphaSkewNormal, x::Real, i::Int64)
    quadgk(t -> mpdf(d, t, i), -Inf, x)[1]
end

function Statistics.quantile(d::AbstractAlphaSkewNormal, β, i)
    if β == 0.0
        return -Inf
    elseif β == 1.0
        return Inf
    else
        find_zero(x -> cdf(d, x, i) - β, 0.0)
    end
end

function H(a, b, c, d, mvd::AbstractAlphaSkewNormal)
    a < b && throw(ArgumentError("Region is improper. a = $(a) and b = $(b)"))
    c < d && throw(ArgumentError("Region is improper. c = $(c) and d = $(d)"))

    A4 = (a == Inf) || (c == Inf) ?  0.0 : cdf(mvd, [a, c])
    A2 = a == Inf ? 0.0 : cdf(mvd, [a, d])
    A3 = c == Inf ? 0.0 : cdf(mvd, [b, c])
    A1 = (d == -Inf) && (b == -Inf) ? 1.0 : cdf(mvd, [b, d])
    V = A1 - A2 - A3 + A4
    V ≤ 0 && throw(ArgumentError("Volume is negatpive. A1 = $A1, A12= $A2, A3 = $A3, A4 = $A4, d = $mvd"))
    # V ≤ 0 && throw(ArgumentError("Volume is negative. a = $a, b = $b, c = $c, d = $d"))
    isnan(V) && throw(ArgumentError("Volume is NaN. A1 = $A1, A1 = $A2, A3 = $A3, A4 = $A4"))
    return V
end

# Loss function
function loss(t::ConTab, cumx, cumy, d::AbstractAlphaSkewNormal)
    ξx = [quantile(d, x, 1) for x in cumx]
    ξy = [quantile(d, y, 2) for y in cumy]
    reverse!.((ξx, ξy))
    I, J = size(t.X)
    h = zeros(Float64, I, J)
    @fastmath for i in 1:I, j in 1:J
        if (i == I) && (j == J) && (sum(h) < 1.0)
            h[i, j] = 1.0 - sum(h)
        elseif i == I
            h[i, j] = t.my[j] - sum(h[1:end-1, j])[1]
        elseif j == J
            h[i, j] = t.mx[i] - sum(h[i, 1:end-1])[1]
        else
            h[i, j] = H(ξx[i], ξx[i+1], ξy[j], ξy[j+1], d)
        end
    end
    if any(h .≤ 0.0)
        h = abs.(h)
    end
    h[h .≤ 0.0] .= minimum(h[h .!== 0.0])/2.0
    h = h ./ sum(h)
    return - sum(t.X .* log.(h))
end

function asnpolyc(t::ConTab)
    cumx = [0.0; cumsum(t.mx)[1:end-1]; 1.0]
    cumy = [0.0; cumsum(t.my)[1:end-1]; 1.0]
    # Optimization ρ and thresholds, simultaneously
    opt = optimize(θ -> loss(t, cumx, cumy, AlphaSkewNormal([.0, .0], [1.0 θ[1]; θ[1] 1.0], [θ[2], θ[3]])), [-1.0, -Inf, -Inf], [1.0, Inf, Inf], [0.0, 0.0, 0.0], Fminbox(NelderMead()))
    # calculate parameters on marginal distributions
    θ = opt.minimizer
    optd = AlphaSkewNormal([.0, .0], [1.0 θ[1]; θ[1] 1.0], [θ[2], θ[3]])
    # α₁, α₂ = θ[2], θ[3]
    return θ[1]
end

function asnpolyc(x, y)
    tab = contab(x, y; verbose = false)
    # Fix marginal freq
    asnpolyc(tab)
end

function asnpolycor(X)
    J = size(X, 2)
    r = Matrix{Float64}(undef, J, J)
    r[diagind(r)] .= 1.0
    for i in 1:J
        x = @view X[:, i]
        for j in i+1:J
            y = @view X[:, j]
            r[j, i] = asnpolyc(x, y)
            r[i, j] = r[j, i]
        end
    end
    # isposdef(r) && @warn "Matrix is not a positive definite."
    return r
end
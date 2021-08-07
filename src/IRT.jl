# Heuristic IRT module

function icc(θ, a, b)
    logistic(a * (θ - b))
end

function icc(x, θ, a, b)
    if x == 0
        p = 1 - icc(θ, a, b[1])
    elseif x == length(b)
        p = icc(θ, a, b[end])
    else
        p = icc(θ, a, b[x]) - icc(θ, a, b[x + 1])
    end
    return p
end

function generate_response(θ, a, b)
    X = Matrix{Int64}(undef, length(θ), length(a))
    for j in axes(X, 2)
        x = 0:1:length(b[j])
        for i in axes(X, 1)
            ps = cumsum(icc.(x, θ[i], a[j], Ref(b[j])))
            u = rand(Uniform(0, 1), 1)
            X[i, j] = count(ps .< u)
        end
    end
    return X
end

"""
    v2vv
Transform vector to vector of vector
"""
function v2vv(v)
    return [[i] for i in v]
end

# Heuristic IRT parameter estimation based on the polychoric correlations.

mutable struct HeuristicIRT
    a
    d
    b
end

function transform_FA_IRT(α::T, τ::Vector{T}) where T <: Real
    σ = √(1 - α^2)
    a = α / σ
    d = -τ ./ σ
    b = -d ./ a
    return a, d[begin+1:end-1], b[begin+1:end-1]
end

function heuristicIRT(X; method = :em)
    J = size(X, 2)
    r = Matrix{Float64}(undef, J, J)
    r[diagind(r)] .= 1.0
    τ = Vector{Vector{Float64}}(undef, J)
    for i in 1:J
        x = @view X[:, i]
        for j in i+1:J
            y = @view X[:, j]
            p = polyc(x, y; verbose = false)
            r[j, i] = p.ρ
            r[i, j] = r[j, i]
            if j == J
                τ[i] = p.τ₁
            end
        end
    end
    # FA
    replace_diagonal!(r)
    f = fit(FactorAnalysis, r; mean = fill(0.0, J), maxoutdim = 1, method = method)
    α = loadings(f)
    tp = transform_FA_IRT.(α, τ)
    return HeuristicIRT(getindex.(tp, 1), getindex.(tp, 2), getindex.(tp, 3))
end

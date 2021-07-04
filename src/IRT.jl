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
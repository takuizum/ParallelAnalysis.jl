using Distributions, ForwardDiff, Cuba

ForwardDiff.jacobian(pdf())

function semifinite_integrand(d, x, c)
    J = jacobian(x -> [pdf(d, x)], x)
    pdf(d, [c1 + x/(1-x), c2 + y/(1-y)])
end


function Distributions.cdf(d::AbstractAlphaSkewNormal, c::AbstractArray{T, 1}) where T
    # hypercubeに座標変換して積分を計算する。
    J = jacobian(x -> pdf(d, c), x)
    cuhre( (x, f) -> f[1] = semifinite_integrand(d, x[1], x[2], c...) ).integral[1]
    # ヤコビアンを計算して，それとpafをかけ合わせ，Cubaで処理する。ヤコビアンはForwardDiffで計算可能。
end

using ForwardDiff
ForwardDiff.jacobian(x -> [sin(x[1]) * cos(x[2])], [0 1;1 0])
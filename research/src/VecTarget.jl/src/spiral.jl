struct Spiral{T}
    gms::T
end

function Spiral(n_gaussians::Int, σ::AbstractFloat)
    θs = range(0, (6 * π)^2; length=n_gaussians)
    θs = sqrt.(θs)
    xs = (1 / 40) * θs .* cos.(θs)
    ys = (1 / 40) * θs .* sin.(θs)
    μs = cat(xs', ys'; dims=1)
    mixing = ones(n_gaussians) / n_gaussians
    return GaussianMixtures(mixing, BroadcastedNormalStd(μs, [σ]))
end

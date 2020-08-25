struct GaussianMixtures{M, C}
    mixing::M
    components::C
end

# Default 1D Gaussian mixtures
OneDimGaussianMixtures() = GaussianMixtures(
    [0.25, 0.4, 0.35],
    BroadcastedNormalStd(
        [-1.0 0.0 1.0], [0.25]
    )
)

# Default 2D Gaussian mixtures
TwoDimGaussianMixtures() = GaussianMixtures(
    [0.25, 0.4, 0.35],
    BroadcastedNormalStd(
        [-1.0 0.0 1.0; -1.0 0.0 1.0], [0.25]
    )
)

function rand(gms::GaussianMixtures, n::Int=1)
    @unpack mixing, components = gms
    components = [
        BroadcastedNormalStd(mean(components)[:,i], std(components)) for i in 1:length(mixing)
    ]
    samples = []
    for _ in 1:n
        i = rand(Categorical(mixing))
        g = components[i]
        push!(samples, rand(g))
    end
    return cat(samples...; dims=ndims(samples[1])+1)
end

function _logpdf(gms::GaussianMixtures, x)
    @unpack mixing, components = gms
    lps = logpdf(components, x)
    lps = dropdims(sum(lps; dims=1); dims=1)
    lps_norm = lps .+ log.(mixing)
    return dropdims(logsumexp(lps_norm; dims=1); dims=1)
end

function logpdf(gms::GaussianMixtures, x::AbstractVector)
    x = reshape(x, length(x), 1, 1)
    lp = _logpdf(gms, x)
    return only(lp)
end

function logpdf(gms::GaussianMixtures, x::AbstractMatrix)
    dim, n = size(x)
    x = reshape(x, dim, 1, n)
    return _logpdf(gms, x)
end

function get_target(gms::GaussianMixtures)
    logdensity(x) = logpdf(gms, x)
    return (
        dim = size(gms.components.m, 1), 
        logdensity = logdensity, 
        get_grad = x -> get_∂ℓπ∂θ_reversediff(logdensity, x),
    )
end

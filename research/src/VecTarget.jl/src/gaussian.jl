# High-dimensional standard Gaussian
HighDimGaussian(dim::Int) = BroadcastedNormalStd(zeros(dim), ones(dim))

function get_target(normal::BroadcastedNormalStd)
    ℓπ(θ::AbstractVecOrMat) = logpdf(normal, θ)

    ℓπ(θ::AbstractVector) = sum(logpdf(normal, θ))

    ℓπ(θ::AbstractMatrix) = dropdims(sum(logpdf(normal, θ); dims=1); dims=1)

    function ∂ℓπ∂θ(normal, x::AbstractVecOrMat{T}) where {T}
        diff = x .- normal.m
        v = -(log(2 * T(pi)) .+ logvar(normal) .+ diff .* diff ./ var(normal)) / 2
        g = -diff
        return v, g
    end

    function ∂ℓπ∂θ(θ::AbstractVector)
        v, g = ∂ℓπ∂θ(normal, θ)
        return sum(v), g
    end

    function ∂ℓπ∂θ(θ::AbstractMatrix)
        v, g = ∂ℓπ∂θ(normal, θ)
        return dropdims(sum(v; dims=1); dims=1), g
    end

    return (dim=size(normal.m, 1), logdensity=ℓπ, get_grad=(x -> ∂ℓπ∂θ))
end

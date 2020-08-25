# High-dimensional standard Gaussian

struct HighDimGaussian{Tm, Ts}
    m::Tm
    s::Ts
end

HighDimGaussian(dim::Int) = HighDimGaussian(zeros(dim), ones(dim))

function ℓπ_gaussian(g::AbstractVecOrMat{T}, s) where {T}
    return .-(log(2 * T(pi)) .+ 2 .* log.(s) .+ abs2.(g) ./ s.^2) ./ 2
end

ℓπ_gaussian(m, s, x) = ℓπ_gaussian(m .- x, s)

function ∇ℓπ_gaussianl(m, s, x)
    g = m .- x
    v = ℓπ_gaussian(g, s)
    return v, g
end

function get_ℓπ(g::HighDimGaussian)
    ℓπ(x::AbstractVector) = sum(ℓπ_gaussian(g.m, g.s, x))
    ℓπ(x::AbstractMatrix) = dropdims(sum(ℓπ_gaussian(g.m, g.s, x); dims=1); dims=1)
    return ℓπ
end

function get_∇ℓπ(g::HighDimGaussian)
    function ∇ℓπ(x::AbstractVector)
        val, grad = ∇ℓπ_gaussianl(g.m, g.s, x)
        return sum(val), grad
    end
    function ∇ℓπ(x::AbstractMatrix)
        val, grad = ∇ℓπ_gaussianl(g.m, g.s, x)
        return dropdims(sum(val; dims=1); dims=1), grad
    end
    return ∇ℓπ
end

function get_target(normal::HighDimGaussian)
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

    return (
        dim=size(normal.m, 1), 
        logdensity=ℓπ, 
        get_grad=x -> ∂ℓπ∂θ
        )
end

struct Banana end

function get_target(banana::Banana)
    function _ℓπ(θ::AbstractVecOrMat)
        x1, x2 = θ[1,:], θ[2,:]
        U = (1 .- x1).^2 + 10(x2 - x1.^2).^2
        return -U
    end

    ℓπ(θ::AbstractVector) = only(_ℓπ(θ))

    ℓπ(θ::AbstractMatrix) = _ℓπ(θ)

    function _∂ℓπ∂θ(θ::AbstractVecOrMat)
        x1, x2 = θ[1,:], θ[2,:]
        x1sq = x1.^2
        x2x1sq_diff = x2 - x1sq
        dx1 = 2(1 .- x1) + 40x2x1sq_diff .* x1
        dx2 = -20x2x1sq_diff
        return ℓπ(θ), cat(dx1', dx2'; dims=1)
    end

    function ∂ℓπ∂θ(θ::AbstractVector)
        v, g = _∂ℓπ∂θ(θ)
        return v, dropdims(g; dims=2)
    end

    ∂ℓπ∂θ(θ::AbstractMatrix) = _∂ℓπ∂θ(θ)

    return (dim=2, logdensity=ℓπ, get_grad=x -> ∂ℓπ∂θ)
end

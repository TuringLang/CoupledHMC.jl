using AdvancedHMC: AdvancedHMC, PhasePoint, sample_init, AbstractMomentumRefreshment

struct SharedRefreshment <: AbstractMomentumRefreshment end
struct ContractiveRefreshment <: AbstractMomentumRefreshment end


function AdvancedHMC.refresh(
    rng::Union{AbstractRNG, AbstractVector{<:AbstractRNG}},
    ::SharedRefreshment,
    h::Hamiltonian,
    z::PhasePoint
)
    return phasepoint(h, z.θ, rand(rng, h.metric))
end

function AdvancedHMC.refresh(
    rng::Union{AbstractRNG, AbstractVector{<:AbstractRNG}},
    ::ContractiveRefreshment,
    h::Hamiltonian,
    z::PhasePoint
)
    κ = 1.0
    x, y = z.θ[:,1], z.θ[:,2]
    Δ = x - y
    normΔ = norm(Δ)
    rx = rand(rng, h.metric)[:,1]
    if iszero(normΔ)
        ry = rx
    else
        Δ̄ = Δ / normΔ
        logu = log(rand())
        prob = logpdf(Normal(0, 1), Δ̄' * rx + κ * normΔ) - logpdf(Normal(0, 1), Δ̄' * rx)
        ry = logu < prob ? rx + κ * Δ : rx - 2 * (Δ̄' * rx) * Δ̄
    end
    return phasepoint(h, z.θ, cat(rx, ry; dims=2))
end

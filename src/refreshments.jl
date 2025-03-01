struct SharedRefreshment <: AdvancedHMC.AbstractMomentumRefreshment end
struct ContractiveRefreshment <: AdvancedHMC.AbstractMomentumRefreshment end


function AdvancedHMC.refresh(
    rng::Union{AbstractRNG, AbstractVector{<:AbstractRNG}},
    ::SharedRefreshment,
    h::AdvancedHMC.Hamiltonian,
    z::AdvancedHMC.PhasePoint
)
    # TODO(tor): We're not passing `rng` here. We should find a better way.
    p = rand(h.metric, h.kinetic)[:, 1]
    p_mat = repeat(p, 1, 2)
    res = AdvancedHMC.phasepoint(h, z.θ, p_mat)
    return res
end

function AdvancedHMC.refresh(
    rng::Union{AbstractRNG, AbstractVector{<:AbstractRNG}},
    ::ContractiveRefreshment,
    h::AdvancedHMC.Hamiltonian,
    z::AdvancedHMC.PhasePoint
)
    κ = 1.0
    x, y = z.θ[:,1], z.θ[:,2]
    Δ = x - y
    normΔ = norm(Δ)
    # TODO(tor): We're not passing `rng` here. We should find a better way.
    rx = rand(h.metric, h.kinetic)[:, 1]
    if iszero(normΔ)
        ry = rx
    else
        Δ̄ = Δ / normΔ
        logu = log(rand(rng))
        prob = logpdf(Normal(0, 1), Δ̄' * rx + κ * normΔ) - logpdf(Normal(0, 1), Δ̄' * rx)
        ry = logu < prob ? rx + κ * Δ : rx - 2 * (Δ̄' * rx) * Δ̄
    end
    return AdvancedHMC.phasepoint(h, z.θ, cat(rx, ry; dims=2))
end

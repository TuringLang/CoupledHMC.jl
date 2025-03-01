struct MixtureKernel{
    F<:AbstractFloat, T1<:AdvancedHMC.AbstractMCMCKernel, T2<:AdvancedHMC.AbstractMCMCKernel
} <: AdvancedHMC.AbstractMCMCKernel
    γ::F
    τ1::T1
    τ2::T2
end

function AdvancedHMC.transition(
    rng::AbstractRNG,
    h::Hamiltonian,
    mix::MixtureKernel,
    z::PhasePoint
)
    return if rand(rng) < mix.γ
        AdvancedHMC.transition(rng, h, mix.τ1, z)
    else
        AdvancedHMC.transition(rng, h, mix.τ2, z)
    end
end

function AdvancedHMC.transition(
    rng::AbstractVector{<:AbstractRNG},
    h::Hamiltonian,
    mix::MixtureKernel,
    z::PhasePoint
)
    # TODO: is this always the correct thing to do?
    # Ideally we'd allow different elements in the "batch"/vectorization
    # use different components, BUT this will be faster and the resulting
    # chains should still be valid. Similar to:
    # https://github.com/TuringLang/AdvancedHMC.jl/pull/166#pullrequestreview-367216534
    return if rand(rng) < mix.γ
        AdvancedHMC.transition(rng, h, mix.τ1, z)
    else
        AdvancedHMC.transition(rng, h, mix.τ2, z)
    end
end


struct MixtureKernel{
    F<:AbstractFloat, T1<:AbstractMCMCKernel, T2<:AbstractMCMCKernel
} <: AbstractMCMCKernel
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
    if rand(rng) < mix.γ
        return transition(rng, h, mix.τ1, z)
    else
        return transition(rng, h, mix.τ2, z)
    end
end

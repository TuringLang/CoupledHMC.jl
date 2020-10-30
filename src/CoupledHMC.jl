module CoupledHMC

using RCall, ProgressMeter, Logging
using Random, LinearAlgebra, Statistics, Distances, Distributions, JuMP, Clp, OptimalTransport, AdvancedHMC
using DocStringExtensions: TYPEDEF, TYPEDFIELDS

function __init__()
    R"library(coda)"
end

include("utilities.jl")
export rands

### AdvancedHMC extensions

using AdvancedHMC: AdvancedHMC, PhasePoint, sample_init

const REFRESHMENT = Ref(:shared)

function set_refreshment!(refreshment)
    !(refreshment in (:shared, :contractive)) && error("Unsupoorted refreshment: $refreshment")
    REFRESHMENT[] = refreshment
end

function AdvancedHMC.refresh(
    rng::AbstractVector{<:MersenneTwister},
    z::PhasePoint,
    h::Hamiltonian
)
    if REFRESHMENT[] == :shared
        z = phasepoint(h, z.θ, rand(rng, h.metric))
    elseif REFRESHMENT[] == :contractive
        κ = 1.0
        x, y = z.θ[:,1], z.θ[:,2]
        Δ = x - y
        normΔ = norm(Δ)
        Δ̄ = Δ / normΔ
        rx = rand(rng, h.metric)[:,1]
        logu = log(rand())
        prob = logpdf(Normal(0, 1), Δ̄' * rx + κ * normΔ) - logpdf(Normal(0, 1), Δ̄' * rx)
        ry = logu < prob ? rx + κ * Δ : rx - 2 * (Δ̄' * rx) * Δ̄
        z = phasepoint(h, z.θ, cat(rx, ry; dims=2))
    end
    return z
end

struct HMCIterator
    rng
    h
    τ
    θ0
end

# FIXME: Adaptation is not supported.
function Base.iterate(iter::HMCIterator, state=sample_init(iter.rng, iter.h, iter.θ0)[2])
    state = step(iter.rng, iter.h, iter.τ, state.z)
    return (state.z.θ, state)
end

include("couplings.jl")
export IndependentCoupling, QuantileCoupling, MaximalCoupling, OTCoupling, ApproximateOTCoupling

include("kernels.jl")
const MetropolisTS = EndPointTS
export set_refreshment!, MetropolisTS, MultinomialTS, CoupledMultinomialTS

### CoupledHMC abstractions
abstract type AbstractSampler end

"""
$(TYPEDEF)
HMC (without coupling) with trajectory sampler `TS`, step size `ϵ` and step number `L`.
# Fields
$(TYPEDFIELDS)
"""
Base.@kwdef struct HMCSampler{
    _TS<:AbstractTrajectorySampler, F<:AbstractFloat, R<:Function
} <: AbstractSampler
    rinit::R
    TS::Type{_TS}
    ϵ::F
    L::Int
end

"""
$(TYPEDEF)
Coupled HMC with trajectory sampler `TS`, step size `ϵ` and step number `L`.
A mixture with a MH kernel with standard deviation `σ` can be enabled by using a
mixture probability `γ` larger than 0. A maximum reflection MH kernel can be 
enabled by using a tuning parameter `κ` larger than 0.
# Fields
$(TYPEDFIELDS)
"""
Base.@kwdef struct CoupledHMCSampler{
    _TS<:AbstractTrajectorySampler, F<:AbstractFloat, R<:Function
} <: AbstractSampler
    rinit::R
    TS::Type{_TS}
    ϵ::F
    L::Int
    γ::F=1/20
    σ::F=1e-3
    κ::F=0.0
end
export HMCSampler, CoupledHMCSampler

include("analysis.jl")
export get_k_m, does_meet, τ_of, H_of, i_of, v_of

### Sampling interface for `AbstractSampler`
function get_ahmc_primitives(target, alg::HMCSampler, theta0)
    if isnothing(theta0)
        theta0 = alg.rinit(target.dim)
    end
    rng = MersenneTwister(randseed())
    metric = UnitEuclideanMetric(target.dim)
    hamiltonian = Hamiltonian(metric, target.logdensity, target.get_grad(theta0))
    integrator = Leapfrog(alg.ϵ)
    # Get the corresponding marginal trajectory sampler
    TS = if alg.TS <: EndPointTS
        EndPointTS
    elseif alg.TS <: CoupledMultinomialTS
        MultinomialTS
    else
        error("Marginal sampler for `$(alg.TS)` is not defined.")
    end
    proposal = StaticTrajectory{TS}(integrator, alg.L)
    return rng, hamiltonian, proposal, theta0
end

function get_ahmc_primitives(target, alg::CoupledHMCSampler, theta0)
    if isnothing(theta0)
        # Sample (X_0, Y_0)
        x0, y0 = alg.rinit(target.dim), alg.rinit(target.dim)
        # Transit X_0 to X_1
        samples = sample(target, HMCSampler(rinit=alg.rinit, TS=alg.TS, ϵ=alg.ϵ, L=alg.L), 1; theta0=x0)
        # Return (X_1, Y_0)
        theta0 = cat(samples[end], y0; dims=2)
    end
    rng = MersenneTwister(fill(randseed(), 2))
    metric = UnitEuclideanMetric((target.dim, 2))
    hamiltonian = Hamiltonian(metric, target.logdensity, target.get_grad(theta0))
    integrator = Leapfrog(fill(alg.ϵ, 2))
    proposal = 
    if iszero(alg.κ)
        MixtureProposal(
            alg.γ, MaxCoupledMH(alg.σ), StaticTrajectory{alg.TS}(integrator, alg.L)
        )
    else
        MixtureProposal(
            alg.γ, RefMaxCoupledMH(alg.σ, alg.κ), StaticTrajectory{alg.TS}(integrator, alg.L),
        )
    end
    return rng, hamiltonian, proposal, theta0
end

function sample(target, alg::HMCSampler, n_samples::Int; theta0=nothing, progress=false)
    rng, hamiltonian, proposal, theta0 = get_ahmc_primitives(target, alg, theta0)
    samples, stats = AdvancedHMC.sample(
        rng, hamiltonian, proposal, theta0, n_samples; progress=progress, verbose=false
    )
    return samples
end

function sample_until_meeting(
    rng, hamiltonian, proposal, theta0; n_samples_max=100_000, progress=false
)
    progress = progress ? ProgressUnknown("Sampling until meeting. urrent iteration:") : nothing
    iter = HMCIterator(rng, hamiltonian, proposal, theta0)
    i = Ref(0)
    return collect(
        takeuntil(iter) do x
            i[] += 1
            if i[] <= n_samples_max
                meet = does_meet(x)
                meet && !isnothing(progress) && finish!(progress)
                meet
            else
                true
            end
        end
    )
end

sample_until_meeting(target, alg::CoupledHMCSampler; theta0=nothing, kwargs...) =
    sample_until_meeting(get_ahmc_primitives(target, alg, theta0)...; kwargs...)

function sample(target, alg::CoupledHMCSampler, n_samples::Int; theta0=nothing, progress=false)
    rng, hamiltonian, proposal, theta0 = get_ahmc_primitives(target, alg, theta0)
    samples = sample_until_meeting(rng, hamiltonian, proposal, theta0; n_samples_max=n_samples)
    n_samples_left = n_samples - length(samples)
    samples_after_meeting = 
    if n_samples_left > 0
        samples_after_meeting = sample(
            target, HMCSampler(rinit=alg.rinit, TS=alg.TS, ϵ=alg.ϵ, L=alg.L), n_samples_left; 
            theta0=samples[end][:,1], progress=progress
        )
        # Duplicate the `n_samples_left` part to make the dimension consistent
        map(s -> cat(s, s; dims=2), samples_after_meeting)
    else
        []
    end
    return cat(samples, samples_after_meeting; dims=1)
end

export sample_until_meeting, sample

end # module

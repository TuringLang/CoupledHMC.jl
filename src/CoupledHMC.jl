module CoupledHMC

using RCall
using ProgressMeter
using Logging
using Random
using LinearAlgebra
using Statistics
using Distances
using Distributions
using JuMP
using Clp
using OptimalTransport
using AdvancedHMC
using DocStringExtensions: TYPEDEF, TYPEDFIELDS
using SimpleUnPack: @unpack

using LogDensityProblems: LogDensityProblems
using VecTargets: VecTargets

function __init__()
    R"library(coda)"
end

include("utilities.jl")
export rands

### AdvancedHMC extensions
include("refreshments.jl")
export SharedRefreshment, ContractiveRefreshment

struct HMCIterator
    rng
    h
    κ
    θ0
end

# FIXME: Adaptation is not supported.
function Base.iterate(iter::HMCIterator, state=AdvancedHMC.sample_init(iter.rng, iter.h, iter.θ0)[2])
    state = AdvancedHMC.transition(iter.rng, iter.h, iter.κ, state.z)
    return (state.z.θ, state)
end

include("couplings.jl")
export IndependentCoupling, QuantileCoupling, MaximalCoupling, OTCoupling, ApproximateOTCoupling

include("kernels.jl")
include("trajectory_samplers.jl")

const MetropolisTS = EndPointTS
export EndPointTS, MetropolisTS, MultinomialTS, CoupledMultinomialTS

### CoupledHMC abstractions
abstract type AbstractSampler end

"""
$(TYPEDEF)
HMC (without coupling) with trajectory sampler `TS`, step size `ϵ` and step number `L`.
# Fields
$(TYPEDFIELDS)
"""
Base.@kwdef struct HMCSampler{
    _TS<:AdvancedHMC.AbstractTrajectorySampler, 
    F<:Union{AbstractFloat, Missing}, 
    I<:Union{Int, Missing}, 
    R<:Function,
    MR<:AdvancedHMC.AbstractMomentumRefreshment
} <: AbstractSampler
    rinit::R
    TS::Type{_TS}
    ϵ::F=missing
    L::I=missing
    momentum_refreshment::MR=SharedRefreshment()
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
    _TS<:AdvancedHMC.AbstractTrajectorySampler,
    F<:AbstractFloat,
    R<:Function,
    MR<:AdvancedHMC.AbstractMomentumRefreshment
} <: AbstractSampler
    rinit::R
    TS::Type{_TS}
    ϵ::F
    L::Int
    γ::F=1/20
    σ::F=1e-3
    κ::F=0.0
    momentum_refreshment::MR=SharedRefreshment()
end
export HMCSampler, CoupledHMCSampler

include("analysis.jl")
export get_k_m, does_meet, τ_of, H_of, i_of, v_of

# By default, we work with LogDensitProblems.jl models.
logdensity(f, x::AbstractVector) = LogDensityProblems.logdensity(f, x)
logdensity(f, x::AbstractMatrix) = map(Base.Fix1(logdensity, f), eachcol(x))

function logdensity_and_gradient(f, x::AbstractVector)
    return LogDensityProblems.logdensity_and_gradient(f.f, x)
end
function logdensity_and_gradient(f, x::AbstractMatrix)
    result = map(Base.Fix1(LogDensityProblems.logdensity_and_gradient, f.f), eachcol(x))
    lps = map(first, result)
    grads = map(last, result)
    return stack(lps), stack(grads)
end

target_dim(f) = LogDensityProblems.dimension(f.f)

# For anything else that implements VecTargets.jl interface.
const VecTargetModelTypes = Union{
    VecTargets.HighDimGaussian,
    VecTargets.LogisticRegression,
    VecTargets.LogGaussianCoxPointProcess,
    VecTargets.Banana,
    VecTargets.Funnel,
    VecTargets.Spiral
}
logdensity(f::VecTargetModelTypes, x::AbstractVector) = VecTargets.logpdf(f, x)
logdensity(f::VecTargetModelTypes, x::AbstractMatrix) = VecTargets.logpdf(f, x)

logdensity_and_gradient(f::VecTargetModelTypes, x::AbstractVector) = VecTargets.logpdf_grad(f, x)
logdensity_and_gradient(f::VecTargetModelTypes, x::AbstractMatrix) = VecTargets.logpdf_grad(f, x)

target_dim(f::VecTargetModelTypes) = VecTargets.dim(f)

### Sampling interface for `AbstractSampler`
function get_ahmc_primitives(target, alg::HMCSampler, theta0)
    rng = MersenneTwister(randseed())

    if isnothing(theta0)
        theta0 = alg.rinit(rng, target_dim(target))
    end

    metric = AdvancedHMC.UnitEuclideanMetric(target_dim(target))
    hamiltonian = AdvancedHMC.Hamiltonian(
        metric,
        Base.Fix1(logdensity, target),
        Base.Fix1(logdensity_and_gradient, target)
    )

    momentum_refreshment = if (alg.momentum_refreshment isa SharedRefreshment) || (alg.momentum_refresment isa ContractiveRefreshment)
        AdvancedHMC.FullMomentumRefreshment()
    else
        alg.momentum_refreshment
    end

    if ismissing(alg.ϵ) && ismissing(alg.L)
        integrator = AdvancedHMC.Leapfrog(find_good_stepsize(rng, hamiltonian, theta0))
        @assert alg.TS <: AdvancedHMC.MultinomialTS
        trajectory = AdvancedHMC.Trajectory{AdvancedHMC.MultinomialTS}(integrator, AdvancedHMC.GeneralisedNoUTurn())
        adaptor = AdvancedHMC.StanHMCAdaptor(
            AdvancedHMC.MassMatrixAdaptor(metric),
            AdvancedHMC.StepSizeAdaptor(0.8, integrator)
        )
        kernel = AdvancedHMC.HMCKernel(momentum_refreshment, trajectory)
        return rng, hamiltonian, kernel, adaptor, theta0
    else
        integrator = AdvancedHMC.Leapfrog(alg.ϵ)
        # Get the corresponding marginal trajectory sampler
        # TODO(tor): Improve this.
        TS = if alg.TS <: AdvancedHMC.EndPointTS
            AdvancedHMC.EndPointTS
        elseif alg.TS <: CoupledMultinomialTS || alg.TS <: AdvancedHMC.MultinomialTS
            AdvancedHMC.MultinomialTS
        else
            error("Marginal sampler for `$(alg.TS)` is not defined.")
        end
        trajectory = AdvancedHMC.Trajectory{TS}(integrator, AdvancedHMC.FixedNSteps(alg.L))
        kernel = AdvancedHMC.HMCKernel(momentum_refreshment, trajectory)

        return rng, hamiltonian, kernel, theta0
    end
end

# TODO(tor): Improve this so we can run on more targets.
function get_ahmc_primitives(target, alg::CoupledHMCSampler, theta0)
    rng_init = MersenneTwister(randseed())
    rng = MersenneTwister(fill(randseed(), 2))

    if isnothing(theta0)
        # Sample (X_0, Y_0)
        x0 = alg.rinit(rng_init, target_dim(target))
        y0 = alg.rinit(rng_init, target_dim(target))
        # Transit X_0 to X_1
        samples = sample(target, HMCSampler(rinit=alg.rinit, TS=alg.TS, ϵ=alg.ϵ, L=alg.L), 1; theta0=x0)
        # Return (X_1, Y_0)
        theta0 = cat(samples[end], y0; dims=2)
    end

    metric = UnitEuclideanMetric((target_dim(target), 2))
    hamiltonian = AdvancedHMC.Hamiltonian(
        metric,
        Base.Fix1(logdensity, target),
        Base.Fix1(logdensity_and_gradient, target)
    )

    integrator = Leapfrog(fill(alg.ϵ, 2))
    trajectory = Trajectory{alg.TS}(integrator, FixedNSteps(alg.L))
    kernel_hmc = HMCKernel(alg.momentum_refreshment, trajectory)
    kernel = if iszero(alg.κ)
        kernel_mh = MaxCoupledMH(alg.σ)
        MixtureKernel(
            alg.γ, kernel_mh, kernel_hmc
        )
    else
        kernel_mh = RefMaxCoupledMH(alg.σ, alg.κ)
        MixtureKernel(
            alg.γ, kernel_hm, kernel_hmc,
        )
    end
    return rng, hamiltonian, kernel, theta0
end

function sample(target, alg::HMCSampler, n_samples::Int; theta0=nothing, progress=false)
    rng, hamiltonian, proposal, theta0 = get_ahmc_primitives(target, alg, theta0)
    samples, stats = AdvancedHMC.sample(
        rng, hamiltonian, proposal, theta0, n_samples; progress=progress, verbose=false
    )
    return samples
end

function sample(target, alg::HMCSampler{_TS, Missing, Missing}, n_samples::Int, n_adapts::Int; theta0=nothing, progress=false) where {_TS}
    rng, hamiltonian, proposal, adaptor, theta0 = get_ahmc_primitives(target, alg, theta0)
    samples, stats = AdvancedHMC.sample(
        rng, hamiltonian, proposal, theta0, n_samples, adaptor, n_adapts; progress=progress, verbose=false
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

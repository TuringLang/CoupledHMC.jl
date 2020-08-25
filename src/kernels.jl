using AdvancedHMC: AdvancedHMC, AbstractProposal, AbstractTrajectory, @unpack, step,
                   phasepoint, mh_accept_ratio, accept_phasepoint!, Transition, energy,
                   AbstractRNG, AbstractTrajectorySampler, randcat, logsumexp, rand_coupled
import AdvancedHMC: transition, sample_phasepoint

struct MixtureProposal{
    F<:AbstractFloat, T1<:AbstractProposal, T2<:AbstractProposal
} <: AbstractProposal
    γ::F
    τ1::T1
    τ2::T2
end

function transition(rng, mix::MixtureProposal, h, z)
    if rand() < mix.γ
        return transition(rng, mix.τ1, h, z)
    else
        return transition(rng, mix.τ2, h, z)
    end
end

# TODO: merge RefMaxCoupledMH and MaxCoupledMH

struct RefMaxCoupledMH{F<:AbstractFloat} <: AbstractProposal
    σ::F
    κ::F
end

"""
Sample from reflection max coupling of Normal(0, `sqrtD^2` * I), Normal(0, `sqrtD^2` * I) 
given position `mu1` and `mu2` and where D is diagonal, specified by sqrt(D) and 1/sqrt(D).
`kappa` is a tuning parameter and when `kappa == 0`, this reduces to maximal coupling for MH.
Code is ported from: 
https://github.com/pierrejacob/statisfaction-code/blob/master/2019-09-stan-logistic.R#L241-L264
with minimal and necessary editions.
"""
function reflmaxcoupling(rng, mu1, mu2, sqrtD, kappa)
    dim = size(mu1, 1)
    momentum1 = randn(dim)
    local momentum2, samesame
    logu = log(rand())
    z = (mu1 - mu2) / sqrtD
    normz = sqrt(sum(z.^2))
    evector = z / normz
    edotxi = dot(evector, momentum1)
    if logu < sum(logpdf.(Normal(0, 1), edotxi + kappa * normz)) - sum(logpdf.(Normal(0, 1), edotxi))
        momentum2 = momentum1 + kappa * z
        samesame = true
    else
        if iszero(normz)    # otherwise it will give numeric erros and ended up rejecting
            momentum2 = zeros(dim)  # this is equivalent to rejection
        else
            momentum2 = momentum1 - 2 * edotxi * evector
        end
        samesame = false
    end
    momentum1 = momentum1 * sqrtD
    momentum2 = momentum2 * sqrtD
    return (momentum1=momentum1, momentum2=momentum2, samesame=samesame)
end

function transition(rng, mh::RefMaxCoupledMH, h, z)
    H0 = energy(z)
    @unpack θ, r = z
    res = reflmaxcoupling(rng, θ[:,1], θ[:,2], mh.σ, mh.κ)
    θ = 
    let θ1 = θ[:,1] + res.momentum1
        cat(θ1, res.samesame ? θ1 : θ[:,2] + res.momentum2; dims=2)
    end
    z′ = phasepoint(h, θ, r)
    is_accept, α = mh_accept_ratio(rng, energy(z), energy(z′))
    z = accept_phasepoint!(z, z′, is_accept)
    H = energy(z)
    tstat = (
        is_accept = is_accept,
        acceptance_rate = α,
        does_meet = all(is_accept) && res.samesame,
    )
    return Transition(z, tstat)
end

struct MaxCoupledMH{F<:AbstractFloat} <: AbstractProposal
    σ::F
end

# Ref: https://github.com/pierrejacob/debiasedhmc/blob/master/inst/scalingdimension/scaling.hmc.meetings.R#L101-L142
function transition(rng, mh::MaxCoupledMH, h, z)
    H0 = energy(z)
    @unpack θ, r = z
    x, y = θ[:,1], θ[:,2]
    p, q = MvNormal(x, mh.σ), MvNormal(y, mh.σ)
    x = rand(p)
    does_meet = false
    θ =
    if logpdf(p, x) + log(rand()) <= logpdf(q, x)
        does_meet = true
        cat(x, x; dims=2)
    else
        local y′
        while true
            y′ = rand(q)
            if logpdf(q, y′) + log(rand()) > logpdf(p, y′)
                break
            end
        end
        cat(x, y′; dims=2)
    end
    z′ = phasepoint(h, θ, r)
    is_accept, α = mh_accept_ratio(rng, energy(z), energy(z′))
    z = accept_phasepoint!(z, z′, is_accept)
    H = energy(z)
    tstat = (
        is_accept = is_accept,
        acceptance_rate = α,
        does_meet = does_meet,
    )
    return Transition(z, tstat)
end

struct CoupledMultinomialTS{C<:AbstractCoupling} <: AbstractTrajectorySampler end

function sample_phasepoint(rng, τ::StaticTrajectory{CoupledMultinomialTS{C}}, h, z) where {C}
    n_steps = abs(τ.n_steps)
    # TODO: Deal with vectorized-mode generically.
    #       Currently the direction of multiple chains are always coupled
    n_steps_fwd = rand_coupled(rng, 0:n_steps) 
    zs_fwd = step(τ.integrator, h, z, n_steps_fwd; fwd=true, full_trajectory=Val(true))
    n_steps_bwd = n_steps - n_steps_fwd
    zs_bwd = step(τ.integrator, h, z, n_steps_bwd; fwd=false, full_trajectory=Val(true))
    zs = vcat(reverse(zs_bwd)..., z, zs_fwd...)
    ℓweights = -energy.(zs)
    if eltype(ℓweights) <: AbstractVector
        ℓweights = cat(ℓweights...; dims=2)
    end
    unnorm_ℓprob = ℓweights
    prob = exp.(unnorm_ℓprob .- logsumexp(unnorm_ℓprob; dims=2))

    if C == QuantileCoupling || C == MaximalCoupling
        coupling = C(prob[1,:], prob[2,:])
    elseif C == OTCoupling || C == ApproximateOTCoupling
        τ¹ = cat(map(z -> z.θ[:,1], zs)...; dims=2)
        τ² = cat(map(z -> z.θ[:,2], zs)...; dims=2)
        coupling = C(prob[1,:], prob[2,:], τ¹, τ²)
    else
        error("Unknown coupling method for CoupledMultinomialTS `$C`.")
    end

    i, j = rand(coupling)
    
    z′ = similar(z)
    foreach(enumerate([i, j])) do (i_chain, i_step)
        zi = zs[i_step]
        z′.θ[:,i_chain] = zi.θ[:,i_chain]
        z′.r[:,i_chain] = zi.r[:,i_chain]
        z′.ℓπ.value[i_chain] = zi.ℓπ.value[i_chain]
        z′.ℓπ.gradient[:,i_chain] = zi.ℓπ.gradient[:,i_chain]
        z′.ℓκ.value[i_chain] = zi.ℓκ.value[i_chain]
        z′.ℓκ.gradient[:,i_chain] = zi.ℓκ.gradient[:,i_chain]
    end
    
    # Computing adaptation statistics for dual averaging as done in NUTS
    Hs = -ℓweights
    ΔH = Hs .- energy(z)
    α = exp.(min.(0, -ΔH))  # this is a matrix for vectorized mode and a vector otherwise
    α = typeof(α) <: AbstractVector ? mean(α) : vec(mean(α; dims=2))
    return z′, true, α
end

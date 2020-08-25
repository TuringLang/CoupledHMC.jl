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

using Hungarian, JuMP, Clp

abstract type AbstractContractionMethod end

struct NoContraction <: AbstractContractionMethod end
struct Pairing <: AbstractContractionMethod end
struct OT <: AbstractContractionMethod end

struct ContractiveMultinomialTS{C<:AbstractContractionMethod} <: AbstractTrajectorySampler end

distancebetween(z1, z2) = norm(z1.θ[:,1] - z2.θ[:,2])

"""Find the maximum probability of entering branch 1 via binary search"""
function find_prob_cond_binsearch(prob_q, prob_r; n_iter_max=1_000)
    low, high = 0.0, 1.0
    prob_cond = 0.5
    iter = 1
    while iter <= n_iter_max
        prob_debias = prob_q - prob_r * prob_cond
        if all(prob_debias .>= 0)
            prob_cond′ = (prob_cond + high) / 2
            low = prob_cond
            if abs(prob_cond′ - prob_cond) < 1e-6
                return prob_cond
            end
        else
            prob_cond′ = (low + prob_cond) / 2
            high = prob_cond
        end
        prob_cond = prob_cond′
        iter = iter + 1
    end
    return prob_cond
end

function rand_joint(rng, pairs, prob_p, prob_q; n_iter_max=1_000)
    i = rand(Categorical(prob_p))
    # Probability vector for the twisted distribution
    prob_r = [prob_p[pairs[k]] for k in 1:length(pairs)]
    prob_cond = find_prob_cond_binsearch(prob_q, prob_r)
    if prob_cond < 1e-3
        i, j = randcat(rng, cat(prob_p, prob_q; dims=2))
        return (i=i, j=j)
    else
        if rand() <= prob_cond
            return (i=i, j=pairs[i])
        else
            # Debiased probability vector
            prob_debias = (prob_q - prob_r * prob_cond) / (1 - prob_cond)
            try
                j = rand(Categorical(prob_debias))
                return (i=i, j=j)
            catch e
                println(e)
                println(prob_p, prob_q)
                println(prob_cond)
                println(prob_debias)
            end
        end
    end
end

"""Pairing by the Hungarian algorithm"""
function pair_hungarian(M, prob_p, prob_q)
    assignment, cost = hungarian(M .* prob_p')
    return assignment
end

euclidsq(x::T, y::T) where {T<:AbstractVector} = 
    euclidsq(reshape(x, 1, length(x)), reshape(y, 1, length(y)))

function euclidsq(X::T, Y::T) where {T<:AbstractMatrix}
    XiXj = transpose(X) * Y
    x² = sum(X .^ 2; dims=1)
    y² = sum(Y .^ 2; dims=1)
    return transpose(x²) .+ y² - 2XiXj
end

function euclidsq(zs::Vector)
    x = cat(map(z -> z.θ[:,1], zs)...; dims=2)
    y = cat(map(z -> z.θ[:,2], zs)...; dims=2)
    return euclidsq(x, y)
end

function rand_joint(rng::AbstractVector{<:AbstractRNG}, G)
    I = collect(Iterators.product(1:size(G, 1), 1:size(G, 2)))
    i, j = I[AdvancedHMC.randcat(rng[1], vec(G))]
    rand(rng[2])    # dummy call to sync RNG
    return (i=i, j=j)
end

function init_ot_model(p::AbstractVector, q::AbstractVector)
    model = Model(Clp.Optimizer)
    MOI.set(model, MOI.Silent(), true) # turn off logging
    
    # variable
    @variable(model, γ[1:length(p), 1:length(q)])
    
    # constraints
    @constraint(model, marginal_p_con, vec(sum(γ; dims = 2)) .== p)
    @constraint(model, marginal_q_con, vec(sum(γ; dims = 1)) .== q)

    @constraint(model, γ .≥ 0)
    
    return model
end

function sample_phasepoint(rng, τ::StaticTrajectory{ContractiveMultinomialTS{C}}, h, z) where {C}
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

    if C == NoContraction
        i, _ = randcat(rng, prob')
        _, j = randcat(rng, prob')
    end
    
    if C == Pairing
        M = euclidsq(zs)
        pairs = pair_hungarian(M, prob[1,:], prob[2,:])
        i, j = rand_joint(rng, pairs, prob[1,:], prob[2,:])
    end

    if C == OT
        M = euclidsq(zs)
        G = let ot_model = init_ot_model(prob[1,:], prob[2,:])
            @objective(ot_model, Min, sum(ot_model.obj_dict[:γ] .* M))
            optimize!(ot_model)
            value.(ot_model.obj_dict[:γ])
        end
        i, j = rand_joint(rng, G)
    end
    
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

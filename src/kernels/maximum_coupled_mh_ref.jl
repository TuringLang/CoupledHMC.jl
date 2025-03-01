struct RefMaxCoupledMH{F<:AbstractFloat} <: AdvancedHMC.AbstractMCMCKernel
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

    # TODO(tor): Use `randexp` and sign-flip instead.
    logu = log(rand(rng))
    z = (mu1 - mu2) / sqrtD
    normz = sqrt(sum(z.^2))
    evector = z / normz
    edotxi = dot(evector, momentum1)

    momentum2, sameasme = if logu < sum(logpdf.(Normal(0, 1), edotxi + kappa * normz)) - sum(logpdf.(Normal(0, 1), edotxi))
        (momentum1 + kappa * z, true)
    else
        (iszero(normz) ? zeros(dim) : momentum1 - 2 * edotxi * evector, false)
    end

    momentum1 = momentum1 * sqrtD
    momentum2 = momentum2 * sqrtD
    return (momentum1=momentum1, momentum2=momentum2, samesame=samesame)
end

function transition(rng, mh::RefMaxCoupledMH, h, z)
    H0 = AdvancedHMC.energy(z)
    @unpack θ, r = z
    res = reflmaxcoupling(rng, θ[:,1], θ[:,2], mh.σ, mh.κ)
    θ = 
    let θ1 = θ[:,1] + res.momentum1
        cat(θ1, res.samesame ? θ1 : θ[:,2] + res.momentum2; dims=2)
    end
    z′ = AdvancedHMC.phasepoint(h, θ, r)
    is_accept, α = AdvancedHMC.mh_accept_ratio(rng, AdvancedHMC.energy(z), AdvancedHMC.energy(z′))
    z = AdvancedHMC.accept_phasepoint!(z, z′, is_accept)
    H = AdvancedHMC.energy(z)
    tstat = (
        is_accept = is_accept,
        acceptance_rate = α,
        does_meet = all(is_accept) && res.samesame,
    )
    return AdvancedHMC.Transition(z, tstat)
end

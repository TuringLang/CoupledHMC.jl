import AdvancedHMC: transition, energy, Transition, mh_accept_ratio, accept_phasepoint!, @unpack

struct MaxCoupledMH{F<:AbstractFloat} <: AbstractMCMCKernel
    σ::F
end

# Ref: https://github.com/pierrejacob/debiasedhmc/blob/master/inst/scalingdimension/scaling.hmc.meetings.R#L101-L142
function transition(rng, h, mh::MaxCoupledMH, z)
    H0 = energy(z)
    @unpack θ, r = z
    x, y = θ[:, 1], θ[:, 2]
    p, q = MvNormal(x, mh.σ), MvNormal(y, mh.σ)
    x = rand(p)
    does_meet = false
    θ = if logpdf(p, x) + log(rand()) <= logpdf(q, x)
        does_meet = true
        cat(x, x; dims=2)
    else
        y′ = rand(q)
        while logpdf(q, y′) + log(rand()) > logpdf(p, y′)
            y′ = rand(q)
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

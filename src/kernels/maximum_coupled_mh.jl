struct MaxCoupledMH{F<:AbstractFloat} <: AdvancedHMC.AbstractMCMCKernel
    σ::F
end

# Ref: https://github.com/pierrejacob/debiasedhmc/blob/master/inst/scalingdimension/scaling.hmc.meetings.R#L101-L142
function AdvancedHMC.transition(rng, h, mh::MaxCoupledMH, z)
    H0 = AdvancedHMC.energy(z)
    @unpack θ, r = z
    x, y = θ[:, 1], θ[:, 2]
    p, q = MvNormal(x, mh.σ), MvNormal(y, mh.σ)
    x = rand(p)
    does_meet = false
    # TODO(tor): Can this not be done using MH ratio function?
    θ = if logpdf(p, x) + log(rand()) <= logpdf(q, x)
        does_meet = true
        cat(x, x; dims=2)
    else
        # `local` is necessary so that we can replace it within the `while` loop
        local y′
        while true
            y′ = rand(q)
            if logpdf(q, y′) + log(rand()) > logpdf(p, y′)
                break
            end
        end
        cat(x, y′; dims=2)
    end
    z′ = AdvancedHMC.phasepoint(h, θ, r)
    is_accept, α = AdvancedHMC.mh_accept_ratio(rng, AdvancedHMC.energy(z), AdvancedHMC.energy(z′))
    z = AdvancedHMC.accept_phasepoint!(z, z′, is_accept)
    H = AdvancedHMC.energy(z)
    tstat = (
        is_accept = is_accept,
        acceptance_rate = α,
        does_meet = does_meet,
    )
    return AdvancedHMC.Transition(z, tstat)
end

using DrWatson
@quickactivate "Research"

import Research
using Comonicon, ProgressMeter, CoupledHMC, VecTargets

@main function exp_gmm(
    refreshment, TS, epsilon::Float64, L::Int;
    n_mc::Int=500, n_samples_max::Int=100, gamma::Float64=1/20, sigma::Float64=1e-3
)
    fname = savename(@ntuple(refreshment, TS, epsilon, L, gamma, sigma, n_mc, n_samples_max), "bson"; connector="-")

    refreshment = Research.parse_refreshment(refreshment)
    TS = Research.parse_trajectory_sampler(TS)
    alg = CoupledHMCSampler(
        rinit=rand, TS=TS, ϵ=epsilon, L=L, γ=gamma, σ=sigma,
        momentum_refreshment=refreshment
    )

    target = TwoDimGaussianMixtures()

    does_meets = Vector{Bool}(undef, n_mc)
    progress = Progress(n_mc)
    Threads.@threads for i in 1:n_mc
        samples = sample_until_meeting(target, alg; n_samples_max=n_samples_max)
        does_meets[i] = does_meet(samples[end])
        next!(progress)
    end 
    n_meeting = sum(does_meets)
    efficiency = round(n_meeting / n_mc; digits=3)
    
    @info "Efficiency: $efficiency"
    wsave(projectdir("results", "gmm", fname), @dict(efficiency))
end

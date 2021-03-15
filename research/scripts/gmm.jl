using DrWatson
@quickactivate "Research"

using Comonicon, ProgressMeter, CoupledHMC, VecTargets
using VecTargets: GaussianMixtures
include(scriptsdir("helper.jl"))

@main function exp_gmm(
    TS, epsilon::Float64, L::Int;
    n_mc::Int=500, n_samples_max::Int=100, gamma::Float64=1/20, sigma::Float64=1e-3,
    refreshment::String="SharedRefreshment",
)
    fname = savename(@ntuple(TS, epsilon, L), "bson"; connector="-")

    refreshment = parse_refreshment(refreshment)
    TS = parse_trajectory_sampler(TS)
    alg = CoupledHMCSampler(
        rinit=rand, TS=TS, ϵ=epsilon, L=L, γ=gamma, σ=sigma,
        momentum_refreshment=refreshment
    )

    target = GaussianMixtures(
        [0.25, 0.4, 0.35], [-1.0 0.0 1.0; -1.0 0.0 1.0], [0.25]
    )
    
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

using DrWatson
@quickactivate "Research"

using Comonicon, ProgressMeter, Statistics, CoupledHMC, VecTargets
include(scriptsdir("helper.jl"))

@main function exp_banana(
    refreshment, TS;
    n_mc::Int=500, n_samples_max::Int=500, 
    epsilon::Float64=1/50, L::Int=50, gamma::Float64=1/20, sigma::Float64=1e-3
)
    fname = savename(@ntuple(refreshment, TS), "bson"; connector="-")

    refreshment = parse_refreshment(refreshment)
    TS = parse_trajectory_sampler(TS)
    alg = CoupledHMCSampler(
        rinit=rand, TS=TS, ϵ=epsilon, L=L, γ=gamma, σ=sigma,
        momentum_refreshment=refreshment
    )

    target = Banana()

    τs = zeros(Int, n_mc)
    progress = Progress(n_mc)
    Threads.@threads for i in 1:n_mc
        samples = sample_until_meeting(target, alg; n_samples_max=n_samples_max)
        τs[i] = length(samples)
        next!(progress)
    end 
    m, s = round(mean(τs); digits=3), round(std(τs); digits=3)
    
    @info "Average meeting time: $m +/- $s"
    wsave(projectdir("results", "banana", fname), @dict(τs))
end

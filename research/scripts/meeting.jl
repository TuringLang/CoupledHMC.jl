using DrWatson
@quickactivate "Research"

using Comonicon, ProgressMeter, Statistics, CoupledHMC, VecTarget

@main function exp_meeting(
    model, TS, epsilon::Float64, L::Int;
    n_mc::Int=10, n_samples_max::Int=1_000, gamma::Float64=1/20, sigma::Float64=1e-3, lambda::Float64=0.01, n_grids::Int=16, 
    saveraw_on::Bool=false
)
    fname = savename(@ntuple(model, TS, epsilon, L), "bson"; connector="-")
    fpath = projectdir("results", "meeting", fname)
    TS = Base.eval(CoupledHMC, Meta.parse(TS)) # parse TS

    if isfile(fpath)
        @info "$fpath exists -- skipping."
    else
        @info "$fpath doesn't exist -- producing."
        if model == "gaussian"
            target = get_target(HighDimGaussian(1_000))
        elseif model == "lr"
            target = get_target(LogisticRegression(datadir(), lambda))
        elseif model == "coxprocess"
            target = get_target(LogGaussianCoxPointProcess(datadir(), n_grids))
        else
            error("Unkown model name $model.")
        end
        alg = CoupledHMCSampler(rinit=randn, TS=TS, ϵ=epsilon, L=L, γ=gamma, σ=sigma)
        τs = zeros(Int, n_mc)
        if saveraw_on
            chains = Vector{Any}(undef, n_mc)
        end
        progress = Progress(n_mc)
        Threads.@threads for i in 1:n_mc
            samples = sample_until_meeting(target, alg; n_samples_max=n_samples_max)
            τs[i] = length(samples)
            if saveraw_on
                chains[i] = samples
            end
            next!(progress)
        end
        m, s = round(mean(τs); digits=3), round(std(τs); digits=3)

        @info "Average meeting time: $m +/- $s"
        if saveraw_on
            wsave(fpath, @dict(τs, chains))
        else
            wsave(fpath, @dict(τs))
        end
    end
end

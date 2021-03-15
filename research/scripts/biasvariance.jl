using DrWatson
@quickactivate "Research"

using Comonicon, ProgressMeter, Statistics, CoupledHMC, VecTargets
include(scriptsdir("helper.jl"))

@main function exp_biasvariance(
    model, TS;
    n_mc_taus::Int=100, n_mc_long::Int=100, n_samples_max::Int=1_000, 
    gamma::Float64=1/20, sigma::Float64=1e-3, lambda::Float64=0.01, n_grids::Int=16,
    refreshment::String="SharedRefreshment"
)
    fname = savename(@ntuple(model, TS), "bson"; connector="-")
    fpath = projectdir("results", "biasvariance", fname)

    refreshment = parse_refreshment(refreshment)
    TS = parse_trajectory_sampler(TS)

    if isfile(fpath)
        @info "$fpath exists -- skipping."
    else
        @info "$fpath doesn't exist -- producing."

        # 1. Asymptotic variance of long-run HMC
        # 2. Get target
        # 3. Use best L and ϵ from grid search
        if model == "lr"
            # alg = CrudeHMC(0.03, 10, EndPointTS)
            # samples = get_samples(target, alg, 1_000 + 10_000; progress=true)
            # v_of(samples[1_000+1:end])
            ### NUTS: 20.93; adapted parameters: 0.02, 24
            ### Multinomial HMC: 38.09
            ### Metropolis HMC: 34.91
            # NOTE: We are conservative here by using Metropolis HMC's asymptotic variance.
            v_crude = 34.91
            target = LogisticRegression(lambda)
            if TS == MetropolisTS
                ϵ, L = 0.0125, 10
            elseif TS == CoupledMultinomialTS{QuantileCoupling}
                ϵ, L = 0.03, 10
            elseif TS == CoupledMultinomialTS{MaximalCoupling}
                ϵ, L = 0.04, 10
            elseif TS == CoupledMultinomialTS{OTCoupling}
                ϵ, L = 0.0325, 10
            elseif TS == CoupledMultinomialTS{ApproximateOTCoupling}
                ϵ, L = 0.0325, 10
            else
                error("Unkown TS type $TS.")
            end
        elseif model == "coxprocess"
            # alg = CrudeHMC(0.3, 10, EndPointTS)
            # samples = get_samples(target, alg, 1_000 + 10_000; progress=true)
            # v_of(samples[1_000+1:end])
            ### NUTS: 9052
            ### Metropolis HMC: 9620
            v_crude = 9_620
            target = LogGaussianCoxPointProcess(n_grids)
            if TS == MetropolisTS
                ϵ, L = 0.13, 10
            elseif TS == CoupledMultinomialTS{QuantileCoupling}
                ϵ, L = 0.39, 10
            elseif TS == CoupledMultinomialTS{MaximalCoupling}
                ϵ, L = 0.37, 10
            elseif TS == CoupledMultinomialTS{OTCoupling}
                ϵ, L = 0.29, 10
            elseif TS == CoupledMultinomialTS{ApproximateOTCoupling}
                ϵ, L = 0.29, 10
            else
                error("Unkown TS type $TS.")
            end
        else
            error("Unkown model name $model.")
        end
        @info "Asymptotic variance: $v_crude" 

        alg = CoupledHMCSampler(
            rinit=randn, TS=TS, ϵ=ϵ, L=L, γ=gamma, σ=sigma,
            momentum_refreshment=refreshment
        )
        τs = zeros(Int, n_mc_taus)
        progress = Progress(n_mc_taus)
        Threads.@threads for i in 1:n_mc_taus
            samples = sample_until_meeting(target, alg; n_samples_max=n_samples_max)
            τs[i] = length(samples)
            next!(progress)
        end
        
        # Getting k and m
        k_median,   m_median_5    = get_k_m(τs, 5;  method=:median)
        k_median,   m_median_10   = get_k_m(τs, 10; method=:median)
        k_quantile, m_quantile_5  = get_k_m(τs, 5;  method=:quantile)
        k_quantile, m_quantile_10 = get_k_m(τs, 10; method=:quantile)
        m_max = max(m_median_10, m_quantile_10)
        @info "Using" k_median m_median_5 m_median_10 k_quantile m_quantile_5 m_quantile_10 m_max

        # Longer runs until m_max
        chains = Vector{Any}(undef, n_mc_long)
        progress = Progress(n_mc_long)
        Threads.@threads for i in 1:n_mc_long
            samples = sample(target, alg, m_max)
            chains[i] = samples
            next!(progress)
        end

        @info "" i_of(chains, k_median,   m_median_5)...
        @info "" i_of(chains, k_median,   m_median_10)...
        @info "" i_of(chains, k_quantile, m_quantile_5)...
        @info "" i_of(chains, k_quantile, m_quantile_10)...
        wsave(fpath, @dict(τs, chains))
    end
end

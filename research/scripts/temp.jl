using DrWatson
@quickactivate "Research"

using Comonicon, ProgressMeter, Statistics, CoupledHMC, VecTargets
using CoupledHMC.AdvancedHMC


target = LogisticRegression(0.01)
# target = LogGaussianCoxPointProcess(16)

# alg = HMCSampler(rinit=randn, TS=MultinomialTS)
# rng, hamiltonian, proposal, adaptor, theta0 = CoupledHMC.get_ahmc_primitives(target, alg, nothing)
# samples, stats = AdvancedHMC.sample(
#     rng, hamiltonian, proposal, theta0, 1_000 + 10_000, adaptor, 1_000; progress=true, verbose=false
# )
# @info "" mean(map(s -> s.tree_depth^2, stats)) mean(map(s -> s.step_size, stats)) v_of(samples[1_000+1:end])

alg = HMCSampler(rinit=randn, TS=MultinomialTS, ϵ=0.02, L=24)
rng, hamiltonian, proposal, theta0 = CoupledHMC.get_ahmc_primitives(target, alg, nothing)
samples, stats = AdvancedHMC.sample(
    rng, hamiltonian, proposal, theta0, 10_000; progress=true, verbose=false
)
@info "" v_of(samples[1_000+1:end])


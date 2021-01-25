using DrWatson
@quickactivate "Research"

using Comonicon, ProgressMeter, Statistics, CoupledHMC, VecTarget
using CoupledHMC.AdvancedHMC


target = get_target(LogGaussianCoxPointProcess(datadir(), 16))
alg = HMCSampler(rinit=randn, TS=MultinomialTS)
rng, hamiltonian, proposal, adaptor, theta0 = CoupledHMC.get_ahmc_primitives(target, alg, nothing)
samples, stats = AdvancedHMC.sample(
    rng, hamiltonian, proposal, theta0, 11_000, adaptor, 1_000; progress=true, verbose=false
)
@info "" mean(map(s -> s.tree_depth^2, stats)) mean(map(s -> s.step_size, stats))


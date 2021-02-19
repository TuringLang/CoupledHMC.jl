using AdvancedHMC: AdvancedHMC, AbstractMCMCKernel, AbstractTrajectory, @unpack, step,
                   phasepoint, mh_accept_ratio, accept_phasepoint!, Transition, energy,
                   AbstractRNG, AbstractTrajectorySampler, randcat, logsumexp, rand_coupled
import AdvancedHMC: transition, sample_phasepoint

include("kernels/mixture.jl")
include("kernels/maximum_coupled_mh.jl")
include("kernels/maximum_coupled_mh_ref.jl")

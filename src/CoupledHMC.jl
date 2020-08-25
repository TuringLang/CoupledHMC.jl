__precompile__(false)
module CoupledHMC

using Random, LinearAlgebra, Distances, Distributions, OptimalTransport, AdvancedHMC

include("couplings.jl")

export IndependentCoupling, QuantileCoupling, MaximalCoupling, OTCoupling, ApproximateOTCoupling

end # module

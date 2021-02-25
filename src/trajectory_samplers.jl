import AdvancedHMC: AbstractTrajectorySampler

struct CoupledMultinomialTS{C<:AbstractCoupling} <: AbstractTrajectorySampler end

function AdvancedHMC.sample_phasepoint(
    rng,
    τ::Trajectory{CoupledMultinomialTS{C}},
    h,
    z
) where {C}
    n_steps = abs(AdvancedHMC.nsteps(τ))
    # TODO: Deal with vectorized-mode generically.
    #       Currently the direction of multiple chains are always coupled
    n_steps_fwd = AdvancedHMC.rand_coupled(rng, 0:n_steps)
    zs_fwd = AdvancedHMC.step(
        τ.integrator, h, z, n_steps_fwd;
        fwd=true, full_trajectory=Val(true)
    )
    n_steps_bwd = n_steps - n_steps_fwd
    zs_bwd = AdvancedHMC.step(
        τ.integrator, h, z, n_steps_bwd;
        fwd=false, full_trajectory=Val(true)
    )
    zs = vcat(reverse(zs_bwd)..., z, zs_fwd...)
    ℓweights = -AdvancedHMC.energy.(zs)
    if eltype(ℓweights) <: AbstractVector
        ℓweights = cat(ℓweights...; dims=2)
    end
    unnorm_ℓprob = ℓweights
    prob = exp.(unnorm_ℓprob .- AdvancedHMC.logsumexp(unnorm_ℓprob; dims=2))

    if C == QuantileCoupling || C == MaximalCoupling
        coupling = C(prob[1,:], prob[2,:])
    elseif C == OTCoupling || C == ApproximateOTCoupling
        τ¹ = cat(map(z -> z.θ[:,1], zs)...; dims=2)
        τ² = cat(map(z -> z.θ[:,2], zs)...; dims=2)
        coupling = C(prob[1,:], prob[2,:], τ¹, τ²)
    else
        error("Unknown coupling method for CoupledMultinomialTS `$C`.")
    end

    i, j = rand(rng, coupling)
    z′ = similar(z)
    foreach(enumerate([i, j])) do (i_chain, i_step)
        zi = zs[i_step]
        z′.θ[:,i_chain] = zi.θ[:,i_chain]
        z′.r[:,i_chain] = zi.r[:,i_chain]
        z′.ℓπ.value[i_chain] = zi.ℓπ.value[i_chain]
        z′.ℓπ.gradient[:,i_chain] = zi.ℓπ.gradient[:,i_chain]
        z′.ℓκ.value[i_chain] = zi.ℓκ.value[i_chain]
        z′.ℓκ.gradient[:,i_chain] = zi.ℓκ.gradient[:,i_chain]
    end

    # Computing adaptation statistics for dual averaging as done in NUTS
    Hs = -ℓweights
    ΔH = Hs .- AdvancedHMC.energy(z)
    α = exp.(min.(0, -ΔH))  # this is a matrix for vectorized mode and a vector otherwise
    α = typeof(α) <: AbstractVector ? mean(α) : vec(mean(α; dims=2))
    return z′, true, α
end

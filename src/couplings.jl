abstract type AbstractCoupling end

Random.rand(c::AbstractCoupling) = rand_pair(c)

randseed(rng=Random.GLOBAL_RNG) = rand(rng, 1:1_000_000)

"Independent coupling"
struct IndependentCoupling{T<:AbstractVector} <: AbstractCoupling 
    p::T
    q::T
end

rand_pair(ic::IndependentCoupling) = (i = rand(Categorical(ic.p)), j = rand(Categorical(ic.q)))

"Quantile coupling"
struct QuantileCoupling{T<:AbstractVector} <: AbstractCoupling 
    p::T
    q::T
    rngs
end

function QuantileCoupling(p, q)
    seed = randseed()
    rngs = [MersenneTwister(seed) for _ in 1:2]
    return QuantileCoupling(p, q, rngs)
end

function rand_pair(qc::QuantileCoupling)
    return (i = rand(qc.rngs[1], Categorical(qc.p)), j = rand(qc.rngs[2], Categorical(qc.q)))
end

"Maximal coupling"
struct MaximalCoupling{T<:AbstractVector} <: AbstractCoupling 
    p::T
    q::T
end

function rand_pair(mc::MaximalCoupling)
    ω = 1 - totalvariation(mc.p, mc.q)
    pqmin = min.(mc.p, mc.q)
    Z = sum(pqmin)
    u = rand()
    if u < ω
        i = j = rand(Categorical(pqmin / Z))
    else
        i = rand(Categorical((mc.p - pqmin) / (1 - Z)))
        j = rand(Categorical((mc.q - pqmin) / (1 - Z)))
    end
    return (i = i, j = j)
end

"OT coupling"
struct OTCoupling{T1<:AbstractVector, T2<:AbstractMatrix} <: AbstractCoupling
    p::T1
    q::T1
    D::T2
end

OTCoupling(p, q, τ₁, τ₂) = OTCoupling(p, q, euclidsq(τ₁, τ₂))

function euclidsq(X::T, Y::T) where {T<:AbstractMatrix}
    XiXj = transpose(X) * Y
    x² = sum(X .^ 2; dims=1)
    y² = sum(Y .^ 2; dims=1)
    return transpose(x²) .+ y² - 2XiXj
end

euclidsq(x::T, y::T) where {T<:AbstractVector} = 
    euclidsq(reshape(x, 1, length(x)), reshape(y, 1, length(y)))

function rand_joint(J::AbstractMatrix)
    u = collect(Iterators.product(1:size(J, 1), 1:size(J, 2)))
    v = vec(J)
    return u[rand(Categorical(v))]
end

"Covert `Ajoint` to `TM<:AbstractMatrix`."
rand_joint(J::Adjoint{TN, TM}) where {TN, TM<:AbstractMatrix} = rand_joint(TM(J))

function rand_pair(otc::OTCoupling)
    γ = emd(otc.p, otc.q, otc.D)
    i, j = rand_joint(γ)
    return (i = i, j = j)
end

"Approximate OT coupling"
struct ApproximateOTCoupling{
    T1<:AbstractVector, T2<:AbstractMatrix, T3<:AbstractFloat
} <: AbstractCoupling
    p::T1
    q::T1
    D::T2
    ϵ::T3
end

function ApproximateOTCoupling(
    p::T1, q::T1, τ₁::T2, τ₂::T2; reps::T3=0.05
) where {T1<:AbstractVector, T2<:AbstractVecOrMat, T3<:AbstractFloat}
    D = euclidsq(τ₁, τ₂)
    ϵ = reps * mean(D)
    return ApproximateOTCoupling(p, q, D, ϵ)
end

function rand_pair(aotc::ApproximateOTCoupling)
    γ = sinkhorn(aotc.p, aotc.q, aotc.D, aotc.ϵ)
    p_γ, q_γ = vec(sum(γ; dims=2)), vec(sum(γ; dims=1))
    α = min(minimum(aotc.q ./ q_γ), minimum(aotc.p ./ p_γ))
    u = rand()
    if u < α
        i, j = rand_joint(γ)
    else
        p_debias = (aotc.p - (1 - α) * p_γ) / α
        q_debias = (aotc.q - (1 - α) * q_γ) / α
        i = rand(Categorical(p_debias))
        j = rand(Categorical(q_debias))
    end
    return (i = i, j = j)
end

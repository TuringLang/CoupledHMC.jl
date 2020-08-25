### IterTools extensions

using IterTools
using IterTools: @ifsomething

struct TakeUntil{I}
    cond::Function
    xs::I
end

"""
    takeuntil(cond, xs)
An iterator that yields values from the iterator `xs` as long as the
predicate `cond` is true. Unlike `takewhile`, it also take the last 
value for which the predicate `cond` is false.
```jldoctest
julia> collect(takeuntil(x-> x^2 < 10, 1:100))
3-element Array{Int64,1}:
 1
 2
 3
 4
```
"""
takeuntil(cond, xs) = TakeUntil(cond, xs)

function Base.iterate(it::TakeUntil, state=(false, nothing))
    is_cond, state_xs = state
    is_cond && return nothing
    (val, state_xs) = 
        @ifsomething (state_xs === nothing ? iterate(it.xs) : iterate(it.xs, state_xs))
    val, (it.cond(val), state_xs)
end

Base.IteratorSize(it::TakeUntil) = Base.SizeUnknown()
Base.eltype(::Type{TakeUntil{I}}) where {I} = eltype(I)
IteratorEltype(::Type{TakeUntil{I}}) where {I} = IteratorEltype(I)

### Random extensions

using Random
using Random: GLOBAL_RNG

Random.MersenneTwister(seeds::AbstractVector{Int}) = MersenneTwister.(seeds)

"Sample a random seed to be used"
randseed(rng) = rand(rng, Int16) + 2^16
randseed() = randseed(GLOBAL_RNG)

"""
    rands(rng, dim::Int; R=1)

Sample a `dim`-dimensional vector from U(`-R`, `R`).
"""
rands(rng, dim::Int; R=1) = R * (2 * rand(rng, dim) .- 1)
rands(args...; kwargs...) = rands(GLOBAL_RNG, args...; kwargs...)

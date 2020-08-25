const ATOL_MEETING = 1e-10

"The first `n` moments for `x` concatnated in a vector."
moments(x::AbstractVector; n::Int=2) = cat((i == 1 ? x : x.^i for i in 1:n)...; dims=1)

"""
    k, m = get_k_m(τs, λ; method=:quantile)
Heuristically picking `k` amd `m` from a sample set of 
meeting time (`τs`) following Heng and Jacob (2019).
Firstly, `k` is picked depending on the value of `method`
- `:quantile`: the 90% quantile of `τs`
- `:median`: the median of `τs`
After that, `m` is the to a `λ * k`.
Typically values for the multiplier `λ` are 5 and 10.
"""
function get_k_m(τs, λ; method=:quantile)
    local k
    if method == :median
        k = median(τs)
    elseif method == :quantile
        for τ in sort(τs)
            if sum(τ .> τs) / length(τs) >= 0.9
                k = τ
            end
        end
    end
    m = λ * k
    return floor(Int, k), floor(Int, m)
end

function get_xs_ys(coupled_samples)
    xs = map(s -> s[:,1], coupled_samples)  # xs stores X_1, ..., X_{n+1}
    ys = map(s -> s[:,2], coupled_samples)  # ys stores Y_0, ..., Y_n
    return xs, ys
end

does_meet(x, y) = isapprox(x, y; atol=ATOL_MEETING)
does_meet(xy) = does_meet(xy[:,1], xy[:,2])

function τ_of(samples)
    xs, ys = get_xs_ys(samples)
    # Compute the (relaxed) meeting time
    τ = findfirst(==(true), does_meet.(xs, ys))
    return isnothing(τ) ? length(samples) + 1 : τ
end

M_of(h, xs, k, m) = k < m ? mean(h, xs[k+1:m]) : zeros(length(h(xs[1])))

"""
Coupled chains
|-----+-----+-----+--->
1     k     τ     m
"""
function H_of(h, samples, k, τ, m)
    xs, ys = get_xs_ys(samples)
    M = M_of(h, xs, k, m)
    if τ - 1 >= k + 1   # avoid computing the bias correction term if meet too early
        H = M + mapreduce(+, k+1:τ-1) do n
            min(1, (n - k) / (m - k + 1)) * (h(xs[n]) - h(ys[n]))
        end
    else
        H = M
    end
    return H
end

function i_of(h::Function, chains, k::Int, m::Int)
    τs = τ_of.(chains)
    Hs = map(zip(τs, chains)) do (τ, samples)
        H_of(h, samples, k, τ, m)
    end
    cost = mean(2 * (τs .- 1) + max.(1, m + 1 .- τs))
    v = sum(var(Hs))
    return (ineff=cost * v, cost=cost, v=v)
end
i_of(chains, k::Int, m::Int) = i_of(moments, chains, k, m)

function v_of(h::Function, samples)
    h_rm = hcat(map(moments, samples)...)'  # row major h
    return sum(rcopy(R"spectrum0.ar($h_rm)")[:spec]) 
end
v_of(samples) = v_of(moments, samples)

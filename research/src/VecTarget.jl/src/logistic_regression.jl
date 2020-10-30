struct LogisticRegression{TX, Ty, T}
    X::TX
    y::Ty
    lambda::T
end

function LogisticRegression(datadir::String, lambda)
    @unpack design_matrix, response = BSON.load(joinpath(datadir, "germancredit.bson"))
    return LogisticRegression(design_matrix', response, lambda)
end

function get_target(lr::LogisticRegression)
    function _logdensity(theta)
        n_chains = size(theta, 2)
        logv = theta[1,:]
        v = exp.(logv)      # n_chains
        a = theta[2,:]      # n_chains
        b = theta[3:end,:]  # 300 x n_chains
        logitp = a' .+ lr.X' * b
        p = logistic.(logitp)
        p = p * (1 - 2 * eps()) .+ eps()    # numerical stability

        logabsdetjacob = logv
        logprior = logpdf.(Ref(Exponential(1 / lr.lambda)), v) + logabsdetjacob
        s = sqrt.(v)
        T = eltype(s)
        logprior += logpdf(BroadcastedNormalStd(zeros(T, 1), s), a)
        logprior_b = logpdf(BroadcastedNormalStd(zeros(T, 1, 1), s'), b)
        logprior += dropdims(sum(logprior_b; dims=1); dims=1)
        loglike_elementwise = logpdf.(Bernoulli.(p), lr.y)
        loglike = dropdims(sum(loglike_elementwise; dims=1); dims=1)
        return logprior + loglike
    end
    
    function logdensity(theta::AbstractVector)
        theta = reshape(theta, length(theta), 1)
        lp = _logdensity(theta)
        return only(lp)
    end

    logdensity(theta::AbstractMatrix) = _logdensity(theta)

    return (
        dim = size(lr.X, 1) + 2, 
        logdensity = logdensity, 
        get_grad = x -> get_∂ℓπ∂θ_reversediff(logdensity, x),
    )
end

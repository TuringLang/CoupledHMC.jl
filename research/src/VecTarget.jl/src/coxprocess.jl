struct LogGaussianCoxPointProcess{T1, T2}
    counts::T1
    ngrid::Int
    dimension::Int
    sigmasq::T2
    mu::T2
    beta::T2
    area::T2
end

# Ref: https://github.com/pierrejacob/debiasedhmc/blob/master/inst/coxprocess/model.R#L6-L22
function LogGaussianCoxPointProcess(ngrid)
    data_counts =
    rcopy(R"""
    library(spatstat)

    # load pine saplings dataset
    data(finpines)
    data_x <- (finpines$x + 5) / 10 # normalize data to unit square
    data_y <- (finpines$y + 8) / 10
    #plot(x = data_x, y = data_y, type = "p")

    ngrid <- $ngrid
    grid <- seq(from = 0, to = 1, length.out = ngrid+1)
    dimension <- ngrid^2
    data_counts <- rep(0, dimension)
    for (i in 1:ngrid){
      for (j in 1:ngrid){
        logical_y <- (data_x > grid[i]) * (data_x < grid[i+1])
        logical_x <- (data_y > grid[j]) * (data_y < grid[j+1])
        data_counts[(i-1)*ngrid + j] <- sum(logical_y * logical_x)
      }
    }

    data_counts
    """)
    sigmasq = 1.91
    mu = log(126) - 0.5 * sigmasq
    beta = 1 / 33
    dimension = ngrid^2
    area = 1 / dimension

    return LogGaussianCoxPointProcess(Int.(data_counts), ngrid, dimension, sigmasq, mu, beta, area)
end

function get_target(lgcpp::LogGaussianCoxPointProcess)
    @unpack mu, dimension, ngrid, sigmasq, beta, counts, area = lgcpp
    μ = fill(mu, dimension)
    Σ = Matrix{Float64}(undef, dimension, dimension)
    for m in 1:dimension, n in 1:dimension
        i = [floor(Int, (m - 1) / ngrid) + 1, (m - 1) % ngrid + 1]
        j = [floor(Int, (n - 1) / ngrid) + 1, (n - 1) % ngrid + 1]
        Σ[m,n] = sigmasq * exp(-sqrt(sum((i - j).^2)) / (ngrid * beta))
    end
    prior_X = MvNormal(μ, Σ)
    
    function loglikelihood(x::AbstractVector)
        cumsum = 0
        for i in 1:length(x)
            cumsum += x[i] * counts[i] - area * exp(x[i])
        end
        return cumsum
    end

    function loglikelihood(x::AbstractMatrix)
        n_chains = size(x, 2)
        return map(n -> loglikelihood(x[:,n]), 1:n_chains)
    end
    
    _logdensity(x) = logpdf(prior_X, x) + loglikelihood(x)
    
    function logdensity(x::AbstractVector)
        theta = reshape(x, length(x), 1)
        lp = _logdensity(x)
        return only(lp)
    end

    logdensity(x::AbstractMatrix) = _logdensity(x)

    return (
        dim = lgcpp.dimension, 
        logdensity = logdensity, 
        get_grad = x -> get_∂ℓπ∂θ_reversediff(logdensity, x),
    )
end

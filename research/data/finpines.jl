using DrWatson

@quickactivate "Research"

using RCall

for ngrid in [16, 32, 64]

    data_counts = rcopy(R"""
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

    data_counts = Int.(data_counts)
    sigmasq = 1.91
    mu = log(126) - 0.5 * sigmasq
    beta = 1 / 33
    dimension = ngrid^2
    area = 1 / dimension

    wsave(datadir("finpines-$ngrid.bson"), @dict(data_counts, ngrid, dimension, sigmasq, mu, beta, area))

end
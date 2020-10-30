using DrWatson

@quickactivate "Research"

using RCall

germancredit_path = datadir("germancredit.txt")
    
# Ref: https://github.com/pierrejacob/debiasedhmc/blob/master/inst/logistic/model.R
design_matrix, response = 
rcopy(R"""
germancredit <- read.table($germancredit_path)

design_matrix <- scale(germancredit[, 1:24])
response <- germancredit[, 25] - 1
nsamples <- nrow(design_matrix)
dimension <- ncol(design_matrix)
interaction_terms <- matrix(nrow = nsamples, ncol = dimension*(dimension-1) / 2)
index <- 1
for (j in 1:(dimension-1)){
    for (jprime in (j+1):dimension){
    interaction_terms[, index] <- design_matrix[, j] * design_matrix[, jprime]
    index <- index + 1
    }
}
design_matrix <- cbind(design_matrix, scale(interaction_terms))

list(design_matrix, response)
""")

wsave(datadir("germancredit.bson"), @dict(design_matrix, response))

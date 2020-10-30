using DrWatson
@quickactivate "Research"

using InteractiveUtils
versioninfo()

model_list = ["lr", "coxprocess"]

TS_list = [
    "MetropolisTS",
    "CoupledMultinomialTS{QuantileCoupling}",
    "CoupledMultinomialTS{MaximalCoupling}",
    "CoupledMultinomialTS{OTCoupling}",
    "CoupledMultinomialTS{ApproximateOTCoupling}",
]

slavepath = scriptsdir("biasvariance.jl")
for model in model_list, TS in TS_list
    cmd = `julia $slavepath $model $TS`
    @info "Running" cmd
    run(cmd)
end

wsave(scriptsdir("biasvariance-sweep.bson"), @dict(model_list, TS_list))

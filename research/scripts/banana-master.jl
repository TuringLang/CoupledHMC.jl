using DrWatson
@quickactivate "Research"

using InteractiveUtils
versioninfo()

refreshment_list = ["shared", "contractive"]

TS_list = [
    "MetropolisTS",
    "CoupledMultinomialTS{QuantileCoupling}",
    "CoupledMultinomialTS{MaximalCoupling}",
    "CoupledMultinomialTS{OTCoupling}",
    "CoupledMultinomialTS{ApproximateOTCoupling}",
]

slavepath = scriptsdir("banana.jl")
for refreshment in refreshment_list, TS in TS_list
    cmd = `julia $slavepath $refreshment $TS`
    @info "Running" cmd
    run(cmd)
end

wsave(scriptsdir("banana-sweep.bson"), @dict(refreshment_list, TS_list))

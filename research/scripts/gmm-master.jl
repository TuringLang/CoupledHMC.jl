using DrWatson
@quickactivate "Research"

using InteractiveUtils
versioninfo()

refreshments = ["SharedRefreshment"]

TS_list = [
    "MetropolisTS",
    "CoupledMultinomialTS{QuantileCoupling}",
    "CoupledMultinomialTS{MaximalCoupling}",
    "CoupledMultinomialTS{OTCoupling}",
    "CoupledMultinomialTS{ApproximateOTCoupling}",
]

epsilon_inc_list = [(_epsilon, 10) for _epsilon in collect(0.1:0.05:0.3)]

L_inc_list = [(0.1, _L) for _L in collect(10:5:30)]

slavepath = scriptsdir("gmm.jl")
for refreshment in refreshments, TS in TS_list, (epsilon, L) in unique([epsilon_inc_list..., L_inc_list...])
    cmd = `julia $slavepath $refreshment $TS $epsilon $L`
    @info "Running" cmd
    run(cmd)
end

wsave(scriptsdir("gmm-sweep.bson"), @dict(TS_list, epsilon_inc_list, L_inc_list))

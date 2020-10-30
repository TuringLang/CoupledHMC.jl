using DrWatson
@quickactivate "Research"

using InteractiveUtils
versioninfo()

model_list = ["gaussian", "lr", "coxprocess"]

TS_list = [
    "MetropolisTS",
    "CoupledMultinomialTS{QuantileCoupling}",
    "CoupledMultinomialTS{MaximalCoupling}",
    "CoupledMultinomialTS{OTCoupling}",
    "CoupledMultinomialTS{ApproximateOTCoupling}",
]

epsilon_list_dict = Dict(
    "gaussian" => collect(0.05:0.02:0.45),
    "lr" => collect(0.01:0.0025:0.04),
    "coxprocess" => collect(0.05:0.02:0.45),
)

L_list_dict = Dict(
    "gaussian" => collect(5:5:15),
    "lr" => collect(10:10:30),
    "coxprocess" => collect(10:10:30),
)

slavepath = scriptsdir("meeting.jl")
for model in model_list, TS in TS_list
    epsilon_list = epsilon_list_dict[model]
    L_list = L_list_dict[model]
    # Double length for coupled multinomial HMC
    # if TS != "MetropolisTS"
    #     d = L_list[end] - L_list[end-1]
    #     L_list = [L_list..., (L_list[end]+d:d:2L_list[end])...]
    # end

    for epsilon in epsilon_list, L in L_list
        cmd = `julia $slavepath $model $TS $epsilon $L`
        @info "Running" cmd
        run(cmd)
    end
end

wsave(scriptsdir("meeting-sweep.bson"), @dict(model_list, TS_list, epsilon_list_dict, L_list_dict))

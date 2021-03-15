function parse_refreshment(refreshment)
    # parse refreshment
    e = Meta.parse(refreshment)
    return if e isa Symbol
        # Is not instantiated => instantiate
        Base.eval(CoupledHMC, Expr(:call, e))
    else
        # Is instantiated => do nothing
        Base.eval(CoupledHMC, e)
    end
end

parse_trajectory_sampler(TS) = Base.eval(CoupledHMC, Meta.parse(TS))

function parse_termination_criterion(criterion)
end

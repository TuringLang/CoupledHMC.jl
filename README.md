# CoupledHMC.jl

This package implemented the coupled HMC kernels used in the AISTATS 2021 paper 
*Couplings for Multinomial Hamiltonian Monte Carlo* by Kai Xu, Tor Erlend Fjelde, Charles Sutton and Hong Ge.

See the `research` folder for scripts to reproduce the results.

You can cite this paper using the BibTeX entry below.

```
@inproceedings{xu2021couplings,
  title={{Couplings for Multinomial {H}amiltonian Monte Carlo}},
  author={Kai Xu and Fjelde, Tor Erlend and Sutton, Charles and Ge, Hong},
  booktitle={International Conference on Artificial Intelligence and Statistics},
  year={2021}
}
```

## Using this package directly

If you would like to use `CoupledHMC` without going through our `research` folder,
you will have to install `VecTarget`(https://github.com/xukai92/VecTargets.jl) and the master branch of `AdvancedHMC` manually.
For example, running the command below will install them globally

``` sh
julia -e 'using Pkg; Pkg.add(url="https://github.com/xukai92/VecTargets.jl"); Pkg.add(name="AdvancedHMC", rev="master")'
```

Or below to install inside your local environment

``` sh
julia --project=@. -e 'using Pkg; Pkg.add(url="https://github.com/xukai92/VecTargets.jl"); Pkg.add(name="AdvancedHMC", rev="master")'
```


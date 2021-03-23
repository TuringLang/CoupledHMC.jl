This folder contains the source code for producing the results in the AISTATS 2021 paper 
*Couplings for Multinomial Hamiltonian Monte Carlo* by Kai Xu, Tor Erlend Fjelde, Charles Sutton and Hong Ge.

You can cite this paper using the BibTeX entry below.

```
@inproceedings{xu2021couplings,
  title={{Couplings for Multinomial {H}amiltonian Monte Carlo}},
  author={Kai Xu and Fjelde, Tor Erlend and Sutton, Charles and Ge, Hong},
  booktitle={International Conference on Artificial Intelligence and Statistics},
  year={2021}
}
```

The code for coupled HMC kernels are based on [AdvancedHMC.jl](https://github.com/TuringLang/AdvancedHMC.jl).

# Running the experiment scripts

You will need to have [Julia](https://julialang.org/) installed first.
After this, make sure [DrWatson.jl](https://github.com/JuliaDynamics/DrWatson.jl) is installed globally.
This can be done by doing `] add DrWatson` in the Julia REPL.

## How to set up the dependencies

Before the first time of running scripts, you will have to
1. Start the REPL from this folder
2. Activate the environment by `] activate .`
3. do `] instantiate` to instantiate all the dependencies

This is the only time for which you need to activate the environment (manually).
DrWatson.jl will save your effort of activating environments in the future.

## Reproducing results from our paper

The folder `scripts` contains all of our experiment scripts.
Files suffixed with `master` are those for running a set of experiments while the corresponding one without `master` are for a single run with specific parameters.

You can run scripts in the `scripts` folder to reproduce our results.
For example to reproduce the GMM experiment in our appendix
``` sh
julia scripts/gmm-master.jl
```

## Multi-threading

You may find using multi-threading helpful to speed up the running.
You can do that by setting up the environment variable `JULIA_NUM_THREADS` for this purpose.
For example to use 10 threads

``` sh
JULIA_NUM_THREADS=10 julia scripts/gmm-master.jl
```

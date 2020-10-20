# Bayesian Pseudocoresets Experiments

This repository implements the experiments from

D. Manousakas, Z. Xu, C. Mascolo, T. Campbell. "Bayesian Pseudocoresets," Advances in Neural Information Processing Systems, 2020.

- Clone and install the [Bayesian coresets library repository](https://www.github.com/trevorcampbell/bayesian-coresets)
- In each `example_*` folder in this repo, run the `run.sh` script to perform the experiment
- In each `example_*` folder in this repo, run the `plot_*.sh` script to plot the results

Note: Adapt the `num_processes` parameters inside the `main.py` file of each experiment  according to your computational resources. This is used to parallelize the (DP-)PSVI experiment across the considered coreset sizes.

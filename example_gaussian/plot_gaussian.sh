#!/bin/bash

n_trials=10
plot_every=5
fldr_plts='figs'
fldr_res='results'
python3 plot_kl.py $n_trials $plot_every $fldr_plts $fldr_res

#!/bin/bash

fldr_figs='figs'
fldr_res='results'
for dnm in 'santa100K' 'ds1.100' 'fma'
do
  python3 plot_kl.py $dnm $fldr_figs $fldr_res
done

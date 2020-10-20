#!/bin/bash

for ID in {1..10}
do
    for alg in "GIGAO" "GIGAR" "RAND" "PRIOR" "SVI" "BPSVI"
    do
	python3 main.py $alg $ID
    done
done

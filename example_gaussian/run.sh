#!/bin/bash

for ID in {1..10}
do
    for alg in "BPSVI" "SVI" "GIGAO" "GIGAR" "RAND"
    do
	    python3 main.py $alg $ID
    done
done

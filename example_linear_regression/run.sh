#!/bin/bash

for ID in {1..5}
do
    for alg in "SVI" "BPSVI" "GIGAO" "GIGAR" "RAND" "PRIOR"
    do
	python3 main.py $alg $ID
    done
done

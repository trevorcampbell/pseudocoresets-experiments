#!/bin/bash
for ID in {1..10}
do
  # i. Transactions Dataset
	dnm="santa100K"
	stan_samples="True"
	samplediag="True"
	graddiag="False"
	for alg in "RAND" "PRIOR" "GIGAO" "GIGAR"
        do
	  python3 main.py $alg $dnm $ID $stan_samples $samplediag $graddiag 
        done
	alg="SVI"
	i0="0.1"
	python3 main.py $alg $dnm $ID $stan_samples $samplediag $graddiag $i0
	for alg in  "DPBPSVI" "BPSVI"
        do
	        i0="1."
	 	python3 main.py $alg $dnm $ID $stan_samples $samplediag $graddiag $i0
	done

	# ii. ChemReact100 Dataset
	dnm="ds1.100"
	stan_samples="False"
	samplediag="False"
	graddiag="True"
        for alg in "RAND" "PRIOR" "GIGAO" "GIGAR"
        do
	  python3 main.py $alg $dnm $ID $stan_samples $samplediag $graddiag 
        done
	alg="SVI"
	i0="0.1"
	python3 main.py $alg $dnm $ID $stan_samples $samplediag $graddiag $i0
	for alg in "BPSVI" "DPBPSVI"
	do
	  i0="10.0"
          python3 main.py $alg $dnm $ID $stan_samples $samplediag $graddiag $i0
	done

	# iii. Music Dataset
	dnm="fma"
	stan_samples="True"
	samplediag="True"
	graddiag="True"
        for alg in "RAND" "PRIOR" "GIGAO" "GIGAR"
        do
          python3 main.py $alg $dnm $ID $stan_samples $samplediag $graddiag
        done
	alg="SVI"
	i0="1.0"
	python3 main.py $alg $dnm $ID $stan_samples $samplediag $graddiag $i0
	for alg in "BPSVI" "DPBPSVI"
	do
	  i0="10.0"
	  python3 main.py $alg $dnm $ID $stan_samples $samplediag $graddiag $i0
	done
done

#!/bin/bash

# exmpale
# mpirun -np 9 python main.py -v 1 -e atari -g Alien -k 10 -r v1_k10_run0
# bash run.sh game version runtimes run_name_prefix

runtimes=$3
version=$2
game=$1
k=10
run_name=$4

i=0

while [ $i -lt $runtimes ]
do
    mpirun -np 9 python NCSCC.py -v $version -e atari -g $game -k $k -r ${run_name}-v${version}-k${k}-run${i}
    i=$((i+1))
done

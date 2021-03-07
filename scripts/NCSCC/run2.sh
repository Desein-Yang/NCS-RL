#!/bin/bash
# 调参实验
# 调整k

# exmpale
# mpirun -np 9 python main.py -v 1 -e atari -g Alien -k 10 -r v1_k10_run0
# bash run.sh k i run_name_prefix

k=$1
i=$2
run_name=$3
N=9

version=4
games=(Alien Qbert Freeway SpaceInvaders)


for game in ${games[@]}
do
    mpirun -np $N python main.py -v $version -e atari -g $game -k $k -r tiaocan-${run_name}-v${version}-k${k}-N${N}-run${i}
    # echo "mpirun -np $N python main.py -v $version -e atari -g $game -k $k -r tiaocan-${run_name}-v${version}-k${k}-N${N}-run${i}"
done
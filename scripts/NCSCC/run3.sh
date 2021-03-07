#!/bin/bash
# 调参实验
# 调整 N和epoch

# exmpale
# mpirun -np 9 python main.py -v 1 -e atari -g Alien -k 10 -r v1_k10_run0
# bash run.sh k i run_name_prefix

k=5
i=$1
# run_name=$2
# Ns=(7 9 11)
Ns=(7)
epochs=(5 10)

version=4
games=(Alien Qbert Freeway SpaceInvaders)


for game in ${games[@]}
do
    for N in ${Ns[@]}
    do
        for epoch in ${epochs[@]}
        do
            mpirun -np $N python main.py -v $version -e atari -g $game -k $k --epoch ${epoch} -r tiaocan-epoch${epoch}-v${version}-k${k}-N${N}-run${i}
            # echo "mpirun -np $N python main.py -v $version -e atari -g $game -k $k --epoch ${epoch} -r tiaocan-epoch${epoch}-v${version}-k${k}-N${N}-run${i}"
        done
    done
done
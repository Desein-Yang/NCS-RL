#!/bin/bash
# 补齐实验 兼 参数k调整

k=$1
nums=$2
part=$3
# run_name=$2
# Ns=(7 9 11)
N=7
epoch=5

version=4
# games=(Alien BeamRider Breakout Enduro Freeway MontezumaRevenge Pitfall Pong Qbert Seaquest SpaceInvaders Venture)
games=(Alien BeamRider Breakout Enduro Pong Seaquest Venture)


count=0
for game in ${games[@]}
do
    i=0
    while [ $i -lt 3 ]
    do
        if [ $((count % nums)) == $part ]
        then
        mpirun -np $N python main.py -v $version -e atari -g $game -k $k --epoch ${epoch} -r runall-epoch${epoch}-v${version}-k${k}-N${N}-run${i}
        # echo "mpirun -np $N python main.py -v $version -e atari -g $game -k $k --epoch ${epoch} -r runall-epoch${epoch}-v${version}-k${k}-N${N}-run${i}"
        fi
        i=$((i+1))
        count=$((count+1))
    done
done
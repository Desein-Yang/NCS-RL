#!/bin/bash
# 补齐实验 兼 参数k调整

k=7
nums=$1
part=$2
# run_name=$2
# Ns=(7 9 11)
N=7
epoch=5

version=4
# games=(Alien BeamRider Breakout Enduro Freeway MontezumaRevenge Pitfall Pong Qbert Seaquest SpaceInvaders Venture)
# games=(Alien BeamRider Breakout Enduro Pong Seaquest Venture)
games=(Alien Freeway Qbert)
r=0.8
sigma0s=(0.2 1.0)

count=0

for sigma0 in ${sigma0s[@]}
do
    for game in ${games[@]}
    do
        i=0
        while [ $i -lt 3 ]
        do
            if [ $((count % nums)) == $part ]
            then
            CUDA_VISIBLE_DEVICES="" mpirun -np $N python main.py -v $version -e atari -g $game -k $k --epoch ${epoch} --sigma0 $sigma0 --rvalue $r -r tiaocan2--sigma0${sigma0}-r${r}-epoch${epoch}-v${version}-k${k}-N${N}-run${i}
            # echo "mpirun -np $N python main.py -v $version -e atari -g $game -k $k --epoch ${epoch} --sigma0 $sigma0 --rvalue $r -r tiaocan2--sigma0${sigma0}-r${r}-epoch${epoch}-v${version}-k${k}-N${N}-run${i}"
            fi
            i=$((i+1))
            count=$((count+1))
        done
    done
done
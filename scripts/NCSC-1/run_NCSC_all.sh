#!/bin/bash
# run_name=$2
# Ns=(7 9 11)
#games=(Atlantis Bowling Freeway Frostbite Asteroids Venture Alien Pong Kangaroo Hero)
games=(Atlantis)
i=1
for game in ${games[@]}
do
#while [ $i -lt 3 ]
#do
{
    echo "start run ${game}"
    mpirun -np 7 python ./NCS-C-base.py -e 'atari' -g ${game}  -r NCS${game}${i} -c './config/NCSC-1/atari-opt.json' > ./logs/NCSC-1/${game}-NCSC1-100M-${i}.log
}&
#count=$((count+1))
#done
done

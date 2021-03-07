#!/bin/bash
# run_name=$2
# Ns=(7 9 11)
#games=(Atlantis Bowling Freeway Frostbite Asteroids Venture Alien Pong Kangaroo Hero)
#games=(Atlantis Bowling Freeway Frostbite Asteroids)
games=(Atlantis Freeway Hero)
i=1
for game in ${games[@]}
do
#while [ $i -lt 3 ]
#do
{
    echo "start run ${game}"
    mpirun -np 25 python  ./NCSRE.py -e 'atari' -g ${game}  -r 0917${game}${i} -c './config/NCSRE/atari-opt.json' > ./logs/${game}-NCSRE-100M-${i}.log
}&
#count=$((count+1))
#done
done

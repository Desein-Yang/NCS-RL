#!/bin/bash
# run_name=$2
# Ns=(7 9 11)
#games=(Atlantis Bowling Freeway Frostbite Asteroids Venture Alien Pong Kangaroo Hero)
games=(Freeway Bowling Frostbite)
#games=(Alien)
i=3
for game in ${games[@]}
do
#while [ $i -lt 3 ]
#do
{
    echo "start run ${game}"
    mpirun -np 15 python  ./NCSRE.py -e 'atari' -g ${game}  -r dell-all-${game}${i} -c './config/NCSRE/atari-opt.json' > ./logs/0924-${game}-NCSRE-100M-${i}.log
}&
#count=$((count+1))
#done
done

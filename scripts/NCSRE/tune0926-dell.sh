#!/bin/bash
# run_name=$2
# Ns=(7 9 11)
#games=(Atlantis Bowling Freeway Frostbite Asteroids Venture Alien Pong Kangaroo Hero)
#games=(Freeway Bowling Frostbite)
#games=(Alien)
games=(Freeway)
i=5
for game in ${games[@]}
do
#while [ $i -lt 3 ]
#do
mpirun -np 21 python  ./NCSRE.py -e 'atari' -g ${game}  -r tune0-${game}${i} -c './config/NCSRE/tune0-atari-opt.json' > ./logs/0926-${game}-NCSRE-100M-${i}.log & mpirun -np 15 python  ./NCSRE.py -e 'atari' -g ${game}  -r tune1-${game}${i} -c './config/NCSRE/tune1-atari-opt.json' > ./logs/0926-${game}-NCSRE-100M-${i}.log & mpirun -np 11 python  ./NCSRE.py -e 'atari' -g ${game}  -r tune2-${game}${i} -c './config/NCSRE/tune2-atari-opt.json' > ./logs/0926-${game}-NCSRE-100M-${i}.log

#count=$((count+1))
#done
done

# This is config version
game=()
for i in 1
do
    # rl test
    # mpirun -np 9 python -m memory_profiler ./NCSRE.py -e 'atari' -g ${game}  -r 0901-${game}${i} -c './config/NCSRE/atari-opt.json' > ./logs/${game}-NCSRE-100M-${i}.log
    mpirun -np 9 python  ./NCSRE.py -e 'atari' -g ${game}  -r 0912-${game}${i} -c './config/NCSRE/atari-opt.json' > ./logs/${game}-NCSRE-100M-${i}.log

    # function test
    #f=2
    #d=100
    #mpirun -np 17 python ./NCSRE.py -e 'function' -f ${f} -d ${d} -r NCSRE-d${d}-2 -c './config/NCSRE/func-opt.json'> ./logs/f${f}-NCSRE-1M-${i}.log
done

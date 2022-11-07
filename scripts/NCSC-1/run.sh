# This is config version
game=Atlantis
for i in 1
do
    # rl test
     mpirun -np 7 python ./NCS-C-base.py -e 'atari' -g ${game}  -r NCS${game}${i} -c './config/NCSC-1/atari-opt.json' > ./logs/${game}-NCS-100M-${i}.log

    # function test
    #f=2
    #d=1000
    #mpirun -np 7 python ./NCS-C-base.py -e 'function' -f ${f} -d ${d} -r NCSd${d} -c './config/NCS/func-opt.json'> ./logs/f${f}-NCS-1M-${i}.log
done

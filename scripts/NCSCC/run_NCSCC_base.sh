# This is config version
game=Atlantis
for i in 1
do
    # rl test
    # mpirun -np 7 python ./NCSCC-base.py -e 'atari' -g ${game}  -r NCSCC${game}${i} -c './config/NCSC-1/atari-opt.json' > ./logs/${game}-NCSCC-100M-${i}.log

    # function test
    f=2
    d=1000
    mpirun -np 7 python ./NCSCC-base.py -e 'function' -f ${f} -d ${d} -r NCSCCd${d} -c './config/NCSCC/func-opt.json'> ./logs/f${f}-NCSCC-1M-${i}.log
done

# This is config version
game=Bowling
for i in 1
do
    # rl test
    mpirun -np 9 python NCSRE.py -r debug1 -g Alien --lam 8 --mu 1 

    # function test
    #f=2
    #d=100
    mpirun -np 17 python ./NCSRE.py -e 'function' -f ${f} -d ${d} -r NCSRE-d${d}-2 -c './config/NCSRE/func-opt.json'> ./logs/f${f}-NCSRE-1M-${i}.log
done

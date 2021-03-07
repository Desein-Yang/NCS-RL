game=Atlantis
sigma0=0.2
r=0.8
k=7
epoch=10
for i in 1
do
    # rl test
    mpirun -np 7 python ./NCS-C-base.py -g ${game}  -r NCS${game}${i} --sigma0 ${sigma0} --rvalue ${r} -k ${k} --epoch ${epoch} > ./logs/${game}-NCS-100M-${i}.log

    # function test 
    # mpirun -np 7 python ./NCS-C-base.py -e function -f 1 -d 100 -r NCSfunction1 --sigma0 ${sigma0} --rvalue ${r} -k ${k} --epoch ${epoch} --record_interval 20000 > ./logs/f1-NCS-100M-${i}.log
done

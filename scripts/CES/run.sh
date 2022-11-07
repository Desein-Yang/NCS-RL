game=Atlantis
sigma0=0.2
r=0.8
k=7
epoch=10
for i in 1
do
    mpirun -np 40 python ./CES-base.py -g ${game}  -r CES${game}${i} --stepmax 25000000 --lr 1.0 --seed None> ./logs/${game}-CES-100M-${i}.log
    
    # mpirun -np 40 python ./CES-base.py -e function -f 1 -d 100 -r CESfunction1${i} --stepmax 1000000 -k 1 --lr 1.0 --seed 0 --checkpoint_interval 20000 > ./logs/${game}-CES-1M-${i}.log
done

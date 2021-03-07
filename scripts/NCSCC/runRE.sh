game=Alien
lam=8
mu=1
d=100
len=20
cpus=7
k=10
for i in 1
do
    mpirun -np cpus python ./CESRE-base.py -g ${game}  -r RUN-${i} -k ${k} --lam ${lam} --mu ${mu} --effdim ${d} > ./logs/${game}-CESRE-100M-${i}.log
done

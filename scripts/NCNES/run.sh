game=Bowling
for i in 1
do
    # rl test
    mpirun -np 17 python NCNES.py --lam 8 --mu 2 -r debug1 -g Alien
done
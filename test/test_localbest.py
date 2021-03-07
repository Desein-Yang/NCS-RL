from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
pid = comm.Get_rank()
group_cpus = 3
cpus = 10
lam = 3

Bestscore = np.ones(1) * np.random.randint(10)
Bestscore_all = np.zeros((cpus, 1))
Bestparams = np.random.uniform(-1,1,(1,2))
Bestparams_all = np.zeros((cpus, 2))

LocalBestscores = np.zeros((lam, 1))
LocalBestparams = np.empty((lam, 2))
GlobalBestscore = 0
GlobalBestparam = np.zeros((1,2))
comm.Allgather( [Bestscore        ,MPI.DOUBLE],
                [Bestscore_all    ,MPI.DOUBLE])
comm.Allgather( [Bestparams        ,MPI.DOUBLE],
                [Bestparams_all    ,MPI.DOUBLE])
if pid == 0:
    print('Bestall %s id %d'%(str(Bestscore_all.flatten()),pid))
    print('Bestparamsall %s id %d'%(str(Bestparams_all.flatten()),pid))

    for i,score in enumerate(Bestscore_all[1:]):
        # Notes: cpu 0 is not included
        rank = i % group_cpus
        if score > LocalBestscores[rank]:
            LocalBestscores[rank] = score
            LocalBestparams[rank] = Bestparams_all[i+1].copy()


comm.Bcast([LocalBestscores   ,MPI.DOUBLE], root = 0)
comm.Bcast([LocalBestparams   ,MPI.DOUBLE], root = 0)
if pid == 0:
    print('Lbest %s id %d'%(str(LocalBestscores),pid))
    print('Lbestparam %s id %d'%(str(LocalBestparams),pid))
    idx = np.argmax(LocalBestscores)
    print(idx)

    if LocalBestscores[idx] > GlobalBestscore:
        GlobalBestscore = LocalBestscores[idx]
        GlobalBestparam = LocalBestparams[idx].copy()
if pid == 0:
    print(GlobalBestscore)
    print(GlobalBestparam)

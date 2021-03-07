from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.rank

if rank == 0:
    data = {'a':1,'b':2,'c':3}
else:
    data = None

# data 从 rank 0 到所有节点
data = comm.bcast(data, root=0)
print ('rank',rank,data)

from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
cpus = comm.Get_size()

data = np.ones(1)
data = data * rank
data_all = np.empty((cpus, 1))

print(cpus)
print("rank %d, data:" % rank, data)
# comm.Allgather([data, MPI.FLOAT], [data_all, MPI.FLOAT])
print("rank %d, data_all" % rank, data_all)

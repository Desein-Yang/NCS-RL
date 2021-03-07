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
comm.Allgather([data, MPI.FLOAT], [data_all, MPI.FLOAT])
print("rank %d, data_all" % rank, data_all)

# 该api针对numpy中的矩阵，可以把不同线程中的同一个矩阵变量（名字相同）（shape：[a1, a2, .., an]）聚合到一个更大的矩阵A（shape:[cpus, a1, a2, …, an]），并且所有线程中都存在该矩阵A这个变量。

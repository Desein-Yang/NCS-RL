from mpi4py import MPI
import numpy as np
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
cpus = comm.Get_size()

test_dict = {"m": rank}
# 将一个线程某个变量的值同步到其余线程中相同名字的变量上
print("rank %d, initial value %d" %(rank, test_dict['m']))
test_dict = comm.bcast(test_dict, root=0)
print("rank %d, after value %d" %(rank, test_dict['m']))

data = np.ones(1)
data = data * rank
# 该api在功能上bcast相同，但是针对numpy.array类型
print("rank %d, initial value" % rank, data)
comm.Bcast([data, MPI.FLOAT], root=0)
print("rank %d, after value" % rank, data)

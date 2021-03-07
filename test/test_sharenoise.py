import numpy as np
import ctypes, multiprocessing
from mpi4py import MPI
from memory_profiler import profile
from memory_profiler import memory_usage
import tracemalloc
import time

tracemalloc.start()

class SharedNoiseTable(object):
    def __init__(self,seed=123, count = 250000000):
        # 1 gigabyte of 32-bit numbers. Will actually sample 2 gigabytes below
        print('Sampling {} random numbers with seed {}'.format(count, seed))
        self._shared_mem = multiprocessing.Array(ctypes.c_float, count)
        self.noise = np.ctypeslib.as_array(self._shared_mem.get_obj())
        assert self.noise.dtype == np.float32
        self.noise[:] = np.random.RandomState(seed).randn(count)  # 64-bit to 32-bit conversion here
        print('Sampled {} bytes'.format(self.noise.size * 4))

    def get(self, i, dim):
        return self.noise[i:i + dim]

    def sample_index(self, stream, dim, size):
        return stream.randint(0, len(self.noise) - dim + 1, size=size)


#shared_noise_table = SharedNoiseTable()
 

# https://stackoverflow.com/questions/32485122/shared-memory-in-mpi4py



comm = MPI.COMM_WORLD 

# create a shared array of size 1000 elements of type double
size = 1*10**7
cpus = MPI.DOUBLE.Get_size() 
if comm.Get_rank() == 0: 
    nbytes = size * cpus 
else: 
    nbytes = 0

# on rank 0, create the shared block(i.e. nbytes)
# on rank 1 get a handle to it (known as a window in MPI speak)
win = MPI.Win.Allocate_shared(nbytes, cpus, comm=comm) 

# size=nbytes : Size of the memory window in bytes.
# cpus=disp_unit : Local unit size for displacements, in bytes.

# create a numpy array whose data points to the shared mem
buf, cpus = win.Shared_query(0) 
assert cpus == MPI.DOUBLE.Get_size() 
ary = np.ndarray(buffer=buf, dtype='f4', shape=(size,)) 
# d means double float

# in process rank 1:
# write the numbers 0.0,1.0,..,4.0 to the first 5 elements of the array
if comm.rank == 1: 
    ary[:] = np.random.RandomState(123).randn(size,)
    print(ary.__sizeof__())
# wait in process rank 0 until process 1 has written to the array
comm.Barrier() 

# check that the array is actually shared and process 0 can see
# the changes made in the array by process 1
if comm.rank == 0: 
    start_time = time.time()
    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics('lineno')
    print(ary[100:105])
    now_time = time.time()
    print(now_time-start_time)

    print(type(ary[0]))
    print("[ Top 10 ]")
    for stat in top_stats[:10]:
        print(stat)
    
if comm.rank == 2:
    start_time = time.time()
    print(ary[100:105])
    now_time = time.time()
    print(now_time-start_time)

# ----------------------- Conclusion --------------------
# 1. index time don't have a significant difference between different size of 
# noise table(i.e. 10^7 or 10^9) 
# 2. larger noise table will take longger time to generate initially
# 3. memory seem to be same but can't make sure it make sense
# -------------------------------------------------------
# (tf1.5) [yangqi@localhost NCSRE]$ mpirun -np 3 python ./test/test_sharenoise.py 
# 96
# [ 0.6420547  -1.977888    0.71226466  2.598304   -0.02462598]
# 0.002242565155029297
# [ 0.6420547  -1.977888    0.71226466  2.598304   -0.02462598]
# 0.0039675235748291016
# <class 'numpy.float32'>
# [ Top 10 ]
# ./test/test_sharenoise.py:11: size=1648 B, count=7, average=235 B
# ./test/test_sharenoise.py:39: size=1144 B, count=2, average=572 B
# ./test/test_sharenoise.py:24: size=136 B, count=1, average=136 B
# ./test/test_sharenoise.py:21: size=136 B, count=1, average=136 B
# ./test/test_sharenoise.py:12: size=136 B, count=1, average=136 B
# ./test/test_sharenoise.py:51: size=96 B, count=1, average=96 B
# ./test/test_sharenoise.py:53: size=80 B, count=1, average=80 B
# ./test/test_sharenoise.py:45: size=56 B, count=1, average=56 B
# (tf1.5) [yangqi@localhost NCSRE]$ mpirun -np 3 python ./test/test_sharenoise.py 
# 96
# [ 0.6420547  -1.977888    0.71226466  2.598304   -0.02462598]
# 0.0021474361419677734
# [ 0.6420547  -1.977888    0.71226466  2.598304   -0.02462598]
# 0.002251148223876953
# <class 'numpy.float32'>
# [ Top 10 ]
# ./test/test_sharenoise.py:11: size=1648 B, count=7, average=235 B
# ./test/test_sharenoise.py:39: size=1140 B, count=2, average=570 B
# ./test/test_sharenoise.py:24: size=136 B, count=1, average=136 B
# ./test/test_sharenoise.py:21: size=136 B, count=1, average=136 B
# ./test/test_sharenoise.py:12: size=136 B, count=1, average=136 B
# ./test/test_sharenoise.py:51: size=96 B, count=1, average=96 B
# ./test/test_sharenoise.py:53: size=80 B, count=1, average=80 B
# ./test/test_sharenoise.py:45: size=56 B, count=1, average=56 B

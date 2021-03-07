#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   test_noisetable_mpi.py
@Time    :   2020/09/10 21:30:09
@Author  :   Qi Yang
@Version :   1.0
@Describtion:  Noise table class test (which is from test.sharednoise.py)
                Use MPI allocated shared memory
'''
import numpy as np
import ctypes, multiprocessing
from mpi4py import MPI
from memory_profiler import profile
from memory_profiler import memory_usage
import tracemalloc
import time

class NoiseTable(object):
    def __init__(self,size,seed):
        comm = MPI.COMM_WORLD 
        self.rank = comm.Get_rank()
        self.cpus = MPI.DOUBLE.Get_size() 
        if self.rank == 0: 
            self.nbytes = size * self.cpus 
        else: 
            self.nbytes = 0
        win = MPI.Win.Allocate_shared(self.nbytes, self.cpus, comm=comm) 
        buf, cpus = win.Shared_query(0) 
        assert cpus == MPI.DOUBLE.Get_size() 
        self.ary = np.ndarray(buffer=buf, dtype='f4', shape=(size,)) 

        if self.rank == 0: 
            self.ary[:] = np.random.RandomState(seed).randn(size,)
        comm.Barrier() 

    def randn(self,index,size):
        if type(size) is int:
            return self.ary[index:index+size]
        elif type(size) is tuple and len(size) == 2:
            tmp = np.random.RandomState(index).randint(low=1,size=(size[0],))
            a = np.ndarray(shape=size,dtype='f4')
            for i in tmp:
                a[i,:] = self.ary[index:index+size[1]]
            return a

def test():
    import time
    start = time.time()
    sharedNoisetable = NoiseTable(10**6,123)
    comm = MPI.COMM_WORLD 
    if comm.Get_rank() == 0:
        print(sharedNoisetable.randn(11,5))
        print(sharedNoisetable.randn(11,(5,2)))
        print(time.time()-start)
    if comm.Get_rank() == 1:
        sharedNoisetable.randn(11,(10000,2))
        sharedNoisetable.randn(11,5)

        print(time.time()-start)

# Conclusion
# 1 . get random vector with different size cost similarily(5 or 10000)
# 1 . get random matrix with different size cost similarily(5 or 10000)

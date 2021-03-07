#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   noise.py
@Time    :   2020/09/10 21:32:07
@Author  :   Qi Yang
@Version :   1.0
@Describtion:  Building Noise Table with a  Allocated Memory Block with MPI
'''

import numpy as np
from mpi4py import MPI

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

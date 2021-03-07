#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   sct14.py
@Time    :   2020/04/21 16:13:13
@Describtion:  test sysonevalue and sysonevector
'''

# here put the import lib
from mpi4py import MPI
import numpy as np
import time
import logging


def syncOneVector(v,cpus):
    comm = MPI.COMM_WORLD
    v_t = np.array(v)
    v_all = np.zeros((cpus,v.shape[0]))
    comm.Allgather([v_t, MPI.DOUBLE], [v_all, MPI.DOUBLE])
    return v_all.flatten()

def syncOneValue(v, cpus):
    """工具函数，用于同步每个线程的单个标量值到每个线程的向量中。
    对mpi的Allgather的简单封装

    参数：
        v：标量
    返回值：
        np.ndarray ： 大小为 cpus
    """
    v_t = np.array([v], dtype='d')
    v_all = np.zeros((cpus, 1), dtype= 'd')
    # Notes: it should be add dtype
    comm.Allgather([v_t, MPI.DOUBLE], [v_all, MPI.DOUBLE])
    return v_all.flatten()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
cpus = comm.Get_size()

# params = np.ones((3,)) * rank
# params_all = syncOneVector(params)
# if rank == 0:
#     print(params_all)

params0 = rank
params = np.array(params0) * rank
print(params)
params_all = syncOneValue(params, cpus)
if rank == 0:
    print(params_all)
    cost_steps = np.sum(params_all)
    print(cost_steps)

#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   sct13.py
@Time    :   2020/04/21 16:13:42
@Author  :   Qi Yang
@Version :   1.0
@Describtion:  test simple ncs and mpi api allgather
'''

# here put the import lib
from mpi4py import MPI
import numpy as np
import time
import logging
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
cpus = comm.Get_size()

params = np.ones((3,))
params_all = np.empty((cpus,3))
if rank != 0:
    params = np.ones((3,)) * rank
x_best = np.ones((3,))
y_best = 0

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(level=logging.DEBUG,
                    format="%(asctime)s %(name)s %(levelname)s %(pathname)s %(message)s",
                    datefmt = '%Y-%m-%d  %H:%M:%S %a ' ,
                    filename='log'+str(rank)+'.log' #有了filename参数就不会直接输出显示到控制台，而是直接写入文件
                    )
for i in range(1):
    comm.Bcast([params,MPI.FLOAT],root = 0)
    logger.info("bcast")
    logger.info('rank  %d initial %s %s'% (rank,str(params),str(params_all)))
    if rank != 0:
        time.sleep(2)
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        for i in range(3):
            x_i = params + np.random.randn()
            fitness = np.sum(x_i)
            if fitness >= y_best:
                x_best,y_best = x_i,fitness
        params = x_best
        logger.info('ncs is runing in rank %d'% rank)
comm.Allgather([params,MPI.FLOAT],[params_all,MPI.FLOAT])
logger.info('rank %d finish %s %s'% (rank,str(params),str(params_all)))


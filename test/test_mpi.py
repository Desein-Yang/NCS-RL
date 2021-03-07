# 写一个二阶主从结构测试
# by MPI 



from mpi4py import MPI
import time
import numpy as np
from src.logger import Logger
from src.testfunc import TestEnv

class Master_slave(object):
    '''
    rank 0 : master process for ncsre
    rank 1,11,21,...,71: 8 subprocess master for ncs
    rank 2-10,12-20,..: 9 subprocess for individual
    '''
    def __init__(self,problem_t):
        self.mu = 3
        self.lam = 3
        self.n_episode = 5
        self.logger = Logger('test_log')
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.cpus = self.comm.Get_size()
        assert self.cpus == self.n_worker * self.lam + 1
        self.params = np.ones(1)
        self.params_best = np.ones(1)
        self.params_all = np.empty((self.cpus,1))

        self.score_best = 0
        self.score_all = np.empty((self.cpus,1))

        self.comm.Allgather([self.params_best,MPI.FLOAT],[self.params_all,MPI.FLOAT])
        self.comm.Allgather([self.score_best,MPI.FLOAT],[self.score_all,MPI.FLOAT])
        self.env = TestEnv(1,problem_t)
        self.seed = np.random.randint(100000)
        self.rs = np.random.RandomState(self.seed)

    def run(self):
        for epi in range(self.n_episode):
            self.comm.Bcast([self.params,MPI.FLOAT],root = 0)
            if self.rank != 0: 
                self.ncs()
            self.comm.Allgather([self.params_best,MPI.FLOAT],[self.params_all,MPI.FLOAT])
            self.logger.log('finish %d params all%s score %s'%(epi,str(self.params_all),str(self.score_best)))

    def ncs(self):
        x_best,y_best = 0,0
        self.logger.log('rank is %d'%self.rank)
        for i in range(3):
            x_i = self.params + self.rs.standard_normal(self.params.shape)
            fitness = self.env.get_fitness(self.params)
            if fitness >= y_best:
                x_best,y_best = x_i,fitness
        self.logger.log('params %s'%str(x_best))
        self.logger.log('score %s'%str(y_best))
        self.params_best = x_best  
        self.score_best = y_best
        # comm.send((rank,x_best,y_best),dest = int(rank/10)+1)

if __name__ == "__main__":
    algo = Master_slave(1)
    algo.run()


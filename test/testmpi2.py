from mpi4py import MPI
import numpy as np


class testAllgatherObject(object):

    def __init__(self):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.cpus = self.comm.Get_size()

        self.data = np.ones(1) * self.rank
        self.data_all = np.empty((self.cpus, 1))
        print("rank %d, data:" % self.rank, self.data)

    def allgather(self):
        self.comm.Allgather([self.data, MPI.FLOAT], [self.data_all, MPI.FLOAT])
        print("rank %d, "% self.rank, self.data_all)

if __name__ == "__main__":
    testa = testAllgatherObject()

    testa.allgather()

from mpi4py import MPI
import numpy as np


def testAllgather():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    cpus = comm.Get_size()

    data = np.ones(1)
    data = data * rank
    data_all = np.empty((cpus, 1))
    print("rank %d, data:" % rank, data)
    comm.Allgather([data, MPI.FLOAT], [data_all, MPI.FLOAT])
    print("rank %d, " % rank, data_all)

def testBcast():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    cpus = comm.Get_size()

    data = np.ones(1)
    data = data * rank

    print("rank %d, initial value" % rank, data)
    comm.Bcast([data, MPI.FLOAT], root=0)
    print("rank %d, after value" % rank, data)

def testbcast():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    cpus = comm.Get_size()

    test_dict = {"m": rank}

    print("rank %d, initial value %d" %(rank, test_dict['m']))
    test_dict = comm.bcast(test_dict, root=0)
    print("rank %d, after value %d" %(rank, test_dict['m']))

if __name__ == "__main__":
    testAllgather()
    # testbcast()
    # testBcast()

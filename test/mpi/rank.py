from mpi4py import MPI

comm = MPI.COMM_WORLD

print ("my rank is:",comm.rank)

if comm.rank == 1:
    print ('Doing the task of Rank 1')

if comm.rank == 0:
    print ('Doing the task of Rank 0!')

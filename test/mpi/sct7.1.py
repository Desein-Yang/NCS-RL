from mpi4py import MPI

comm = MPI.COMM_WORLD
rank=comm.rank
size=comm.size
name=MPI.Get_processor_name()


if rank == 0:
    shared = {'d1':55,'d2':42}
    comm.send(shared, dest=1)

if rank == 1:
    receive = comm.recv(source=0)
    print (receive)
    print (receive['d1'])

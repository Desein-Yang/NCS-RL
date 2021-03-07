from mpi4py import MPI

comm = MPI.COMM_WORLD
rank=comm.rank
size=comm.size
name=MPI.Get_processor_name()

if rank == 0:
    shared = {'d1':55,'d2':42}
    comm.send(shared, dest=1, tag=1)

    shared2 = {'d3':25,'d4':22}
    comm.send(shared2, dest=1, tag=2)

if rank == 1:
    receive = comm.recv(source=0, tag=1)
    print (receive)
    receive2 = comm.recv(source=0, tag=2)
    print (receive2)

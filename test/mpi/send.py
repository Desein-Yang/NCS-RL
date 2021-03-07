from mpi4py import MPI
import time

comm = MPI.COMM_WORLD

rank = comm.rank
size = comm.size
name = MPI.Get_processor_name()

shared = (rank+1)*5

# rank 0 负责发送
# rank 1,2 负责接受
if rank == 0:
    data = shared
    comm.send(data, dest=1)
    comm.send(data, dest=2)
    print ('From rank',name,'we sent',data)
    time.sleep(5)

elif rank == 1:
    data = comm.recv(source=0)
    print ('on node',name, 'we received:',data)


elif rank == 2:
    data = comm.recv(source=0)
    print ('on node',name, 'we received:',data)
		

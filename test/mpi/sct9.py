from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

if rank == 0:
   data = [(x+1)**x for x in range(size)]
   print ('we will be scattering:',data)
else:
   data = None
   
# scatter 从rank 0 分发到其他
data = comm.scatter(data, root=0)
print ('rank',rank,'has data:',data)

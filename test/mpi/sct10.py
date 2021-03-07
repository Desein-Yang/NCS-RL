from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

if rank == 0:
   data = [(x+1)**x for x in range(size)]
   print ('we will be scattering:',data)
else:
   data = None
   
data = comm.scatter(data, root=0)
data += 1
print ('rank',rank,'has data:',data)

# 和分发相反 把rank.data 集中到rank 0 的新信息中
newData = comm.gather(data,root=0)

if rank == 0:
   print ('master:',newData)

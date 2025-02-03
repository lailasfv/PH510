#!/bin/python3

import numpy as np
import time
from mpi4py import MPI

# MPI.Init()
comm = MPI.COMM_WORLD

# MPI-related data
rank = comm.Get_rank()
size = comm.Get_size()
nworkers = size # processor rank 0 is leader, the rest are workers

# Set up initial message 
sum1 = np.array(0, dtype=np.float64)

# Integral setup
# Assuming same total number of points regardless of no. processors
# Confirming parallelisation requires all test to run over the same
# total number of points
num_points = int(10000000/nworkers)
dx = (nworkers*(num_points-1))**(-1)

# Integration function
def integrand(x):
	return (4.0 / (1.0 + x**2))

timeStart = time.time()
if rank!=0: # Workers calculate and send their own "chunk" 
	for x_var in np.linspace(rank/nworkers, (rank+1)/nworkers, num_points)[:-1]:
		sum1 += integrand(x_var+1/2*dx) * dx
	request = comm.Isend(sum1, dest=0)
	request.wait()
else: # Leader rank 0 calculates its own "chunk" and receives data from workers
	final_sum = 0
	for x_var in np.linspace(rank/nworkers, (rank+1)/nworkers, num_points)[:-1]:
		final_sum += integrand(x_var+1/2*dx) * dx
	for i in range (1, nworkers):
		comm.Recv(sum1, source=i)
		final_sum += sum1
	# The leader rank is the last rank to process the data
	timeEnd = time.time()
	timeTaken = timeEnd-timeStart
		
	print("The final calculation of pi for", nworkers, "workers is equal to", final_sum)
	print("Total time taken:", timeTaken, "s")

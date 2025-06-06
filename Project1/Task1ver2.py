#!/bin/python3

import numpy as np
from mpi4py import MPI

# MPI.Init()
comm = MPI.COMM_WORLD

# MPI-related data
rank = comm.Get_rank()
size = comm.Get_size()
nworkers = size # processor rank 0 is leader, the rest are workers

# Set up initial message 
sum1 = np.array(0, dtype=np.float64)

# Integration function
# Assuming same total number of points regardless of no. processors
# Could be useful to have a timer to see if adding more processors 
# improves the time, so then we know if parallelisation is working
num_points = int(10000000/nworkers)
def integrand(x):
	return (4.0 / (1.0 + x**2))

dx = (nworkers*(num_points-1))**(-1)

if rank!=0: # Workers
	for x_var in np.linspace(rank/nworkers, (rank+1)/nworkers, num_points)[:-1]:
		sum1 += integrand(x_var+1/2*dx) * dx
	print("Rank", rank, "contribution", sum1)
	request = comm.Isend(sum1, dest=0)
	request.wait()
else: # Leader rank 0
	final_sum = 0
	for x_var in np.linspace(rank/nworkers, (rank+1)/nworkers, num_points)[:-1]:
		final_sum += integrand(x_var+1/2*dx) * dx
	print("Rank 0 contribution:", final_sum)
	for i in range (1, nworkers):
		comm.Recv(sum1, source=i)
		print("Received from rank", i)
		final_sum += sum1
		
	print("The final calculation of pi for", nworkers, "workers is equal to", final_sum)

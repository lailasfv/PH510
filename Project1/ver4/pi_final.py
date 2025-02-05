#!/bin/python3
"""This module uses a definite integral to calculate the value of pi.
This utilises parallelisation and sends chunks of the integral to
each core

MIT License

Copyright (c) 2025 by Tyler Chauvy, Laila Safavi - University of Strathclyde

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE."""

import time
import numpy as np
from mpi4py import MPI

# MPI.Init()
comm = MPI.COMM_WORLD

# MPI-related data
rank = comm.Get_rank()
size = comm.Get_size()
nworkers = size  # processor rank 0 is leader, the rest are workers

# Set up initial message
sum1 = np.array(0, dtype=np.float64)

# Integral setup
# Assuming same total number of points regardless of no. processors
# Confirming parallelisation requires all test to run over the same
# total number of points
num_points = int(100000000/nworkers)
dx = (nworkers*(num_points-1))**(-1)


def integrand(x_point):
    """
    Function we wish to integrate

    Parameters
    ----------
    x :float
        Point at which we evaluate the function at.

    Returns
    -------
    float
        f(x)=4.0/(1+x^2).

    """
    return 4.0 / (1.0 + x_point**2)


timeStart = time.time()
if rank != 0:  # Workers calculate and send their own "chunk"
    for x_var in np.linspace(rank/nworkers, (rank+1)/nworkers,
                             num_points)[:-1]:
        sum1 += integrand(x_var+1/2*dx) * dx
    request = comm.Isend(sum1, dest=0)
    request.wait()  # Ensures rank 0 is ready to receive
else:  # Leader rank 0 calculates its own "chunk"
    FINALSUM = 0
    for x_var in np.linspace(rank/nworkers, (rank+1)/nworkers,
                             num_points)[:-1]:
        FINALSUM += integrand(x_var+1/2*dx) * dx
    for i in range(1, nworkers):  # Takes values from all workers
        comm.Recv(sum1, source=i)
        FINALSUM += sum1
    # The leader rank is the last rank to process the data
    timeEnd = time.time()
    timeTaken = timeEnd-timeStart

    print(f"The final calculation of pi for {nworkers} is equal to %.15f" %
          FINALSUM)
    print("Total time taken:", timeTaken, "s")

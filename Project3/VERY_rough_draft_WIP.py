#!/bin/python3
"""
Created on Fri Feb 21 18:55:08 2025

@author: skyed
"""

import numpy as np
from mpi4py import MPI

# MPI.Init()
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

from numpy.random import SeedSequence, default_rng


class Monte_Carlo:
    def __init__(self,starts, ends, N, f):
        self.starts = starts
        self.ends = ends
        self.f = func
        self.N = N
        self.data = self.integral()

    def __str__(self):
        return f"(Integral: {self.data[0]}, Var: {self.data[1]}, Err: {self.data[2]})"
    
    def integral(self):
        dim = len(self.starts)
        nworkers = size  # processor rank 0 is leader, the rest are workers
        ss = SeedSequence(12345) # getting the random numbers
        child_seeds = ss.spawn(nworkers) # random numbers for each worker
        streams = [default_rng(s) for s in child_seeds]
        points = streams[rank].random((self.N,dim)) # getting the random numbers in arrays we like
        for p in points:
            count = 0
            while count<dim:
                p[count]=p[count]*(self.ends[count]-self.starts[count])+self.starts[count]  # making sure they are in the interval we need them to be
                count=count+1
        # AND LOOK INTO THE INFINITY CASE
        sum_f = np.array(0, dtype=np.float64) # Sending messages in MPI comm requires array
        expect_f_squared = np.array(0, dtype=np.float64)
        for p in points:
            sum_f = sum_f + (self.f(p)) # we get sum(f) for each worker
            expect_f_squared = expect_f_squared + (self.f(p))**2 # we get sum(f**2) for each worker

        FINAL_SUM_F = np.empty(dim, dtype=np.float64)  # Needs to have the same dimensions as the message it's receiving
        FINAL_F_SQUARED = np.empty(dim, dtype=np.float64) 

        comm.Allreduce(sum_f, FINAL_SUM_F)  # These take value of sum_f for all ranks and sum them into FINAL_...
        comm.Allreduce(expect_f_squared, FINAL_F_SQUARED)
        
        prefactor1=1 # this will be used to create the (b-a)(d-c)...
        prefactor2=1/(num_points*(size+1)) # MAKE SURE THE CORRECT NUMBER OF POINTS HERE
        d=0
        while d<dim:
            prefector1=prefactor1*(self.ends[d]-self.starts[d]) # we get our (b-a)(c-d...)
            d=d+1
        FINAL_I = prefactor1*prefactor2*FINAL_SUM_F / nworkers # our integral
        FINAL_VAR = prefactor2*(FINAL_F_SQUARED*prefactor2-(FINAL_SUM_F*prefactor2)**2) # our variance
        FINAL_ERROR = prefactor1*FINAL_VAR # our error

        return np.array([FINAL_I, FINAL_VAR, FINAL_ERROR])


def func(x):
    return x**2

def circle_unit(x):
    return 2 * np.sqrt(1-x**2)

num_points = int(100000)

AH = Monte_Carlo([2], [3], num_points, func)

circle = Monte_Carlo ([1], [5], num_points, circle_unit)
if rank ==0:
    print(f"Test function x^2 between 2 and 3: {AH}")
    print(f"Circle with x 1 to 5: {circle}")

MPI.Finalize()

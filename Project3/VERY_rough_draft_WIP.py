#!/bin/python3
"""
Created on Fri Feb 21 18:55:08 2025

@author: skyed
"""


import numpy as np
from mpi4py import MPI

# MPI.Init()
comm = MPI.COMM_WORLD
from numpy.random import SeedSequence, default_rng


class Monte_Carlo:
    def __init__(self,starts, ends, N, f):
        self.starts = starts
        self.ends = ends
        self.f = func
        self.N = N
        self.data = self.integral()

    def __str__(self):
        return f"({self.data[0]:6f}, {self.data[1]:6f}, {self.data[2]:6f})"
    
    def integral(self):
        dim = len(self.starts)
        rank = comm.Get_rank()
        size = comm.Get_size()
        nworkers = size  # processor rank 0 is leader, the rest are workers
        ss = SeedSequence(12345) # getting the random numbers
        child_seeds = ss.spawn(nworkers) # random numbers for each worker
        streams = [default_rng(s) for s in child_seeds]
        count = 0
        points = streams[rank].random((self.N,dim)) # getting the random numbers in arrays we like
        for p in points:
            while count<dim:
                p[count]=p[count]*(self.ends[count]-self.starts[count])+self.starts[count]  # making sure they are in the interval we need them to be
                count=count+1
        # AND LOOK INTO THE INFINITY CASE
        sum_f = np.array(0, dtype=np.float64) # Sending messages in MPI comm requires array
        expect_f_squared = np.array(0, dtype=np.float64)
        for p in points:
            sum_f = sum_f + (self.f(p)) # we get sum(f) for each worker
            expect_f_squared = expect_f_squared + (self.f(p))**2 # we get sum(f**2) for each worker
        # TEST PRINTS
        print("Rank", rank, "sum_f equals:", sum_f)
        print("Rank", rank, "f_square equals:",expect_f_squared)

        # CHANGE MADE HERE: This process is now done for all ranks. Explanation:
        #                   Instead of just rank 0 taking in all of the final sums
        #                   allreduce takes the values of sum_f and exp_f_sq and sums of these
        #                   for all ranks, otherwise any rank that is not 0 is not returning anything
        #                   and this creates an error in the final class method line

        FINAL_SUM_F = np.array(0, dtype=np.float64)  # This is a receiving message for allreduce
        FINAL_F_SQUARED = np.array(0, dtype=np.float64) # IT NEEDS TO BE THE SAME SHAPE AS THE MESSAGE, the message has *dim* elements - need to figure this out

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
        FINAL_ERROR= prefactor1*FINAL_VAR # our error

        return FINAL_I, FINAL_VAR, FINAL_ERROR


def func(x):
    return x**2

def step(x,y):  # FUNCTION FOR CIRCLE
    return 1 * (round(x**2+y**2,5) <= 1)

num_points = int(100)

AH = Monte_Carlo([2], [3], num_points, func)

circle = Monte_Carlo ([1,2], [5,6], num_points, step)
if rank ==0:
    print("Test function x^2 between 2 and 3:", AH)
    print("Circle area with x from 1-5 and y from 2-6:", circle)

MPI.Finalize()

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
        if rank != 0:  # Workers calculate and send their own "chunk"
            count = 0
            points = streams[rank].random((self.N,dim)) # getting the random numbers in arrays we like
            for p in points:
                while count<dim:
                    p[count]=p[count]*(self.ends[count]-self.starts[count])+self.starts[count]  # making sure they are in the interval we need them to be
                    count=count+1
            # AND LOOK INTO THE INFINITY CASE
            sum_f = 0
            expect_f_squared=0
            for p in points:
                sum_f=sum_f+self.f(p) # we get sum(f) for each worker
                expect_f_squared = expect_f_squared + (self.f(p))**2 # we get sum(f**2) for each worker
            # we send everything to the leader
            request = comm.Isend(sum_f, dest=0)
            request.wait()  # Ensures rank 0 is ready to receive
            request2 = comm.Isend(expect_f_squared, dest=0)
            request2.wait()  # Ensures rank 0 is ready to receive
        else:  # Leader rank 0 calculates its own "chunk"
            # CHANGE THINGS HERE TO MATCH PREVIOUS
            FINAL_SUM_F = 0
            FINAL_F_SQUARED=0
            count = 0
            points=streams[rank].random((self.N,dim)) # getting the random points
            for p in points:
                while count<dim:
                    p[count]=p[count]*(self.ends[count]-self.starts[count])+self.starts[count] # making sure the interval is corrrect
                    count=count+1
            for p in points: # getting the 2 sums we need
                FINAL_SUM_F += self.f(p)
                FINAL_F_SQUARED = expect_f_squared + (self.f(p))**2

            for i in range(1, nworkers):  # Takes values from all workers
                comm.Recv(sum_f, source=i)
                FINAL_SUM_F += sum_f
                comm.Recv(expect_f_squared, source=i)
                FINAL_F_SQUARED +=expect_f_squared
            #calculating the integral, variance and error
            prefactor1=1 # this will be used to create the (b-a)(d-c)...
            prefactor2=1/(num_points*(size+1)) # MAKE SURE THE CORRECT NUMBER OF POITNS HERE
            d=0
            while d<dim:
                prefector1=prefactor1*(self.ends[d]-self.starts[d]) # we get our (b-a)(c-d...)
                d=d+1
            FINAL_I = prefactor1*prefactor2*FINAL_SUM_F # our integral
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

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
nworkers = comm.Get_size()

from numpy.random import SeedSequence, default_rng


class Monte_Carlo:
    def __init__(self,starts, ends, N, func2):
        self.starts = starts
        self.ends = ends
        self.f = func2
        self.N = N
        self.data = self.integral()

    def __str__(self):
        return f"(Integral: {self.data[0]}, Var: {self.data[1]}, Err: {self.data[2]})"
    
    def integral(self):
        dim = len(self.starts)
        R = self.starts[0] # Radius of shape if shape
        ss = SeedSequence(12345) # getting the random numbers
        child_seeds = ss.spawn(nworkers) # random numbers for each worker
        streams = [default_rng(s) for s in child_seeds]
        points = streams[rank].random((self.N,dim)) # getting the random numbers in arrays we like

        sum_f = np.array(0, dtype=np.float64) # Sending messages in MPI comm requires array
        expect_f_squared = np.array(0, dtype=np.float64)

        for p in points:
            count = 0
            while count<dim:
                p[count]=p[count]*(self.ends[count]-self.starts[count])+self.starts[count]  # making sure they are in the interval we need them to be             
                count=count+1
            sum_f = sum_f + (self.f(p, R)) # we get sum(f) for each worker
            expect_f_squared = expect_f_squared + (self.f(p, R))**2 # we get sum(f**2) for each worker

        FINAL_SUM_F = np.array(0, dtype=np.float64)
        FINAL_F_SQUARED = np.array(0, dtype=np.float64)

        comm.Allreduce(sum_f, FINAL_SUM_F)  # These take value of sum_f for all ranks and sum them into FINAL_...
        comm.Allreduce(expect_f_squared, FINAL_F_SQUARED)
        
        prefactor1=1 # this will be used to create the (b-a)(d-c)...
        prefactor2=1/(num_points*nworkers) 
        d=0
        while d<dim:
            prefactor1=prefactor1*(self.ends[d]-self.starts[d]) # we get our (b-a)(c-d...)
            d=d+1
        #print("pref1",prefactor1)
        #print("pref2", prefactor2)
        FINAL_I = prefactor1*prefactor2*FINAL_SUM_F / nworkers # our integral
        FINAL_VAR = prefactor2*(FINAL_F_SQUARED*prefactor2-(FINAL_SUM_F*prefactor2)**2) # our variance
        FINAL_ERROR = prefactor1*FINAL_VAR # our error

        return np.array([FINAL_I, FINAL_VAR, FINAL_ERROR])


def step(x, R):  # FUNCTION FOR ANY ROUND SHAPE
    return 1 * (round(np.sum(np.square(x)),5) <= (R**2))


num_points = int(200000)

radius = 1
radius2 = 3

dimensions = 5

a = np.repeat(-radius, dimensions)
b = np.repeat(radius, dimensions)

a2 = np.repeat(-radius2, dimensions)
b2 = np.repeat(radius2, dimensions)

sphere = Monte_Carlo (a, b, num_points, step)
sphere2 = Monte_Carlo (a2, b2, num_points, step)

if rank == 0:
    print(f"5D Sphere with radius {radius}: {sphere}")
    print(f"5D Sphere with radius {radius2}: {sphere2}")

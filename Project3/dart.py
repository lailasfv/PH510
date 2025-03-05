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

# we will not keep this as a secondary class/file, I just want to check if it works 

# NOTES FOR IMPROVING IMPLEMENTATION
# 1. need to change self.data and the __str__ to not just look at one method so we can use same class for both methods
# 2. The randomisation is still the same too, so we can still separate that into method


class Monte_Dart:
    def __init__(self, starts, ends, N, func):
        self.starts = starts
        self.ends = ends
        self.f = func                 # function for shape
        self.N = N                    # number of points
        self.data = self.contain()    # the class method

    def __str__(self):
        return f"({self.data})"

    def contain(self):
        dim = len(self.starts)
        # THIS SHOULD WORK FOR ANY NUMBER OF DIMENSIONS
        nworkers = size  # processor rank 0 is leader, the rest are workers
        ss = SeedSequence(12345) # getting the random numbers
        child_seeds = ss.spawn(nworkers) # random numbers for each worker
        streams = [default_rng(s) for s in child_seeds]
        points = streams[rank].random((self.N, dim)) # getting the random numbers in arrays we like

        for p in points:
            count = 0
            while count<dim:
                p[count]=p[count]*self.ends[0] # making sure they are in the interval we need them to be
                count=count+1

        sum_f = np.array(0, dtype=np.float64) # Sending messages in MPI comm requires array

        for p in points:
            sum_f = sum_f + self.f(p, self.ends[0]) # The radius is equal to the "end" value which should be the same in all dims

        CONTAINED = np.array(0, dtype=np.float64)  
        comm.Allreduce(sum_f, CONTAINED)  # The total number of points in the shape

        # The ratio of points inside shape : points inside unit shape = volume ratio (e.g. sphere:cube)
        # 2R bc that's the length of the cube

        measure = (2*self.ends[0])**dim * CONTAINED/(self.N*nworkers)

        return measure



def circle_dart(xy, R):
    if np.sum(np.square(xy)) <= (R**2):
        thing = 1
    else:
        thing = 0
    return thing

def sphere_dart(xyz, R):
    if np.sum(np.square(xyz)) <= (R**2):
        thing = 1
    else:
        thing = 0
    return thing

def thing_dart(xyzt, R):
    if np.sum(np.square(xyzt)) <= (R**2):
        thing = 1
    else:
        thing = 0
    return thing


num_points = int(100000)

R = 3
circle_dim = 2
sph_dim = 3
thing_dim = 4

a = np.repeat(-R, circle_dim)  # starts
b = np.repeat(R, circle_dim)   # ends

a2 = np.repeat(-R, sph_dim)
b2 = np.repeat(R, sph_dim)

a3 = np.repeat(-R, thing_dim)
b3 = np.repeat(R, thing_dim)

test = Monte_Dart(a, b, num_points, circle_dart)
real = np.pi*R**2

test2 = Monte_Dart(a2, b2, num_points, sphere_dart)
real2 = 4/3*np.pi*R**3

test3 = Monte_Dart(a3, b3, num_points, thing_dart)
real3 = 1/2*np.pi**2*R**4


if rank == 0:
    print("Circle with R = ", R)
    print("Monte-Carlo value:", test)
    print("Real value:", real)
    print("Sphere with R = ", R)
    print("Monte-Carlo value:", test2)
    print("Real value:", real2)
    print("4D 'sphere' with R = ", R)
    print("Monte-Carlo value:", test3)
    print("Real value:", real3)

MPI.Finalize()

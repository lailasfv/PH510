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
# 1: R is assumed to just be 1, I'm sure we can change this for it to work with the other constructor 
# 1. (cont'd) the circle needs to be centred on the origin for this technique to work though?
# 1. (cont'd) if it is a child class then start and end could just be -R and +R, and pass into class as np.repeat(R, dim)

# 2. also need to change self.data and the __str__ to not just look at one method so we can use same class for both methods
# 3. The randomisation is still the same too, so we can still separate that into method

class Monte_Dart:
    def __init__(self, dim, N, func):
        self.dim = dim                # number of dimensions
        self.f = func                 # function for shape
        self.N = N                    # number of points
        self.data = self.contain()    # the class method

    def __str__(self):
        return f"({self.data})"

    def contain(self):
        # THIS SHOULD WORK FOR ANY NUMBER OF DIMENSIONS
        nworkers = size  # processor rank 0 is leader, the rest are workers
        ss = SeedSequence(12345) # getting the random numbers
        child_seeds = ss.spawn(nworkers) # random numbers for each worker
        streams = [default_rng(s) for s in child_seeds]
        points = streams[rank].random((self.N, self.dim)) # getting the random numbers in arrays we like

        sum_f = np.array(0, dtype=np.float64) # Sending messages in MPI comm requires array

        for p in points:
            sum_f = sum_f + (self.f(p)) 

        CONTAINED = np.array(0, dtype=np.float64)  
        comm.Allreduce(sum_f, CONTAINED)  # The total number of points in the shape

        # The ratio of points inside shape : points inside unit shape = volume ratio (e.g. sphere:cube)
        # 2R bc that's the length of the cube

        measure = 2**self.dim * CONTAINED/(self.N*nworkers)

        return measure



def circle_dart(xy):
    if xy[0]**2 + xy[1]**2 <=1:
        thing = 1
    else:
        thing = 0
    return thing

def sphere_dart(xyz):
    if xyz[0]**2 + xyz[1]**2 + xyz[2]**2 <=1:
        thing = 1
    else:
        thing = 0
    return thing


num_points = int(100000)
R = 1

test = Monte_Dart(2, num_points, circle_dart)
test_compare = np.pi*R**2

test2 = Monte_Dart(3, num_points, sphere_dart)
test2_compare = 4/3*np.pi*R**3

if rank == 0:
    print("Circle")
    print("Monte carlo value:", test)
    print("Actual value:", test_compare)
    print("Sphere")
    print("Monte carlo value:", test2)
    print("Actual value:", test2_compare)

MPI.Finalize()

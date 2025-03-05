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
<<<<<<< HEAD:Project3/dart.py
# 1: R is assumed to just be 1, I'm sure we can change this for it to work with the other constructor 
# 1. (cont'd) the circle needs to be centred on the origin for this technique to work though?
# 1. (cont'd) if it is a child class then start and end could just be -R and +R, and pass into class as np.repeat(R, dim)

# 2. also need to change self.data and the __str__ to not just look at one method so we can use same class for both methods
# 3. The randomisation is still the same too, so we can still separate that into method
=======
# 1. need to change self.data and the __str__ to not just look at one method so we can use same class for both methods
# 2. The randomisation is still the same too, so we can still separate that into method
>>>>>>> 3baca39 (trying to use same constructor):Project3/dart2.py

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
                p[count]=p[count]*self.ends[count] # making sure they are in the interval we need them to be
                count=count+1

        sum_f = np.array(0, dtype=np.float64) # Sending messages in MPI comm requires array

        for p in points:
            sum_f = sum_f + (self.f(p)) 

        CONTAINED = np.array(0, dtype=np.float64)  
        comm.Allreduce(sum_f, CONTAINED)  # The total number of points in the shape

        # The ratio of points inside shape : points inside unit shape = volume ratio (e.g. sphere:cube)
        # 2R bc that's the length of the cube

        measure = 2**dim * CONTAINED/(self.N*nworkers)

        return measure



def circle_dart(xy, R):
    if xy[0]**2 + xy[1]**2 <=R:
        thing = 1
    else:
        thing = 0
    return thing

def sphere_dart(xyz, R):
    if xyz[0]**2 + xyz[1]**2 + xyz[2]**2 <=R:
        thing = 1
    else:
        thing = 0
    return thing


num_points = int(100000)
R = 1
dim = 2  # for circle, dim=2, dim=3 for sphere, etc
a = np.repeat([-R, dim])  # starts
b = np.repeat([R, dim])   # ends

test = Monte_Dart(a, b, num_points, circle_dart)
test_compare = np.pi*R**2

dim2 = 3
a2 = np.repeat([-R, dim2])  # starts
b2 = np.repeat([R, dim2])   # ends

test2 = Monte_Dart(a2, b2, num_points, sphere_dart)
test2_compare = 4/3*np.pi*R**dim2

if rank == 0:
    print("Circle")
    print("Monte carlo value:", test)
    print("Actual value:", test_compare)
    print("Sphere")
    print("Monte carlo value:", test2)
    print("Actual value:", test2_compare)

MPI.Finalize()

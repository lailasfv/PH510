# -*- coding: utf-8 -*-
"""
Created on Wed Mar  5 18:08:28 2025

@author: skyed
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Mar  5 17:07:38 2025

@author: skyed
"""

#!/bin/python3
"""
Created on Fri Feb 21 18:55:08 2025

@author: skyed
"""

import numpy as np

# MPI.Init()

rank = 0
size = 1

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
        print("dim",dim)
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
        #print("points", points)
        sum_f = np.array(0, dtype=np.float64) # Sending messages in MPI comm requires array
        expect_f_squared = np.array(0, dtype=np.float64)
        for p in points:
            sum_f = sum_f + (self.f(p)) # we get sum(f) for each worker
            #print("p", p)
            #print("f(p)",(self.f(p)))
            #print("sum", sum_f)
            #print()
            expect_f_squared = expect_f_squared + (self.f(p))**2 # we get sum(f**2) for each worker

        FINAL_SUM_F = sum_f
        FINAL_F_SQUARED = expect_f_squared
        
        prefactor1=1 # this will be used to create the (b-a)(d-c)...
        prefactor2=1/(num_points*(size)) # MAKE SURE THE CORRECT NUMBER OF POINTS HERE
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


def func(x):
    return x**2



def step(x):  # FUNCTION FOR CIRCLE
    return 1 * (round(x[0]**2+x[1]**2,5) <= 1)

def step(x):  # FUNCTION FOR SPHERE
    return 1 * (round(x[0]**2+x[1]**2+x[2]**2,5) <= 1)

def step(x):  # FUNCTION FOR SPHERE 5D
    return 1 * (round(x[0]**2+x[1]**2+x[2]**2+x[3]**2+x[4]**2,5) <= 1)

num_points = int(200000)
#num_points = int(5)

#AH = Monte_Carlo([2], [3], num_points, func)

sphere2 =Monte_Carlo ([-1,-1,-1,-1,-1], [1,1,1,1,1], num_points, step)
if rank ==0:
    print(f"5D Sphere with x -1 to 1: {sphere2}")

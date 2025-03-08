#!/bin/python3
"""
Created on Fri Feb 21 18:55:08 2025

@author: skyed
"""

from numpy.random import SeedSequence, default_rng
import numpy as np
from mpi4py import MPI

# MPI.Init()
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nworkers = comm.Get_size()


class Monte_Carlo:
    def __init__(self, starts, ends, N, func2, *args):
        self.starts = starts
        self.ends = ends
        self.f = func2
        self.N = N
        self.variables = args # if args are passed, assumed to be a variable array for function
        self.data = 0   # data to be returned for any method

    def __str__(self):
        return f"(Integral: {self.data[0]}, Var: {self.data[1]}, Err: {self.data[2]})"

    def integral(self):
        dim = len(self.starts)
        ss = SeedSequence(12345)  # getting the random numbers
        child_seeds = ss.spawn(nworkers)  # random numbers for each worker
        streams = [default_rng(s) for s in child_seeds]
        # getting the random numbers in arrays we like
        points = streams[rank].random((self.N, dim))

        # Sending messages in MPI comm requires array
        sum_f = np.array(0, dtype=np.float64)
        expect_f_squared = np.array(0, dtype=np.float64)

        for p in points:
            count = 0
            while count < dim:
                # making sure they are in the interval we need them to be
                p[count] = p[count] * \
                    (self.ends[count]-self.starts[count])+self.starts[count]
                count = count+1
            sum_f = sum_f + (self.f(p, *self.variables))  # we get sum(f) for each worker
            expect_f_squared = expect_f_squared + \
                (self.f(p, *self.variables))**2  # we get sum(f**2) for each worker

        FINAL_SUM_F = np.array(0, dtype=np.float64)
        FINAL_F_SQUARED = np.array(0, dtype=np.float64)

        comm.Allreduce(sum_f, FINAL_SUM_F)  # These take value of sum_f for all ranks and sum them into FINAL_...
        comm.Allreduce(expect_f_squared, FINAL_F_SQUARED)

        prefactor1 = 1  # this will be used to create the (b-a)(d-c)...
        prefactor2 = 1/(num_points*nworkers)
        d = 0
        while d < dim:
            # we get our (b-a)(c-d...)
            prefactor1 = prefactor1*(self.ends[d]-self.starts[d])
            d = d+1

        FINAL_I = prefactor1*prefactor2*FINAL_SUM_F  # our integral
        FINAL_VAR = prefactor2 * \
            (FINAL_F_SQUARED*prefactor2-(FINAL_SUM_F*prefactor2)**2)  # our variance
        FINAL_ERROR = prefactor1*np.sqrt(FINAL_VAR)  # our error
        self.data = np.array([FINAL_I, FINAL_VAR, FINAL_ERROR])

        return self.data
    
    def infinity(self):
        '''
        This is used for improper/infinite cases

        '''
        dim = len(self.starts)
        inf_starts = -1  # gross way of doing this tbh
        inf_ends = 1
        ss = SeedSequence(12345)  # getting the random numbers
        child_seeds = ss.spawn(nworkers)  # random numbers for each worker
        streams = [default_rng(s) for s in child_seeds]
        # getting the random numbers in arrays we like
        points = streams[rank].random((self.N, dim))

        # Sending messages in MPI comm requires array
        sum_f = np.array(0, dtype=np.float64)
        expect_f_squared =  np.array(0, dtype=np.float64)

        for p in points:
            count = 0
            while count < dim:
                # making sure they are in the interval we need them to be
                p[count] = p[count] * (inf_ends-inf_starts)+inf_starts
                count = count+1
            # NOTE: x and factor need to be changed to handle multiple input too
            x = p/(1-p**2)  # value to be passed in to f(x)
            factor = (1+p**2)/((1-p**2)**2)  # integral factor
            sum_f = sum_f + (self.f(x, *self.variables))*factor  # we get sum(f) for each worker
            expect_f_squared = expect_f_squared + \
                (self.f(x, *self.variables) * factor)**2  # we get sum(f**2) for each worker

        FINAL_SUM_F = np.array(0, dtype=np.float64)
        FINAL_F_SQUARED = np.array(0, dtype=np.float64)

        comm.Allreduce(sum_f, FINAL_SUM_F)  # These take value of sum_f for all ranks and sum them into FINAL_...
        comm.Allreduce(expect_f_squared, FINAL_F_SQUARED)

        prefactor1 = 1  # this will be used to create the (b-a)(d-c)...
        prefactor2 = 1/(num_points*nworkers)
        d = 0
        while d < dim:
            # we get our (b-a)(c-d...)
            prefactor1 = prefactor1*(self.ends[d]-self.starts[d])
            d = d+1

        FINAL_I = prefactor1*prefactor2*FINAL_SUM_F  # our integral
        FINAL_VAR = prefactor2 * \
            (FINAL_F_SQUARED*prefactor2-(FINAL_SUM_F*prefactor2)**2)  # our variance
        FINAL_ERROR = prefactor1*np.sqrt(FINAL_VAR)  # our error
        self.data = np.array([FINAL_I, FINAL_VAR, FINAL_ERROR])

        return self.data
        

def step(x, R):  # FUNCTION FOR ANY ROUND SHAPE
    return 1 * (round(np.sum(np.square(x)), 5) <= (R**2))

def test(x, a, b):
    return a*x**2 + b

# IDEA - when calling with infinite limits, use different method (Monte_Carlo.inf)
# Method changes limits to t=-1 and t=1, and passes in t/(1-t^2) to given func 
# and multiplies each sum_f and f^2 by (1+t^2)/(1-t^2)^2
# so that we don't have to put this into each individual function if infinite

# I think it might also be a good idea to have starts/ends be optional input
# so then we don't have to arbitrarily put in start/end values for infinite

def gaussian(x, x0, sig):
    return 1/(sig*np.sqrt(2*np.pi)) * np.exp((-(x-x0)**2)/(2*sig**2))

def gaussian_multiD(x, x0, sig):
    sigma = np.sqrt(np.sum(np.square(sig))) # Norm sigma value
    # print("Sigma =", sigma)
    xx0 = np.sqrt(np.sum(np.square(x-x0))) # Norm mean value
    # print("Norm of x-x0:", xx0)
    return 1/(sigma*np.sqrt(2*np.pi)) * np.exp(-xx0**2/(2*sigma**2))


num_points = int(10000)

a = np.array([3])
b = np.array([6])

vari = np.array([1, 2])


# FIGURE OUT BETTER WAY TO CALL THIS pls
test_x_square = Monte_Carlo(a, b, num_points, test, *vari)
integral = Monte_Carlo.integral(test_x_square)

if rank == 0:
    print(f"Evaluating integral of x^2 between {a} and {b}: {test_x_square}")

radius = np.array([1])
radius2 = np.array([3])

dimensions = 5

a = np.repeat(-radius, dimensions)
b = np.repeat(radius, dimensions)

a2 = np.repeat(-radius2, dimensions)
b2 = np.repeat(radius2, dimensions)

sphere_in = Monte_Carlo(a, b, num_points, step, *radius)
sphere2_in = Monte_Carlo(a2, b2, num_points, step, *radius2)

sphere = Monte_Carlo.integral(sphere_in)
sphere2 = Monte_Carlo.integral(sphere2_in)

real = 8/15*np.pi**2*radius[0]**dimensions
real2 = 8/15*np.pi**2*radius2[0]**dimensions

if rank == 0:
    print(f"5D Sphere with radius {radius[0]}: {sphere}")
    print(f"Real value: {real}")
    print(f"5D Sphere with radius {radius2[0]}: {sphere2}")
    print(f"Real value: {real2}")
    

# TESTING 1D GAUSSIAN (it doesn't work)

mean = 4
sigma = 0.4

vari2 = np.array([mean, sigma])
gaussTest = Monte_Carlo([-5], [5], num_points, gaussian, *vari2)
# same output for gaussian_multiD
gaussOutput = Monte_Carlo.infinity(gaussTest)

print(gaussOutput)

MPI.Finalize()

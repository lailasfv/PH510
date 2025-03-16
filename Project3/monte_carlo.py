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
    def __init__(self, starts, ends, N, func2, variables=[]):
        self.starts = starts
        self.ends = ends
        self.f = func2
        self.N = N
        self.variables = variables # variables defaults to an empty array if none are supplied
        self.data = 0   # data to be returned for any method

    def __str__(self):
        return (f"(Integral: {self.data[0,0]:.4f}, Var: {self.data[1,0]:.4f},"
                f" Err: {self.data[2,0]:.4f})")
    
    def random(seed):
        ss = SeedSequence(seed)  # getting the random numbers
        child_seeds = ss.spawn(nworkers)  # random numbers for each worker
        streams = [default_rng(s) for s in child_seeds]
        # getting the random numbers in arrays we like
        return streams

    def integral(self, seed):
        dim = len(self.starts)
        streams = Monte_Carlo.random(seed)
        points = streams[rank].random((int(self.N/nworkers), dim))

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
        prefactor2 = 1/(num_points)
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
    
    def infinity(self, seed):
        '''
        This is used for improper/infinite cases

        '''
        dim = len(self.starts)
        inf_starts = -1  # gross way of doing this tbh
        inf_ends = 1
        streams = Monte_Carlo.random(seed)
        points = streams[rank].random((int(self.N/nworkers), dim))

        # Sending messages in MPI comm requires array
        sum_f = np.array(0, dtype=np.float64)
        expect_f_squared =  np.array(0, dtype=np.float64)

        for p in points:
            count = 0
            while count < dim:
                # making sure they are in the interval we need them to be
                p[count] = p[count] * (inf_ends-inf_starts)+inf_starts
                count = count+1

            x = p/(1-p**2)  # value to be passed in to f(x)
            factor = (1+p**2)/((1-p**2)**2)  # integral factor
            sum_f = sum_f + (self.f(x, *self.variables, factor))  # we get sum(f) for each worker
            expect_f_squared = expect_f_squared + \
                self.f(x, *self.variables, factor**2)  # we get sum(f**2) for each worker

        FINAL_SUM_F = np.empty(dim, dtype=np.float64)
        FINAL_F_SQUARED = np.empty(dim, dtype=np.float64)

        comm.Allreduce(sum_f, FINAL_SUM_F)  # These take value of sum_f for all ranks and sum them into FINAL_...
        comm.Allreduce(expect_f_squared, FINAL_F_SQUARED)

        prefactor1 = 1  # this will be used to create the (b-a)(d-c)...
        prefactor2 = 1/(num_points)
        d = 0
        while d < dim:
            # we get our (b-a)(c-d...)
            prefactor1 = prefactor1*(inf_ends-inf_starts)
            d = d+1

        FINAL_I = prefactor1*prefactor2*FINAL_SUM_F  # our integral
        FINAL_VAR = prefactor2 * \
            (FINAL_F_SQUARED*prefactor2-(FINAL_SUM_F*prefactor2)**2)  # our variance
        FINAL_ERROR = prefactor1*np.sqrt(FINAL_VAR)  # our error
        self.data = np.array([FINAL_I, FINAL_VAR, FINAL_ERROR])

        return self.data
    
    def integral_importance_sampling(self, inverse_samp, seed):
        dim = len(self.starts)
        streams = Monte_Carlo.random(seed)
        points = streams[rank].random((int(self.N/nworkers), dim))

        # Sending messages in MPI comm requires array
        sum_f = np.array(0, dtype=np.float64)
        expect_f_squared = np.array(0, dtype=np.float64)


        for p in points:
            count = 0
            while count < dim:
                p[count] = inverse_samp(p[count])
                count = count+1
            sum_f = sum_f + (self.f(p, *self.variables))  # we get sum(f) for each worker
            expect_f_squared = expect_f_squared + \
                (self.f(p, *self.variables))**2  # we get sum(f**2) for each worker
        
        FINAL_SUM_F = np.empty(dim, dtype=np.float64)
        FINAL_F_SQUARED = np.empty(dim, dtype=np.float64)

        comm.Allreduce(sum_f, FINAL_SUM_F)  # These take value of sum_f for all ranks and sum them into FINAL_...
        comm.Allreduce(expect_f_squared, FINAL_F_SQUARED)
        
        prefactor2 = 1/(num_points)
        FINAL_I = FINAL_SUM_F*prefactor2 # our integral
        FINAL_VAR = prefactor2 * \
            (FINAL_F_SQUARED*prefactor2-(FINAL_SUM_F*prefactor2)**2)  # our variance
        FINAL_ERROR = np.sqrt(FINAL_VAR)  # our error
        self.data = np.array([FINAL_I, FINAL_VAR, FINAL_ERROR])

        return self.data

def funct2_try(x):
    return 2**x

def samp2(x):
    A=1/(1-np.exp(-4))
    return(A*np.exp(x))

def func_imp_2(x):
    return (funct2_try(x)/samp2(x))

def inverse_samp2(x):
    A=1/(1-np.exp(-4))
    #A=1
    return(np.log(x/A+np.exp(-4)))
"""
funct_test = Monte_Carlo([0],[2],num_points,func_w_imp_sampling)
#funct_test = Monte_Carlo([0],[1],num_points,funct_without_imp_sampling)
integral= Monte_Carlo.integral_importance_sampling(funct_test, inverse_samp)
print(integral)
"""
num_points = int(100000)
seed2 = 12345
funct_test2 = Monte_Carlo([-4],[0],num_points,func_imp_2)
integral2= Monte_Carlo.integral_importance_sampling(funct_test2, inverse_samp2,seed2)
print(integral2)
        

def step(x, R):  # FUNCTION FOR ANY ROUND SHAPE
    return round(np.sum(np.square(x)), 5) <= (R**2)

def test(x, a, b):
    return a*x**2 + b

def gaussian1D(x, x0, sig, factor):
    return np.linalg.norm(factor/(sig*(2*np.pi)**0.5) * np.exp((-(x-x0)**2)/(2*sig**2)))

def gaussianMD(x, x0, sig, factor):
    power = len(x)/2
    return np.linalg.norm(factor/(sig*2*(2*np.pi)**power) * np.exp((-(x-x0)**2)/(2*sig**2)))


num_points = int(100000)
seed = 12345

a = np.array([3])
b = np.array([6])

vari = np.array([1, 2])

test_x_square = Monte_Carlo(a, b, num_points, test, variables=vari)
integral = Monte_Carlo.integral(test_x_square, seed)

if rank == 0:
    print(f"Evaluating integral of x^2 between {a} and {b}: {test_x_square}")

radius = np.array([1])
radius2 = np.array([3])

dimensions = 5

a = np.repeat(-radius, dimensions)
b = np.repeat(radius, dimensions)

a2 = np.repeat(-radius2, dimensions)
b2 = np.repeat(radius2, dimensions)

sphere_in = Monte_Carlo(a, b, num_points, step, variables=radius)
sphere2_in = Monte_Carlo(a2, b2, num_points, step, variables=radius2)

sphere = Monte_Carlo.integral(sphere_in, seed)
sphere2 = Monte_Carlo.integral(sphere2_in, seed)

real = 8/15*np.pi**2*radius[0]**dimensions
real2 = 8/15*np.pi**2*radius2[0]**dimensions

if rank == 0:
    print(f"5D Sphere with radius {radius[0]}: {sphere}")
    print(f"Real value: {real}")
    print(f"5D Sphere with radius {radius2[0]}: {sphere2}")
    print(f"Real value: {real2}")
    

# TESTING GAUSSIAN 

mean = 4
sigma = 0.4

vari2 = np.array([mean, sigma])
gaussTest = Monte_Carlo([-5], [5], num_points, gaussian1D, variables=vari2)
gaussOutput = Monte_Carlo.infinity(gaussTest, seed)

mean2 = np.array([2, 5, 6, 9, 4, 2])
sigma2 = np.array([0.2, 0.4, 0.1, 0.3, 0.2, 0.5])

vari2b = np.array([mean2, sigma2])

gaussTest2 = Monte_Carlo([-1, -1, -1, -1, -1, -1], [1, 1, 1, 1, 1, 1], num_points, gaussianMD, variables=vari2b)
gaussOutput2 = Monte_Carlo.infinity(gaussTest2, seed)

if rank == 0:
    print(gaussOutput)
    print(gaussOutput2)

MPI.Finalize()

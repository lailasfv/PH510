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
        prefactor2 = 1/(self.N)
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
        prefactor2 = 1/(self.N)
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
        
        prefactor2 = 1/(self.N)
        FINAL_I = FINAL_SUM_F*prefactor2 # our integral
        FINAL_VAR = prefactor2 * \
            (FINAL_F_SQUARED*prefactor2-(FINAL_SUM_F*prefactor2)**2)  # our variance
        FINAL_ERROR = np.sqrt(FINAL_VAR)  # our error
        self.data = np.array([FINAL_I, FINAL_VAR, FINAL_ERROR])

        return self.data

MPI.Finalize()

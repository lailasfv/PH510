#!/bin/python3

import numpy as np

class relaxation_Poisson:
    def __init__(self,N,h,f,conditions,max_change_criteria, iteration_limit):
        #print("AHAHAHA",N)
        self.N=int(N)
        self.h=h
        self.f = f # this is an array of arrays [x,y,f(x,y)] where f(x,y)
        # is not equal to 0. Pass it an empty array for the laplace case
        self.max_change_criteria=max_change_criteria
        self.iteration_limit=iteration_limit
        """
        array_sub=np.linspace(0,(self.h*(self.N-1)),int(self.N)) # CHECK THIS 
        TWO_D_array=np.zeros([self.N,self.N,2])
        print(array_)
        i = 0
        j = 0
        while i<N:
            j=0
            while j<N:
                TWO_D_array[i,j,0]=array_sub[i]
                TWO_D_array[i,j,1]=array_sub[j]
                j=j+1
            i=i+1
        # Here we get a 3D array, or how I want to think about it:
        # you get a 2D grid going 
        # [0,0], [0,h] ,[0,2h],..,[0,h*(N-1)]
        # [h,0], [h,h] ,[h,2h],..,[h,h*(N-1)]
        # [2h,0], [2h,h] ,[2h,2h],..,[2h,h*(N-1)]
        # ...
        # [h*(N-1),0], [h*(N-1),h] ,[h*(N-1),2h],..,[h*(N-1),h*(N-1)]
        # so you can call TWO_D_array[i,j] and know the position of it in cm
        # IDK if that is actually useful now that I think about it, i might
        # get rid of this
        """
        self.grid = np.zeros([self.N,self.N])
        #print(self.N, "N")
        #print(self.grid, "AH")
        # if we have boundary conditions [[x,y, f],...] AN ARRAY
        for cond in conditions:
            axis1=int(cond[0]/h)
            axis2=int(cond[1]/h)
            self.grid[axis1,axis2]=cond[2]
        self.grid_f = np.zeros([self.N,self.N])
        for values in self.f:
            axis1=int(values[0]/h)
            axis2=int(values[1]/h)
            self.grid_f[axis1,axis2]=values[2]
        #print(self.grid_f, "HELLO")
        
        k = 1 
        while k<N-1:
            m = 1
            while m<N-1:
                self.grid[k,m]=np.random.rand(1)[0]
                m=m+1
            k=k+1
        #print(self.grid)
        
        t = 0
        runtime_check= 10000
        
        while t<self.iteration_limit:
            #print("HELLO EVERYONE")
            check = np.zeros([self.N,self.N])
            x=0
            while x<self.N:
                y=0
                while y<self.N:
                    check[x,y]=self.grid[x,y]
                    y=y+1 
                x=x+1
            k2=1
            while k2<N-1:
                m2 = 1
                while m2<N-1:
                    self.grid[k2,m2]=1/4*(self.grid[k2+1,m2]+self.grid[k2-1,m2]+self.grid[k2,m2+1]+self.grid[k2,m2-1]+self.h**2*self.grid_f[k2,m2]) # I CHANGED THIS HERE TO TRY GO GET POISSON
                    m2=m2+1
                k2=k2+1
            checking_array = abs((check-self.grid)/check)
            max_change = 0
            #print("LOOK HERE", checking_array)
            for line in checking_array:
                if max(line)>max_change:
                    max_change=max(line)*100
            
            if max_change<self.max_change_criteria:
                print("We found the result:")
                print(self.grid)
                print("in", t+1, "iterations")
                t=self.iteration_limit
            elif t==(self.iteration_limit-1):
                print("we ran out of time!")
            t=t+1



h1= 0.5
N1 = 4
"""
(0,0)   (0,0.5)   (0,1.0)
(0.5,0)   (0.5,0.5)   (0.5,1.0)
(1,0)   (1,0.5)   (1,1.0)

"""
PLS = np.array([[0,0,2], [0,0.5,2],[0,1,2],[0,1.5,2],
                [0.5,0,2],[0.5,1.5,2],
                [1,0,2],[1,1.5,2],
                [1.5,0,2],[1.5,0.5,2],[1.5,1,2],[1.5,1.5,2]])
f_trying_smth = np.array([[0.5,0.5,1.1]])
TEST=relaxation_Poisson(N1,h1,f_trying_smth,PLS,10**(-10),2000)

one = np.array([[1,2,3],[4,5,6],[7,8,9]])
two = np.array([[10, 9, 6],[7,2,4],[3,5,1]])

# -*- coding: utf-8 -*-
"""
Created on Sat May 10 20:29:54 2025

@author: skyed
"""

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
            grid2=np.zeros([self.N,self.N])
            index10=0
            while index10<self.N:
                index11 = 0
                while index11<self.N:
                    grid2[index10,index11]=self.grid[index10,index11]
                    index11=index11+1
                index10=index10+1
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
                    grid2[k2,m2]=1/4*(self.grid[k2+1,m2]+self.grid[k2-1,m2]+self.grid[k2,m2+1]+self.grid[k2,m2-1]+self.h**2*self.grid_f[k2,m2]) # I CHANGED THIS HERE TO TRY GO GET POISSON
                    m2=m2+1
                k2=k2+1
            
            checking_array= np.zeros([self.N,self.N])
            index1 = 0
            while index1<self.N:
                index2=0
                while index2<self.N:
                    if check[index1,index2]!=0:
                        checking_array[index1,index2] = abs((check[index1,index2]-grid2[index1,index2])/check[index1,index2])
                    index2=index2+1 
                index1=index1+1
            
            
            max_change = 0
            #print("LOOK HERE", checking_array)
            for line in checking_array:
                if max(line)>max_change:
                    max_change=max(line)*100
            
            if max_change<self.max_change_criteria:
                #print("We found the result:")
                #print(np.round(self.grid,5))
                print("in", t+1, "iterations")
                t=self.iteration_limit
                print("point (5,5)cm", self.grid[50,50])
                print("point (2.5,2.5)cm", self.grid[25,25])
                print("point (0.1,2.5)cm",self.grid[1,25])
                print("point (0.1,0.1cm)",self.grid[1,1])
            elif t==(self.iteration_limit-1):
                print("we ran out of time!")
                print("point (5,5)cm", self.grid[50,50])
                print("point (2.5,2.5)cm", self.grid[25,25])
                print("point (0.1,2.5)cm",self.grid[1,25])
                print("point (0.1,0.1cm)",self.grid[1,1])
            t=t+1
            
            #print("phi")
            #print(self.grid)
            #print("phi'")
            #print(grid2)
            
            self.grid=grid2
            
            

# TASK 5 (/4)
# we want side length of 10cm, so 
# if we have h=1cm, then N=11
# if we have N=101 then h=0.1cm
length2=10 / 100 # m
N2=101
#N2 = 5
h2=length2/(N2-1) #m
# f=0 everywhere initially???

def boundary_conditions(N,h,top,bottom,left,right):
    conditions = []
    index2 = 0
    while index2<N:
        conditions.append([0,index2*h,top])
        conditions.append([(N-1)*h, index2*h, bottom])
        if 0<index2<(N2-1):
            conditions.append([index2*h,0,left])
            conditions.append([index2*h,(N-1)*h,right])
        index2=index2+1
    return(np.array(conditions))
f2 = np.array([])

print("Task 4a)")
conds2_a = boundary_conditions(N2, h2, 1, 1, 1, 1)
task4_relaxation_a = relaxation_Poisson(N2,h2,f2,conds2_a,10**(-3),20000)

print("Task 4b)")
conds2_b = boundary_conditions(N2,h2, 1, 1, -1, -1)
task4_relaxation_b = relaxation_Poisson(N2,h2,f2,conds2_b,10**(-3),20000)

print("Task 4c)")
conds2_c = boundary_conditions(N2,h2, 2, 0, 2, -4)
task4_relaxation_c = relaxation_Poisson(N2,h2,f2,conds2_c,10**(-3),20000)

index3_1=1

charge_each_point_1 = 10 / (N2-2)**2
f3 = []
while index3_1<N2-1:
    index3_2 = 1
    while index3_2 <N2-1:
        f3.append([index3_1*h2,index3_2*h2,charge_each_point_1])
        index3_2=index3_2+1
    index3_1=index3_1+1
f3_array_a = np.array(f3)

print("UNIFORM GRID")
print("condition a")
task4_relaxation_a_a = relaxation_Poisson(N2, h2, f3_array_a, conds2_a, 10**(-3),20000)
print("condition b")
task4_relaxation_b_a = relaxation_Poisson(N2, h2, f3_array_a, conds2_b, 10**(-3),20000)
print("condition c")
task4_relaxation_c_a = relaxation_Poisson(N2, h2, f3_array_a, conds2_c, 10**(-3),20000)

print("GRADIENT")
charge_gradient = np.linspace(1, 0, N2-2) # creates the correct gradient scale over the grid
f4 = []
index4_1=1
while index4_1<(N2-1):
    index4_2=1 
    while index4_2<(N2-1):
        f4.append([index4_1*h2,index4_2*h2,charge_gradient[index4_1-1]])
        index4_2=index4_2+1 
    index4_1=index4_1+1
f4_array_b = np.array(f4)
print("condition a")
task4_relaxation_a_b = relaxation_Poisson(N2, h2, f4_array_b, conds2_a, 10**(-3),20000)
print("condition b")
task4_relaxation_b_b = relaxation_Poisson(N2, h2, f4_array_b, conds2_b, 10**(-3),20000)
print("condition c")
task4_relaxation_c_b = relaxation_Poisson(N2, h2, f4_array_b, conds2_c, 10**(-3),20000)

print("DECAY")
centre = (N2-1)/2 # works best for odd N
f5=[]
index5_1 =1 
while index5_1<N2-1:
    index5_2 = 1 
    while index5_2<N2-1:
        r = np.sqrt(((index5_1 - centre)*h2)**2 + ((index5_2 - centre)*h2)**2)
        f5.append([index5_1*h2,index5_2*h2,np.exp(-2000*r)])
        index5_2=index5_2+1 
    index5_1=index5_1+1 
f5_array_c = np.array(f5)

print("condition a")
task4_relaxation_a_c = relaxation_Poisson(N2, h2, f5_array_c, conds2_a, 10**(-3),20000)
print("condition b")
task4_relaxation_b_c = relaxation_Poisson(N2, h2, f5_array_c, conds2_b, 10**(-3),20000)
print("condition c")
task4_relaxation_c_c = relaxation_Poisson(N2, h2, f5_array_c, conds2_c, 10**(-3),20000)

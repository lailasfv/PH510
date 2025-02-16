# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 21:07:28 2025

@author: skyed
"""

import numpy as np

# Task 1
# ADD THAT WE SPECIFY THAT STUFF IS RADIANS CAUSE WE ARE FANCY

class Vector:
    """
    Vector class for three dimensional quantities in cartesian coordinates
    """
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __str__(self):
        """
        Assumes floating point when printing
        """
        return f"({self.x:6f},{self.y:6f},{self.z:6f})" 

    def norm(self):
        """
        Returns magnitude of given instance
        """
        return np.sqrt(self.x**2+self.y**2+self.z**2)

    def __add__(self, other):
        """
        Returns addition of two instance vectors as instance vector
        """
        return Vector(self.x+other.x, self.y+other.y, self.z+other.z)

    def __sub__(self, other):
        """
        Returns subtraction of two instance vectors as instance vector
        """
        return Vector(self.x-other.x, self.y-other.y, self.z-other.z)

    def __dot__(self, other):
        """
        Returns scalar product of two instance vectors as value
        """
        return self.x*other.x+self.y*other.y+self.z*other.z
        # Vector.__dot__(v1,v2)

    def __cross__(self, other):
        """
        Returns vector product of two instance vectors as instance vector
        """
        return Vector((self.y*other.z-self.z*other.y),
                      (self.z*other.x-self.x*other.z),
                      (self.x*other.y-self.y*other.x))
    def __area__(self, other):
        """"LOOK HERE"""
        return (Vector.__cross__(self, other)).norm()/2
    def __areaVertices__(self, other,other2):
        """"LOOK HERE"""
        v_1 = other - self
        v_2 = other2 - self
        return (Vector.__cross__(v_1, v_2)).norm()/2

# Task 2


class Vector_spherical(Vector):
    """
    Vector class for three dimensional quantities in spherical coordinates.
    Input instance for theta and phi is taken to be in RADIAN units 
    """
    def __init__(self, rho, theta, phi):
        """
        Initialises instance with cartesian vector components
        """
        self.x = rho*np.sin(theta)*np.cos(phi)  # this is in radians
        self.y = rho*np.sin(theta)*np.sin(phi)
        self.z = rho*np.cos(theta)

    def __str__(self):
        rho = np.sqrt((self.x)**2+(self.y)**2+(self.z)**2)
        # theta = np.arctan(self.y/self.x)  # HERE WE MIGHT GET THE ISSUE OF
        # theta = np.arctan2(self.y,self.x)
        theta=np.arccos(self.z/np.sqrt(self.x**2+self.y**2+self.z**2))
        # THE TAN NOT SELECTING THE CORRECT ANGLE, FIND A WORK AROUND
        # arctan2 -> takes 2 values and figures that out issue
        # phi = np.arccos(self.z/np.sqrt((self.x)**2+(self.y)**2+(self.z)**2))
        phi = np.arctan2(self.y,self.x)
        return f"({rho},{theta},{phi})"


# you can use to test:
# v1 = Vector(3., 1.22, -0.5)
# v2 = Vector(5, -2, 10)
v3_sph = Vector_spherical(2, np.pi/2, np.pi/6)
print(f"Spherical vector test: {v3_sph}")

# Task 3
# (a)
def area_triangle(v1, v2):
    # SEE IF WE PUT THIS IN CLASS
    # WE COULD PUT THAT IN THE CLASS YEAH
    return (Vector.__cross__(v1, v2)).norm()/2

def internal_angle(v1, v2):
    # a.b/|a||b|
	return np.arccos((Vector.__dot__(v1, v2))/(Vector.norm(v1)*Vector.norm(v2)))



v1a = Vector(0, 0, 0)
v1b = Vector(1, 0, 0)
v1c = Vector(0, 1, 0)

v1a_sph = Vector_spherical(0, 0, 0)
v1b_sph = Vector_spherical(1, 0, 0)
v1c_sph = Vector_spherical(1, np.pi/2, 0)

area_1_cart = area_triangle(Vector.__sub__(v1b, v1a), Vector.__sub__(v1c, v1a))
area_1_sph = area_triangle(v1b_sph-v1a_sph, v1c_sph - v1a_sph) # THIS DOES NOT WORK - the child class cannot seem to use the parent class methods that are used in the area_triangle function
print(f"Triangle 1 area (cartesian) :{area_1_cart}")
print(f"Triangle 1 area (spherical) :{area_1_sph}")
angle_1_cart = np.array([internal_angle(Vector.__sub__(v1b, v1a), Vector.__sub__(v1c, v1a)), 
        internal_angle(Vector.__sub__(v1c, v1a), Vector.__sub__(v1c, v1b)), 
        internal_angle(Vector.__sub__(v1c, v1b), Vector.__sub__(v1b, v1a))]) * 180/np.pi
print(f"Triangle 1 angles:{angle_1_cart}")


# REMOVE THIS AFTER
# v2a = (-1, -1, -1)
# v2b = (0, -1, -1)
# v2c = (-1, 0, -1)
v2_a_cart = Vector(0-1, -1+1, -1+1) #b-a
v2_b_cart = Vector(-1+1, -1, -1+1) #c-a
v2_c_cart = Vector(-1-0, 0+1, -1+1) #c-b
area_2_cart = area_triangle(v2_a_cart, v2_b_cart)
print(f"Triangle 2 area:{area_2_cart}")
angle_2_cart = np.array([internal_angle(v2_a_cart, v2_b_cart), 
        internal_angle(v2_a_cart, v2_c_cart), internal_angle(v2_b_cart, v2_c_cart)]) * 180/np.pi
print(f"Triangle 2 angles:{angle_2_cart}")

# REMOVE THIS AFTER
# v3a = (1, 0, 0)
# v3b = (0, 0, 1)
# v3c = (0, 0, 0)
v3_a_cart = Vector(-1, 0, 1) #b-a
v3_b_cart = Vector(-1, 0, 0) #c-a
v3_c_cart = Vector(0, 0, -1) #c-b
area_3_cart = area_triangle(v3_a_cart, v3_b_cart)
print(f"Triangle 3 area:{area_3_cart}")
angle_3_cart = np.array([internal_angle(v3_a_cart, v3_b_cart), 
        internal_angle(v3_a_cart, v3_c_cart), internal_angle(v3_b_cart, v3_c_cart)]) * 180/np.pi
print(f"Triangle 3 angles:{angle_3_cart}")


# REMOVE THIS AFTER
# v4a = (0, 0, 0)
# v4b = (1, -1, 0)
# v4c = (0, 0, 1)
v4_a_cart = Vector(1, -1, 0) #b-a
v4_b_cart = Vector(0, 0, 1) #c-a
v4_c_cart = Vector(-1, 1, 1) #c-b
area_4_cart = area_triangle(v4_a_cart, v4_b_cart)
print(f"Triangle 4 area:{area_4_cart}")
angle_4_cart = np.array([internal_angle(v4_a_cart, v4_b_cart), 
        internal_angle(v4_a_cart, v4_c_cart), internal_angle(v4_b_cart, v4_c_cart)]) * 180/np.pi
print(f"Triangle 4 angles:{angle_4_cart}")


"""LOOK HERE"""
# r, theta, phi
T1_a = Vector_spherical(0, 0, 0)
T1_b = Vector_spherical(1, 0, 0)
T1_c = Vector_spherical(1, np.pi/2, 0)
area_sph_1= Vector_spherical.__areaVertices__(T1_a,T1_b,T1_c)
print(f"Triangle 1 area (spherical): {area_sph_1}")

T2_a = Vector_spherical(1, 0, 0)
T2_b = Vector_spherical(1, np.pi/2,0)
T2_c = Vector_spherical(1, np.pi/2, np.pi)
area_sph_2 = Vector_spherical.__areaVertices__(T2_a,T2_b,T2_c)
print(f"Triangle 2 area (spherical): {area_sph_2}")

T3_a = Vector_spherical(0, 0, 0)
T3_b = Vector_spherical(2, 0, 0)
T3_c = Vector_spherical(2, np.pi/2, 0)
area_sph_3 = Vector_spherical.__areaVertices__(T3_a,T3_b,T3_c)
print(f"Triangle 3 area (spherical): {area_sph_3}")

T4_a = Vector_spherical(1, np.pi/2, 0)
T4_b = Vector_spherical(1,np.pi/2,np.pi)
T4_c = Vector_spherical(1,np.pi/2, 3*np.pi/2)
area_sph_4 = Vector_spherical.__areaVertices__(T4_a,T4_b,T4_c)
print(f"Triangle 4 area (spherical): {area_sph_4}")


"""
T1 = Vector_spherical(1, np.pi/2, 0) # HERE
T2 = Vector_spherical(1,np.pi/2,np.pi)
T3 = Vector_spherical(1,np.pi/2, 3*np.pi/2)
def area_triangle2(v1, v2):
    # SEE IF WE PUT THIS IN CLASS
    # WE COULD PUT THAT IN THE CLASS YEAH
    return (Vector_spherical.__cross__(v1, v2)).norm()/2
v_sph_a = T2-T1
v_sph_b = T3-T1
# area_sph = area_triangle2(v_sph_a, v_sph_b)
area_sph_4 = Vector_spherical.__area__(v_sph_a,v_sph_b)

T1_a = Vector_spherical(0, 0, 0)
T1_b = Vector_spherical(1, 0, 0)
T1_c = Vector_spherical(1, np.pi/2, 0)
v_sph_1_a = T1_b - T1_a
v_sph_1_b = T1_c - T1_a
area_sph_1= Vector_spherical.__area__(v_sph_1_a,v_sph_1_b)

T2_a = Vector_spherical(1, 0, 0)
T2_b = Vector_spherical(1, np.pi/2,0)
T2_c = Vector_spherical(1, np.pi/2, np.pi)
v_sph_2_a = T2_b - T2_a
v_sph_2_b = T2_c - T2_a
area_sph_2 = Vector_spherical.__area__(v_sph_2_a, v_sph_2_b)

T3_a = Vector_spherical(0, 0, 0)
T3_b = Vector_spherical(2, 0, 0)
T3_c = Vector_spherical(2, np.pi/2, 0)

v_sph_3_a = T3_b - T3_a
v_sph_3_b = T3_c - T3_a
area_sph_3 = Vector_spherical.__area__(v_sph_3_a, v_sph_3_b)

# areas sph: 1) = 0.5,  2)=1.0, 3)=2.0, 4)=1.0
"""

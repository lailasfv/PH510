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
        return (Vector.__cross__(self, other)).norm()/2

    def __angle__(self, other):
        return np.arccos((Vector.__dot__(v1, v2))/(Vector.norm(v1)*Vector.norm(v2)))



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
        self.x = rho*np.cos(phi)*np.sin(theta)  # this is in radians
        self.y = rho*np.sin(phi)*np.sin(theta)
        self.z = rho*np.cos(theta)

    def __str__(self):
        rho = np.sqrt((self.x)**2+(self.y)**2+(self.z)**2)
        phi = np.arctan(self.y/self.x)  # HERE WE MIGHT GET THE ISSUE OF
        # THE TAN NOT SELECTING THE CORRECT ANGLE, FIND A WORK AROUND
        # arctan2 -> takes 2 values and figures that out issue
        theta = np.arccos(self.z/rho)
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

def area_triangle2(v1, v2):
    # SEE IF WE PUT THIS IN CLASS
    # WE COULD PUT THAT IN THE CLASS YEAH
    return (Vector_spherical.__cross__(v1, v2)).norm()/2

def internal_angle(v1, v2):
    # a.b/|a||b|
	return np.arccos((Vector.__dot__(v1, v2))/(Vector.norm(v1)*Vector.norm(v2)))

def internal_angle2(v1, v2):
    # a.b/|a||b|
	return np.arccos((Vector_spherical.__dot__(v1, v2))/(Vector_spherical.norm(v1)*Vector_spherical.norm(v2)))

#------------------------------
# TRIANGLE 1

v1a = Vector(0, 0, 0)
v1b = Vector(1, 0, 0)
v1c = Vector(0, 1, 0)

v1a_sph = Vector_spherical(0, 0, 0)
v1b_sph = Vector_spherical(1, 0, 0)
v1c_sph = Vector_spherical(1, np.pi/2, 0)

area_1_cart = area_triangle(v1b - v1a, v1c - v1a)
area_1_sph = area_triangle2(v1b_sph-v1a_sph, v1c_sph - v1a_sph) 
print(f"Triangle 1 area (cartesian) :{area_1_cart}")
print(f"Triangle 1 area (spherical) :{area_1_sph}")
angle_1_cart = np.array([internal_angle(v1b - v1a, v1c - v1a), 
        internal_angle(v1c - v1a, v1c - v1b), 
        internal_angle(v1c - v1b, v1b-v1a)]) * 180/np.pi
angle_1_sph = np.array([internal_angle2(v1b_sph - v1a_sph, v1c_sph - v1a_sph), 
        internal_angle(v1c_sph - v1a_sph, v1c_sph - v1b_sph), 
        internal_angle(v1c_sph - v1b_sph, v1b_sph-v1a_sph)]) * 180/np.pi
print(f"Triangle 1 angles:{angle_1_cart}")
print(f"Triangle 1 angles (spherical):{angle_1_sph}")

#------------------------------
# TRIANGLE 2

v2a = Vector(-1, -1, -1)
v2b = Vector(0, -1, -1)
v2c = Vector(-1, 0, -1)

v2a_sph = Vector_spherical(1, 0, 0)
v2b_sph = Vector_spherical(1, np.pi/2, 0)
v2c_sph = Vector_spherical(1, np.pi/2, np.pi)

area_2_cart = area_triangle(v2b - v2a, v2c - v2a)
area_2_sph = area_triangle2(v2b_sph - v2a_sph, v2c_sph - v2a_sph) 
print(f"Triangle 2 area (cartesian) :{area_2_cart}")
print(f"Triangle 2 area (spherical) :{area_2_sph}")
angle_2_cart = np.array([internal_angle(v2b - v2a, v2c - v2a),
        internal_angle(v2c - v2a, v2c - v2b),
        internal_angle(v2c - v2b, v2b-v2a)]) * 180/np.pi
angle_2_sph = np.array([internal_angle2(v2b_sph - v2a_sph, v2c_sph - v2a_sph), 
        internal_angle(v2c_sph - v2a_sph, v2c_sph - v2b_sph), 
        internal_angle(v2c_sph - v2b_sph, v2b_sph-v2a_sph)]) * 180/np.pi
print(f"Triangle 2 angles (cartesian):{angle_2_cart}")
print(f"Triangle 2 angles (spherical):{angle_2_sph}")

#------------------------------
# TRIANGLE 3

v3a = Vector(1, 0, 0)
v3b = Vector(0, 0, 1)
v3c = Vector(0, 0, 0)

v3a_sph = Vector_spherical(0, 0, 0)
v3b_sph = Vector_spherical(2, 0, 0)
v3c_sph = Vector_spherical(2, np.pi/2, 0)

area_3_cart = area_triangle(v3b - v3a, v3c - v3a)
area_3_sph = area_triangle2(v3b_sph - v3a_sph, v3c_sph - v3a_sph) 
print(f"Triangle 3 area (cartesian) :{area_3_cart}")
print(f"Triangle 3 area (spherical) :{area_3_sph}")
angle_3_cart = np.array([internal_angle(v3b - v3a, v3c - v3a),
        internal_angle(v3c - v3a, v3c - v3b),
        internal_angle(v3c - v3b, v3b-v3a)]) * 180/np.pi
angle_3_sph = np.array([internal_angle2(v3b_sph - v3a_sph, v3c_sph - v3a_sph), 
        internal_angle(v3c_sph - v3a_sph, v3c_sph - v3b_sph), 
        internal_angle(v3c_sph - v3b_sph, v3b_sph-v3a_sph)]) * 180/np.pi
print(f"Triangle 3 angles (cartesian):{angle_3_cart}")
print(f"Triangle 3 angles (spherical):{angle_3_sph}")


#------------------------------
# TRIANGLE 4

v4a = Vector(0, 0, 0)
v4b = Vector(1, -1, 0)
v4c = Vector(0, 0, 1)

v4a_sph = Vector_spherical(1, np.pi/2, 0)
v4b_sph = Vector_spherical(1, np.pi/2, np.pi)
v4c_sph = Vector_spherical(1, np.pi/2, 3*np.pi/2)

area_4_cart = area_triangle(v4b - v4a, v4c - v4a)
area_4_sph = area_triangle2(v4b_sph - v4a_sph, v4c_sph - v4a_sph) 
print(f"Triangle 4 area (cartesian) :{area_4_cart}")
print(f"Triangle 4 area (spherical) :{area_4_sph}")
angle_4_cart = np.array([internal_angle(v4b - v4a, v4c - v4a),
        internal_angle(v4c - v4a, v4c - v4b),
        internal_angle(v4c - v4b, v4b-v4a)]) * 180/np.pi
angle_4_sph = np.array([internal_angle2(v4b_sph - v4a_sph, v4c_sph - v4a_sph), 
        internal_angle(v4c_sph - v4a_sph, v4c_sph - v4b_sph), 
        internal_angle(v4c_sph - v4b_sph, v4b_sph-v4a_sph)]) * 180/np.pi
print(f"Triangle 4 angles (cartesian):{angle_4_cart}")
print(f"Triangle 4 angles (spherical):{angle_4_sph}")


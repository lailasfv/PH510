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
        return np.arccos((Vector.__dot__(self, other))/(Vector.norm(self)*Vector.norm(other)))

    def __areaVertices__(self, other,other2):
        v_1 = other - self
        v_2 = other2 - self
        return (Vector.__cross__(v_1, v_2)).norm()/2

    def __angleVertices__(self, other, other2):
        v_1 = other - self
        v_2 = other2 - self
        v_3 = other2 - other
        angles = np.array([np.arccos((Vector.__dot__(v_1, v_2))/(Vector.norm(v_1)*Vector.norm(v_2))),
            np.arccos((Vector.__dot__(v_1, v_3))/(Vector.norm(v_1)*Vector.norm(v_3))),
            np.arccos((Vector.__dot__(v_2, v_3))/(Vector.norm(v_2)*Vector.norm(v_3)))])
        return angles



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
    # REMOVE THESE?
    return (Vector.__cross__(v1, v2)).norm()/2

def area_triangle2(v1, v2):
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

area_1_cart = Vector.__areaVertices__(v1a, v1b, v1c)
area_1_sph = Vector_spherical.__areaVertices__(v1a_sph, v1b_sph, v1c_sph)
print(f"Triangle 1 area (cartesian) :{area_1_cart}")
print(f"Triangle 1 area (spherical) :{area_1_sph}")

angle_1_cart = Vector.__angleVertices__(v1a, v1b, v1c)
angle_1_sph = Vector_spherical.__angleVertices__(v1a_sph, v1b_sph, v1c_sph)
print(f"Triangle 1 angles (cartesian):{angle_1_cart*180/np.pi}")
print(f"Triangle 1 angles (spherical):{angle_1_sph*180/np.pi}")

#------------------------------
# TRIANGLE 2
# THESE TRIANGLES DO NOT APPEAR TO BE THE SAME??
# also cartesian angles broke and add to 90 deg total lol

v2a = Vector(1, -1, -1)
v2b = Vector(0, -1, -1)
v2c = Vector(-1, 0, -1)

v2a_sph = Vector_spherical(1, 0, 0)
v2b_sph = Vector_spherical(1, np.pi/2, 0)
v2c_sph = Vector_spherical(1, np.pi/2, np.pi)

area_2_cart = Vector.__areaVertices__(v2a, v2b, v2c)
area_2_sph = Vector_spherical.__areaVertices__(v2a_sph, v2b_sph, v2c_sph)
print(f"Triangle 2 area (cartesian) :{area_2_cart}")
print(f"Triangle 2 area (spherical) :{area_2_sph}")

angle_2_cart = Vector.__angleVertices__(v2a, v2b, v2c)
angle_2_sph = Vector_spherical.__angleVertices__(v2a_sph, v2b_sph, v2c_sph)
print(f"Triangle 2 angles (cartesian):{angle_2_cart*180/np.pi}")
print(f"Triangle 2 angles (spherical):{angle_2_sph*180/np.pi}")

#------------------------------
# TRIANGLE 3
# DIFFERENT AREA FOR THIS ONE TOO??
# also the angles are backwards

v3a = Vector(1, 0, 0)
v3b = Vector(0, 0, 1)
v3c = Vector(0, 0, 0)

v3a_sph = Vector_spherical(0, 0, 0)
v3b_sph = Vector_spherical(2, 0, 0)
v3c_sph = Vector_spherical(2, np.pi/2, 0)

area_3_cart = Vector.__areaVertices__(v3a, v3b, v3c)
area_3_sph = Vector_spherical.__areaVertices__(v3a_sph, v3b_sph, v3c_sph)
print(f"Triangle 3 area (cartesian) :{area_3_cart}")
print(f"Triangle 3 area (spherical) :{area_3_sph}")

angle_3_cart = Vector.__angleVertices__(v3a, v3b, v3c)
angle_3_sph = Vector_spherical.__angleVertices__(v3a_sph, v3b_sph, v3c_sph)
print(f"Triangle 3 angles (cartesian):{angle_3_cart*180/np.pi}")
print(f"Triangle 3 angles (spherical):{angle_3_sph*180/np.pi}")


#------------------------------
# TRIANGLE 4
# different area and angles AGAIN

v4a = Vector(0, 0, 0)
v4b = Vector(1, -1, 0)
v4c = Vector(0, 0, 1)

v4a_sph = Vector_spherical(1, np.pi/2, 0)
v4b_sph = Vector_spherical(1, np.pi/2, np.pi)
v4c_sph = Vector_spherical(1, np.pi/2, 3*np.pi/2)

area_4_cart = Vector.__areaVertices__(v4a, v4b, v4c)
area_4_sph = Vector_spherical.__areaVertices__(v4a_sph, v4b_sph, v4c_sph)
print(f"Triangle 4 area (cartesian) :{area_4_cart}")
print(f"Triangle 4 area (spherical) :{area_4_sph}")

angle_4_cart = Vector.__angleVertices__(v4a, v4b, v4c)
angle_4_sph = Vector_spherical.__angleVertices__(v4a_sph, v4b_sph, v4c_sph)
print(f"Triangle 4 angles (cartesian):{angle_4_cart*180/np.pi}")
print(f"Triangle 4 angles (spherical):{angle_4_sph*180/np.pi}")


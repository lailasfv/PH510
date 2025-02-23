# -*- coding: utf-8 -*-
"""This module establishes a 3D vector class for use in doing simple
vector calculations; particularly with vertices of triangles."""

import numpy as np

# Task 1

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
        return f"({self.x:2f},{self.y:2f},{self.z:2f})"

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

    def area(self, other):
        """
        Given two instance vectors, returns value for area of triangle.
        """
        return (Vector.__cross__(self, other)).norm()/2

    def angle(self, other):
        """
        Given two instance vectors, returns value (in radians) for angle
        between them.
        """
        return np.arccos((Vector.__dot__(self, other)) /
                         (Vector.norm(self)*Vector.norm(other)))

    def area_vertices(self, other, other2):
        """
        Given three instance vertices, calculates vectors and returns
        value for area of triangle.
        """
        v_1 = other - self
        v_2 = other2 - self
        return (Vector.__cross__(v_1, v_2)).norm()/2

    def angle_vertices(self, other, other2):
        """
        Given three vertices of a triangle, calculates vectors and
        returns values (in radians) for all internal angles
        """
        vba = other - self
        vab = self - other
        vca = other2 - self
        vac = self - other2
        vcb = other2 - other
        vbc = other - other2
        angles = np.array([np.arccos((Vector.__dot__(vba, vca)) /
                            (Vector.norm(vba)*Vector.norm(vca))),
                           np.arccos((Vector.__dot__(vab, vcb)) /
                            (Vector.norm(vab)*Vector.norm(vcb))),
                           np.arccos((Vector.__dot__(vac, vbc)) /
                            (Vector.norm(vac)*Vector.norm(vbc)))])
        return angles



# Task 2

class VectorSpherical(Vector):
    """
    Vector class for three dimensional quantities in spherical coordinates.
    Input instance for theta and phi is taken to be in RADIAN units
    """
    def __init__(self, rho, theta, phi):
        """
        Given spherical polar coordinates (in radians), initialises instance
        with cartesian vector components
        """
        Vector.__init__(self,
                        rho*np.sin(theta) * np.cos(phi),
                        rho*np.sin(theta) * np.sin(phi),
                        rho*np.cos(theta))

    def __str__(self):
        rho = np.sqrt((self.x)**2+(self.y)**2+(self.z)**2)
        theta = np.arccos(self.z/np.sqrt(self.x**2+self.y**2+self.z**2))
        phi = np.arctan2(self.y, self.x)
        return f"({rho:6f},{theta:6f},{phi:6f})"

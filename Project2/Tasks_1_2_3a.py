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
        return f"({self.x:6f},{self.y:6f},{self.z:6f})" # REWRITE THAT WITH THE EXAMPLE

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
        self.x = rho*np.sin(phi)*np.cos(theta)  # this is in radians
        self.y = rho*np.sin(phi)*np.sin(theta)
        self.z = rho*np.cos(phi)

    def __str__(self):
        rho = np.sqrt((self.x)**2+(self.y)**2+(self.z)**2)
        theta = np.arctan(self.y/self.x)  # HERE WE MIGHT GET THE ISSUE OF
        # THE TAN NOT SELECTING THE CORRECT ANGLE, FIND A WORK AROUND
        # arctan2 -> takes 2 values and figures that out issue
        phi = np.arccos(self.z/rho)
        return f"({rho},{theta},{phi})" # REWRITE THAT WITH THE EXAMPLE


# you can use to test:
# v1 = Vector(3., 1.22, -0.5)
# v2 = Vector(5, -2, 10)
# v3_sph = Vector_spherical(2, np.pi/2, np.pi/6)

# Task 3
# (a)
def area_triangle(v1, v2):
    # SEE IF WE PUT THIS IN CLASS
    # WE COULD PUT THAT IN THE CLASS YEAH
    return (Vector.__cross__(v1, v2)).norm()/2

def internal_angle(v1, v2):
    # a.b/|a||b|
	return np.arccos(Vector.__dot__(v1, v2))/(Vector.norm(v1)*Vector.norm(v2))



# REMOVE THIS AFTER
# v1a = (0, 0, 0)
# v1b = (1, 0, 0)
# v1c = (0, 1, 0)
# What if we make class do this? Gives vector for given coordinates?
v1_a_cart = Vector(1-0, 0-0, 0-0)
v1_b_cart = Vector(0-0, 1-0, 0-0)
v1_c_cart = Vector(0-1, 1-0, 0-0)
# area_1_cart = (Vector.__cross__(v1_a_cart,v1_b_cart)).norm()/2
area_1_cart = area_triangle(v1_a_cart, v1_b_cart)
angle_1ab_cart = internal_angle(v1_a_cart, v1_b_cart)


# REMOVE THIS AFTER
# v2a = (-1, -1, -1)
# v2b = (0, -1, -1)
# v2c = (-1, 0, -1)
v2_a_cart = Vector(-1, -1+1, -1+1)
v2_b_cart = Vector(-1+1, -1, -1+1)
v2_c_cart = vector(-1-0, 0+1, -1+1)
area_2_cart = area_triangle(v2_a_cart, v2_b_cart)
angle_2ab_cart = internal_angle(v2_a_cart, v2_b_cart)

v3_a_cart = Vector(1, 0, -1)
v3_b_cart = Vector(1, 0, 0)
area_3_cart = area_triangle(v3_a_cart, v3_b_cart)
angle_3ab_cart = internal_angle(v3_a_cart, v3_b_cart)

v4_a_cart = Vector(-1, 1, 0)
v4_b_cart = Vector(0, 0, -1)
area_4_cart = area_triangle(v4_a_cart, v4_b_cart)
angle_4ab_cart = internal_angle(v4_a_cart, v4_b_cart)
angle_list = np.array([angle_1_cart, angle_2_cart, angle_3_cart, angle_4_cart])
angle_degrees = angle_list*180/np.pi
print(angle_degrees)

# -*- coding: utf-8 -*-
"""This is just testing the vector3d module
This does not need a module docstring"""

import numpy as np
from vector3d import Vector, VectorSpherical

# ------------------------------
# TRIANGLE 1 - Cartesian

v1a = Vector(0, 0, 0)
v1b = Vector(1.0, 0, 0)
v1c = Vector(0, 1.0, 0)

area_1_cart = Vector.area_vertices(v1a, v1b, v1c)
print(f"Triangle 1 area (cartesian) :{area_1_cart}")

angle_1_cart = Vector.angle_vertices(v1a, v1b, v1c)
print(f"Triangle 1 angles (cartesian):{angle_1_cart*180/np.pi}")

# ------------------------------
# TRIANGLE 2 - Cartesian

v2a = Vector(-1.0, -1.0, -1.0)
v2b = Vector(0, -1.0, -1.0)
v2c = Vector(-1.0, 0, -1.0)

area_2_cart = Vector.area_vertices(v2a, v2b, v2c)
print(f"Triangle 2 area (cartesian) :{area_2_cart}")

angle_2_cart = Vector.angle_vertices(v2a, v2b, v2c)
print(f"Triangle 2 angles (cartesian):{angle_2_cart*180/np.pi}")

# ------------------------------
# TRIANGLE 3 - Cartesian

v3a = Vector(1.0, 0, 0)
v3b = Vector(0, 0, 1.0)
v3c = Vector(0, 0, 0)

area_3_cart = Vector.area_vertices(v3a, v3b, v3c)
print(f"Triangle 3 area (cartesian) :{area_3_cart}")

angle_3_cart = Vector.angle_vertices(v3a, v3b, v3c)
print(f"Triangle 3 angles (cartesian):{angle_3_cart*180/np.pi}")


# ------------------------------
# TRIANGLE 4 - Cartesian

v4a = Vector(0, 0, 0)
v4b = Vector(1.0, -1.0, 0)
v4c = Vector(0, 0, 1.0)

area_4_cart = Vector.area_vertices(v4a, v4b, v4c)
print(f"Triangle 4 area (cartesian) :{area_4_cart}")

angle_4_cart = Vector.angle_vertices(v4a, v4b, v4c)
print(f"Triangle 4 angles (cartesian):{angle_4_cart*180/np.pi}")

# ------------------------------
# TRIANGLE 1 - Spherical

v1a_sph = VectorSpherical(0, 0, 0)
v1b_sph = VectorSpherical(1.0, 0, 0)
v1c_sph = VectorSpherical(1.0, np.pi/2, 0)

area_1_sph = VectorSpherical.area_vertices(v1a_sph, v1b_sph, v1c_sph)
print(f"Triangle 1 area (spherical) :{area_1_sph}")

angle_1_sph = VectorSpherical.angle_vertices(v1a_sph, v1b_sph, v1c_sph)
print(f"Triangle 1 angles (spherical):{angle_1_sph*180/np.pi}")

# ------------------------------
# TRIANGLE 2 - Spherical

v2a_sph = VectorSpherical(1.0, 0, 0)
v2b_sph = VectorSpherical(1.0, np.pi/2, 0)
v2c_sph = VectorSpherical(1.0, np.pi/2, np.pi)

area_2_sph = VectorSpherical.area_vertices(v2a_sph, v2b_sph, v2c_sph)
print(f"Triangle 2 area (spherical) :{area_2_sph}")

angle_2_sph = VectorSpherical.angle_vertices(v2a_sph, v2b_sph, v2c_sph)
print(f"Triangle 2 angles (spherical):{angle_2_sph*180/np.pi}")

# ------------------------------
# TRIANGLE 3 - Spherical

v3a_sph = VectorSpherical(0, 0, 0)
v3b_sph = VectorSpherical(2.0, 0, 0)
v3c_sph = VectorSpherical(2.0, np.pi/2, 0)

area_3_sph = VectorSpherical.area_vertices(v3a_sph, v3b_sph, v3c_sph)
print(f"Triangle 3 area (spherical) :{area_3_sph}")

angle_3_sph = VectorSpherical.angle_vertices(v3a_sph, v3b_sph, v3c_sph)
print(f"Triangle 3 angles (spherical):{angle_3_sph*180/np.pi}")

# ------------------------------
# TRIANGLE 4 - Spherical

v4a_sph = VectorSpherical(1.0, np.pi/2, 0)
v4b_sph = VectorSpherical(1.0, np.pi/2, np.pi)
v4c_sph = VectorSpherical(1.0, np.pi/2, 3*np.pi/2)

area_4_sph = VectorSpherical.area_vertices(v4a_sph, v4b_sph, v4c_sph)
print(f"Triangle 4 area (spherical) :{area_4_sph}")

angle_4_sph = VectorSpherical.angle_vertices(v4a_sph, v4b_sph, v4c_sph)
print(f"Triangle 4 angles (spherical):{angle_4_sph*180/np.pi}")

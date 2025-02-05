import numpy as np
import matplotlib.pyplot as plt

N = np.array([2, 4, 8, 16])

T_1 = 117.065

T_N = np.array([117.838, 58.528, 40.711, 23.427])

S_N = T_1/T_N

print(S_N)

plt.plot(N, np.array(S_N))
plt.xlabel("Number of processors")
plt.ylabel("Speedup")

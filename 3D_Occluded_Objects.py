import numpy as np
import matplotlib.pyplot as plt

P1 = np.array([0.000000e+00, 0.000000e+00, 0.000000e+00])
P2 = np.array([0.000000e+00, 1.000000e+00, 0.000000e+00])
P3 = np.array([1.000000e+00, 1.000000e+00, 0.000000e+00])
P4 = np.array([1.000000e+00, 0.000000e+00, 0.000000e+00])
P5 = np.array([0.000000e+00, 0.000000e+00, 1.000000e+00])
P6 = np.array([0.000000e+00, 1.000000e+00, 1.000000e+00])
P7 = np.array([1.000000e+00, 1.000000e+00, 1.000000e+00])
P8 = np.array([1.000000e+00, 0.000000e+00, 1.000000e+00])
C1 = np.array([5.000000e-01, 5.000000e-01, 1.000000e+00])
C2 = np.array([5.000000e-01, 5.000000e-01, 0.000000e+00])
R = 5.000000e-01

N = 500000
I = 0
temp = np.zeros((N, 3))

# Cuboid
V1 = P4 - P1
V2 = P2 - P1
V3 = P5 - P1
V1x = V1 * np.random.rand(N, 1)
V2y = V2 * np.random.rand(N, 1)
V3z = V3 * np.random.rand(N, 1)
Cuboid = V1x + V2y + V3z + P1
Cuboid_Volume = np.linalg.norm(V1) * np.linalg.norm(V2) * np.linalg.norm(V3)
print("Cuboid Volume:", Cuboid_Volume)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(Cuboid[:, 0], Cuboid[:, 1], Cuboid[:, 2], c='g')

# Cylinder
axial_vec = C2 - C1
axial_vec = axial_vec / np.linalg.norm(axial_vec)
axial_points = C1 + (C2 - C1) * np.random.rand(N, 1)
circ_r = np.sqrt(np.random.rand(N, 1)) * R
circ_theta = np.random.rand(N, 1) * 2 * np.pi
circ_points = np.hstack((np.cos(circ_theta) * circ_r, np.sin(circ_theta) * circ_r))
ax_null = np.linalg.norm(np.linalg.svd(axial_vec.reshape(-1, 1))[0][:, -1])
Cylinder = axial_points + circ_points[:, 0][:, np.newaxis] * axial_vec[0] + circ_points[:, 1][:, np.newaxis] * axial_vec[1]
Cylinder_Volume = np.pi * (R ** 2) * np.linalg.norm(axial_vec)
print("Cylinder Volume:", Cylinder_Volume)
ax.scatter(Cylinder[:, 0], Cylinder[:, 1], Cylinder[:, 2], c='b')

for i in range(N):
    Cube = np.array([Cuboid[i, 0], Cuboid[i, 1], Cuboid[i, 2]])
    # if it also exists on Cylinder
    if np.dot((Cube - C1), (C2 - C1)) >= 0 >= np.dot((Cube - C2), (C2 - C1)):
        if np.linalg.norm(np.cross((Cube - C1), (C2 - C1))) / np.linalg.norm(C2 - C1) <= R:
            I += 1  # count the intersection
            temp[I - 1] = [Cuboid[i, 0], Cuboid[i, 1], Cuboid[i, 2]]

ax.scatter(temp[:I, 0], temp[:I, 1], temp[:I, 2], c='m')
print("Points:", I)

ax.set_title('Group 1')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()

Result = (I / N) * Cuboid_Volume
print('Result:', Result)

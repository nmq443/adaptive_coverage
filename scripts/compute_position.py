import numpy as np

pos = np.array([11.25, 1.8349364905389036])
rh = 2.5
poses = []

for i in range(6):
    theta = 2 * np.pi * i / 6
    pi = pos + rh * np.array([np.cos(theta), np.sin(theta)])
    poses.append(pi)

for p in poses:
    print(f"[{p[0]}, {p[1]}],")

import numpy as np

source = np.array([11.25, 6.165063509461096])
pos = np.array([13.36, 7.24])
dir =
rh = 2.5
poses = []

for i in range(6):
    theta = 2 * np.pi * i / 6
    pi = pos + rh * np.array([np.cos(theta), np.sin(theta)])
    poses.append(pi)

for p in poses:
    print(f"[{p[0]}, {p[1]}],")

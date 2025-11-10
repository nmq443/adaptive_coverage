import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


fig, ax = plt.subplots(figsize=(7, 7))
width = 20
height = 10
ax.set_aspect('equal')
offset = 1.0
ax.set_xlim(0 - offset, width + offset)
ax.set_ylim(0 - offset, height + offset)

vertices = [
    [15, 0],
    [5, 0],
    [2, 3],
    [0, 6],
    [1.5, 10],
    [14, 10],
    [17.5, 9],
    [20, 5]
]

vertices = np.array(vertices)
env = Polygon(vertices, closed=True, fill=False)
ax.add_patch(env)

line1 = np.array([[11.5, 10], [12., 7]])
line2 = np.array([[13.34, 5.87], [14.27, 0.]])
ax.plot(line1[:, 0], line1[:, 1], c='black')
ax.plot(line2[:, 0], line2[:, 1], c='black')

positions = np.array([
    [5.0, 4.0],
    [7.5, 4.0],
    [6.25, 6.165063509461096],
    [3.7500000000000004, 6.165063509461097],
    [2.5, 4.0],
    [3.749999999999999, 1.8349364905389036],
    [6.25, 1.8349364905389036],
    [10.0, 4.0],
    [8.75, 6.165063509461096],
    [8.75, 1.8349364905389036],
    [11.25, 6.165063509461096],
    [10.0, 8.330127018922193],
    [7.5, 8.330127018922193],
    [6.25, 6.165063509461096],
    [5.000000000000001, 8.330127018922195],
    [2.500000000000001, 8.330127018922195],
    [1.2500000000000004, 6.165063509461097],
    [12.5, 4.0],
    [11.25, 1.8349364905389036],
    [13.75, 1.8349364905389036]
])

penalty = np.array([
    [13.36, 7.24]
])

ax.scatter(positions[:, 0], positions[:, 1], c='red')
ax.scatter(penalty[:, 0], penalty[:, 1], c='green')

plt.show()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon

fig, ax = plt.subplots(figsize=(8, 8))

rh = 2.5
size = 0.2
robot_i = np.array((0, 0))
robot_m = robot_i - np.array((rh, 0))
for i in range(6):
    phi = 2 * np.pi * i / 6
    node_i = robot_m + rh * np.array([np.cos(phi), np.sin(phi)])
    node_i_patch = Circle(node_i, size, color='red', fill=True)
    ax.add_patch(node_i_patch)

robot_i_patch = Circle(robot_i, size, color='red', fill=True)
robot_m_patch = Circle(robot_m, size, color='red', fill=True)

ax.add_patch(robot_i_patch)
ax.add_patch(robot_m_patch)

ax.set_xlim(-rh * 2, rh * 2)
ax.set_ylim(-rh * 2, rh * 2)

ax.set_aspect('equal')

plt.tight_layout()
plt.show()

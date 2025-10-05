import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon, Wedge

fig, ax = plt.subplots(figsize=(8, 8))

rh = 2.5
size = 0.2
robot_m = np.array((0, 0))
robot_i = robot_m - np.array((rh, 0))
ax.text(robot_m[0] + 2 * size, robot_m[1] + 0.5 * size, "Robot m")
ax.text(robot_i[0] - 7 * size, robot_i[1] + 0.5 * size, "Robot i")

# sensing range
robot_i_sensing_patch = Circle(
    robot_i, rh, edgecolor='black', fill=False, zorder=2)
robot_m_sensing_patch = Circle(
    robot_m, rh, edgecolor='black', fill=False, zorder=1)
ax.add_patch(robot_i_sensing_patch)
ax.add_patch(robot_m_sensing_patch)

obstacle_vertices = [
    [-6, 0.7],
    [-4.3, 1],
    [-4, -0.6],
    [-5.22, -1],
    [-5.95, -0.5]
]

obstacle = Polygon(obstacle_vertices, closed=True, fill=True,
                   edgecolor='black')
ax.add_patch(obstacle)

# robot i's virtual targets
robot_i_virtual_targets = []
for i in range(6):
    phi = 2 * np.pi * i / 6
    node_i = robot_i + rh * np.array([np.cos(phi), np.sin(phi)])
    ax.plot([robot_i[0], node_i[0]], [
            robot_i[1], node_i[1]], c='red', zorder=2)
    node_i_patch = Circle(node_i, size, facecolor='red',
                          edgecolor='black', fill=True, zorder=5)
    ax.text(node_i[0] + 2 * size, node_i[1] +
            2 * size, f'$P_i^{i+1}$', zorder=10)
    if i == 2:
        ax.text(node_i[0] - 2 * size, node_i[1] +
                2 * size, "Robot j", zorder=10)
    if i == 4:
        ax.text(node_i[0] - 2 * size, node_i[1] -
                2 * size, "Robot k", zorder=10)
    robot_i_virtual_targets.append(node_i)
    ax.add_patch(node_i_patch)

for i in range(5):
    if i == 2 or i == 3:
        continue
    node_i = robot_i_virtual_targets[i]
    node_j = robot_i_virtual_targets[i+1]
    ax.plot([node_i[0], node_j[0]], [node_i[1], node_j[1]], c='red', zorder=2)

node_i = robot_i_virtual_targets[0]
node_j = robot_i_virtual_targets[-1]
ax.plot([node_i[0], node_j[0]], [node_i[1], node_j[1]], c='red', zorder=2)

# robot m's virtual targets
robot_m_virtual_targets = []
for i in range(6):
    phi = 2 * np.pi * i / 6
    node_i = robot_m + rh * np.array([np.cos(phi), np.sin(phi)])
    robot_m_virtual_targets.append(node_i)
    ax.plot([robot_m[0], node_i[0]], [
            robot_m[1], node_i[1]], c='red', zorder=2)
    node_i_patch = Circle(node_i, size, facecolor='red',
                          edgecolor='black', fill=True, zorder=5)
    ax.text(node_i[0] - size, node_i[1] -
            3 * size, f'$P_m^{i+1}$', zorder=10)
    ax.add_patch(node_i_patch)

for i in range(5):
    node_i = robot_m_virtual_targets[i]
    node_j = robot_m_virtual_targets[i+1]
    ax.plot([node_i[0], node_j[0]], [node_i[1], node_j[1]], c='red', zorder=2)
node_i = robot_m_virtual_targets[0]
node_j = robot_m_virtual_targets[-1]
ax.plot([node_i[0], node_j[0]], [node_i[1], node_j[1]], c='red', zorder=2)

robot_i_patch = Circle(robot_i, size, facecolor='red',
                       edgecolor='black', fill=True)
robot_m_patch = Circle(robot_m, size, facecolor='red',
                       edgecolor='black', fill=True)

ax.add_patch(robot_i_patch)
ax.add_patch(robot_m_patch)

center = robot_i       # Robot i
radius = rh          # rh
theta1 = -120          # starting angle (deg)
theta2 = 120           # ending angle (deg)

sector = Wedge(center, radius, theta1, theta2,
               facecolor='yellow', edgecolor='none', zorder=1)
ax.add_patch(sector)

ax.text(-1.25, 0.63, "$S_{in}^i$")
ax.text(-3.75, 0.94, "$S_{out}^i$")
ax.text(-5.75, 0.45, "Đỉnh ẩn")

ax.set_xlim(-rh * 2.5, rh * 1.5)
ax.set_ylim(-rh * 1.5, rh * 1.5)

ax.set_aspect('equal')
ax.axis('off')

save_dir = "results/figures"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
plt.savefig(os.path.join(save_dir, "inner_and_outer_area.png"))
plt.tight_layout()
plt.show()

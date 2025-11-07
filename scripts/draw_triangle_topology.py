import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import os

# Coordinates of robots
pos_i = np.array([0.0, 0.5])
pos_j = np.array([-0.35, -0.6])
pos_k = np.array([0.35, -0.6])

# Parameters
r_free = 0.9       # Free region radius (light gray)
r_important = 1.3  # Important region radius (dark gray)
arrow_len = 0.25

# Figure setup
fig, ax = plt.subplots(figsize=(7, 7))
ax.set_aspect('equal')
ax.axis('off')

# Function to draw robot with zones


def draw_robot(ax, pos, label, sensing_range=True):
    if sensing_range:
        # Important zone (dark gray)
        ax.add_patch(
            Circle(pos, r_important, facecolor='gray', edgecolor='black', alpha=0.5, zorder=0))
        # Free zone (light gray)
        ax.add_patch(
            Circle(pos, r_free, facecolor='lightgray', edgecolor='black', alpha=0.5, zorder=1))
    # Robot
    ax.add_patch(Circle(pos, 0.1, facecolor='red',
                 edgecolor='black', linewidth=2, zorder=3))
    ax.text(pos[0], pos[1] + 0.2, label, fontsize=11,
            ha='center', va='bottom')


# Draw robots
draw_robot(ax, pos_i, 'i', False)
draw_robot(ax, pos_j, 'j')
draw_robot(ax, pos_k, 'k')

# Connections
for a, b in [(pos_i, pos_j), (pos_i, pos_k), (pos_j, pos_k)]:
    ax.plot([a[0], b[0]], [a[1], b[1]], color='blue', linewidth=1.5, zorder=2)

# Velocity vector of robot i
vi_dir = np.array([1.0, 0.4])  # direction
vi_dir = vi_dir / np.linalg.norm(vi_dir)
arrow_end = pos_i + arrow_len * vi_dir
ax.arrow(pos_i[0], pos_i[1], arrow_len * vi_dir[0], arrow_len * vi_dir[1],
         head_width=0.08, head_length=0.12, fc='blue', ec='blue', linestyle='dashed', zorder=4)
ax.text(arrow_end[0] + 0.05, arrow_end[1] + 0.05,
        r'$\vec{v_i}$', color='blue', fontsize=12)

# Destination d_i
d_i = vi_dir / 1.5 + pos_i
ax.add_patch(Circle(d_i, 0.01, color='blue', zorder=0))
ax.text(d_i[0], d_i[1] + 0.01,
        r'$d_i$', color='blue', fontsize=12)

# View
ax.set_xlim(-1.75, 1.75)
ax.set_ylim(-2, 0.75)

save_dir = "results/figures"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
plt.savefig(os.path.join(save_dir, "triangle_topo.png"))

plt.show()

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import os

# Coordinates of robots
pos_i = np.array([0.3, -0.3])
pos_j = np.array([-0.75, -0.6])
pos_k = np.array([0.75, -0.6])

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
        ax.add_patch(Circle(pos, r_important, facecolor='gray',
                            edgecolor='black', alpha=0.5, zorder=0))
        # Free zone (light gray)
        ax.add_patch(Circle(pos, r_free, facecolor='lightgray',
                            edgecolor='black', alpha=0.5, zorder=1))
    # Robot body
    ax.add_patch(Circle(pos, 0.1, facecolor='red',
                        edgecolor='blue', linewidth=2, zorder=3))
    ax.text(pos[0], pos[1] + 0.2, label, fontsize=11,
            ha='center', va='bottom')


# Draw robots
draw_robot(ax, pos_i, 'i', False)
draw_robot(ax, pos_j, 'j')
draw_robot(ax, pos_k, 'k')

# === Highlight only j’s important circle ===
highlight = Circle(pos_j, r_important, facecolor='green', alpha=0.35,
                   edgecolor='green', linewidth=2.5, linestyle='-', zorder=2)
ax.add_patch(highlight)

# Connections (solid lines from i to j and i to k)
for a, b in [(pos_i, pos_j), (pos_i, pos_k)]:
    ax.plot([a[0], b[0]], [a[1], b[1]], color='blue', linewidth=1.5, zorder=3)

# === Two short dashed line segments below j and k ===
dash_len = 0.6
angle_deg = -80
angle_rad = np.radians(angle_deg)

end_j = pos_j + dash_len * np.array([np.cos(angle_rad), np.sin(angle_rad)])
ax.plot([pos_j[0], end_j[0]], [pos_j[1], end_j[1]],
        color='blue', linewidth=1.5, linestyle='--', zorder=3)

end_k = pos_k + dash_len * np.array([-np.cos(angle_rad), np.sin(angle_rad)])
ax.plot([pos_k[0], end_k[0]], [pos_k[1], end_k[1]],
        color='blue', linewidth=1.5, linestyle='--', zorder=3)

# Label for I_jk = empty
mid_label = np.array([0.0, -0.8])
ax.text(mid_label[0], mid_label[1],
        r'$I_{jk} = \emptyset$', fontsize=11, ha='center', va='top')

# Velocity vector of robot i
vi_dir = np.array([1.0, 0.4])
vi_dir = vi_dir / np.linalg.norm(vi_dir)
arrow_end = pos_i + arrow_len * vi_dir
ax.arrow(pos_i[0], pos_i[1], arrow_len * vi_dir[0], arrow_len * vi_dir[1],
         head_width=0.08, head_length=0.12, fc='blue', ec='blue', zorder=4)
ax.text(arrow_end[0] + 0.05, arrow_end[1] + 0.05,
        r'$\vec{v_i}$', color='blue', fontsize=12)

# Destination d_i
d_i = vi_dir / 1.5 + pos_i
ax.add_patch(Circle(d_i, 0.03, color='blue', zorder=5))
ax.text(d_i[0] + 0.1, d_i[1], r'$d_i$', color='blue', fontsize=12)

# View limits
ax.set_xlim(-2.5, 2.5)
ax.set_ylim(-2, 1)

# Save figure
save_dir = "results/figures"
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, "one_connect_topo.png"))
plt.show()

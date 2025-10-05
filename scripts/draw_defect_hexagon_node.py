import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# --- Configuration for Plot (a) ---
center = (0, 0)
robot_radius = 0.3
rh_radius = 4.0  # Radius to the hexagonal nodes
num_nodes = 6
# Initial angle for the first node in degrees (like in the image)
phi_0_deg = 30
phi_0_rad = np.deg2rad(phi_0_deg)

# --- Create Plot ---
fig, ax = plt.subplots(figsize=(7, 7))

obstacle_vertices = np.array([
    (-5.0, -1.0),
    (-2.3, - 0.9),
    (-2.0, -3.0),
    (-3.5, -3.0)
])
obstacle = patches.Polygon(
    obstacle_vertices, closed=True, facecolor='blue', edgecolor='black', zorder=1)
ax.add_patch(obstacle)
ax.text(-rh_radius - 0.5, -3.0, 'Vật cản', ha='center',
        va='top', color='k', fontsize=12, fontweight='bold')

# 1. Plot the main circle (background for the hexagonal sensing area)
sensing_circle = patches.Circle(center, rh_radius,
                                facecolor='lightgray', edgecolor='navy', linewidth=1, zorder=0)
ax.add_patch(sensing_circle)

# 2. Plot the Robot (Red Circle at the center)
robot = patches.Circle(center, robot_radius,
                       color='red', zorder=5)
ax.add_patch(robot)
ax.text(center[0], center[1] + robot_radius * 1.2,
        'Robot i', ha='center', va='bottom', fontsize=10)
ax.text(center[0], center[1] + robot_radius * 0.5, 'i',
        ha='center', va='center', fontsize=10, fontstyle='italic')


# 3. Calculate and plot the hexagonal nodes (P_i^1 to P_i^6)
node_coords = []
for i in range(num_nodes):
    angle = phi_0_rad + (2 * np.pi / num_nodes) * i
    x = center[0] + rh_radius * np.cos(angle)
    y = center[1] + rh_radius * np.sin(angle)
    node_coords.append((x, y))

    # Plot the node as a white circle
    node = patches.Circle((x, y), 0.35, facecolor='white',
                          edgecolor='k', linewidth=1)
    ax.add_patch(node)

    # Label the node (P_i^1, P_i^2, etc.)
    text_pos = (x + 0.5, y + 0.6)
    color = 'black'
    if i == 0 or i == 1 or i == 2:
        text_pos = (x, y + 0.6)
    if i == 3:
        text_pos = (x - 0.7, y + 0.4)
        color = 'w'
    if i == 4:
        text_pos = (x, y - 0.8)
    if i == 5:
        text_pos = (x + 0.6, y)
    ax.text(text_pos[0], text_pos[1], r'$P_i^{' + str(i+1) + '}$',
            ha='center', va='bottom', fontsize=12, color=color)

# 4. Draw dotted lines from robot to nodes and form the hexagon
# Lines from robot to nodes
for x, y in node_coords:
    ax.plot([center[0], x], [center[1], y], 'k:', linewidth=1)

# Hexagon lines (connecting the nodes)
for i in range(num_nodes):
    start_node = node_coords[i]
    end_node = node_coords[(i + 1) % num_nodes]
    ax.plot([start_node[0], end_node[0]], [
            start_node[1], end_node[1]], 'k-', linewidth=1)

# Fill the central hexagon area with light gray (adjust zorder to be behind nodes)
hexagon_polygon = patches.Polygon(
    node_coords, closed=True, facecolor='lightgray', edgecolor='none', zorder=0)
ax.add_patch(hexagon_polygon)

# The regions outside the hexagon but inside the circle are dark gray
# This is a bit tricky with patches. A simpler way for visual effect:
# Draw a slightly smaller circle and then overlay the light gray hexagon.
# Or draw segments. For simplicity, we just rely on the background circle and hexagon filling.
# To achieve the dark gray crescents, we can draw the dark gray parts explicitly
# as arcs or by drawing a dark gray circle and then covering the hexagon.
# Let's try drawing arcs for the "uncovered" parts by the hexagon.
# (This is a complex approach, easier to think of it as the hexagon *cutting out* from the full circle).
# For now, we'll assume the `lightgray` for the circle and `lightgray` for the hexagon create the desired effect.
# If truly needed, one would draw a dark gray ring and then a light gray hexagon on top.
# For matching the image, the hexagon is *inside* a larger conceptual sensing area,
# and the "dark gray" crescents are the parts of the larger circle not covered by the central hexagon.
# Let's adjust the overall background.
# A better way for the "crescents" is to draw a dark gray circle, then a light gray hexagon, then a light gray circle inside the hexagon.
# To explicitly create the "dark gray crescent" effect:
dark_gray_circle = patches.Circle(
    center, rh_radius, facecolor='dimgray', edgecolor='none', zorder=-1)
ax.add_patch(dark_gray_circle)
light_gray_hexagon = patches.Polygon(
    node_coords, closed=True, facecolor='lightgray', edgecolor='none', zorder=0)
ax.add_patch(light_gray_hexagon)
light_gray_inner_circle = patches.Circle(
    center, rh_radius * 0.95, facecolor='lightgray', edgecolor='none', zorder=0)  # Small adjustment for aesthetics
ax.add_patch(light_gray_inner_circle)


# 5. Add rh radius arrow
ax.annotate('', xy=(center[0] + rh_radius * np.cos(phi_0_rad), center[1] + rh_radius * np.sin(phi_0_rad)),
            xytext=center, arrowprops=dict(arrowstyle="-", color="k", linewidth=1))
ax.text(center[0], center[1] + rh_radius *
        0.6, '$r_h$', ha='left', va='bottom', fontsize=12)

# 6. Add phi_0 angle arrow
# Arc for phi_0
arc = patches.Arc(center, rh_radius, rh_radius,
                  angle=0, theta1=0, theta2=phi_0_deg, color='k', linewidth=1)
ax.add_patch(arc)
ax.text(center[0] + 1.5, center[1] + 0.1,
        '$\phi_0$', ha='center', va='bottom', fontsize=12)

# X-axis arrow
ax.annotate('', xy=(rh_radius + 1, center[1]), xytext=center,
            arrowprops=dict(arrowstyle="->", color="k", linewidth=1.5))
ax.text(rh_radius + 1.2, center[1], 'X', ha='left',
        va='center', fontsize=12, fontweight='bold')


# 7. Set limits and aspect
limit = rh_radius * 1.5
ax.set_xlim(-limit, limit)
ax.set_ylim(-limit, limit)
ax.set_aspect('equal', adjustable='box')

# 8. Remove axis ticks and labels
ax.axis('off')

# 9. Add title for the subfigure
# plt.text(0.5, 0.05, '(a) Nút lục giác đầy đủ', ha='center',
#  va='center', transform=ax.transAxes, fontsize=14)

ax.text(-rh_radius * 0.75, -rh_radius * 0.25, 'Đỉnh ẩn', ha='center',
        va='top', color='w', fontsize=12, fontweight='bold')

# 10. Display the plot
plt.tight_layout()
save_dir = "results/figures"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
plt.savefig(os.path.join(save_dir, "defect_hexagon_node.png"))
plt.show()

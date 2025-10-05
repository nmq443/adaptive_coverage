'''
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

fig, ax = plt.subplots(figsize=(8, 8))


def normalize(x):
    return x / np.linalg.norm(x)


sensing_range = 5.0
free_range = sensing_range * 0.75
size = 0.4
avoidance_range = 1.0 + size

robot = Circle(xy=(0, 0), radius=size, facecolor='red', edgecolor='black')
sensing_area = Circle(xy=(0, 0), radius=sensing_range,
                      facecolor='gray', edgecolor='black')
free_area = Circle(xy=(0, 0), radius=free_range,
                   facecolor=normalize([177, 188, 204]), edgecolor='black')
avoidance_area = Circle(xy=(0, 0), radius=avoidance_range,
                        facecolor='white', edgecolor='black')

ax.add_patch(sensing_area)
ax.add_patch(free_area)
ax.add_patch(avoidance_area)
ax.add_patch(robot)

ax.text(-2*size, size, "Robot", fontsize=10)
ax.text(-1.5*size, -2.5*size, "Vùng tránh \nvật cản $Sa_i$", fontsize=10)

ax.text(size * 2, 0.1, "$r_a$", fontsize=12)
ax.arrow(0, 0, avoidance_range, 0, fc='k', ec='k')

ax.arrow(0, 0, free_range * np.cos(np.pi/4),
         free_range * np.sin(np.pi/4), fc='k', ec='k')
ax.text(free_range * np.cos(np.pi/3), free_range *
        np.sin(np.pi/4), "$r_f$", fontsize=12)

ax.arrow(0, 0, 0, sensing_range, fc='k', ec='k')
ax.text(-size, sensing_range * 0.85, "$r_c$", fontsize=12)

ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.set_aspect('equal')
ax.axis('off')

plt.show()

'''
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Define the radii for the zones (you can adjust these values)
r_a = 1.0  # Avoidance zone radius (Vùng tránh vật cản)
r_f = 2.5  # Free zone radius (Vùng tự do)
r_c = 4.0  # Consideration zone radius (Vùng quan trọng)

# Calculate epsilon for visual representation (epsilon = r_c - r_f)
epsilon = r_c - r_f

# 1. Setup the figure and axes
fig, ax = plt.subplots(figsize=(8, 8))

# 2. Plot the zones from largest to smallest for layering

# Vùng quan trọng (S_c_i) - Consideration Zone (Dark Gray Outer Ring)
# We plot the entire circle and then use the free zone to cut out the inner part.
c_zone = patches.Circle((0, 0), r_c,
                        facecolor='dimgray', edgecolor='k', linewidth=1)
ax.add_patch(c_zone)

# Vùng tự do (S_f_i) - Free Zone (Light Gray Inner Ring)
f_zone = patches.Circle((0, 0), r_f,
                        facecolor='lightgray', edgecolor='k', linewidth=1)
ax.add_patch(f_zone)

# Vùng tránh vật cản (S_a_i) - Avoidance Zone (White Dashed Circle)
# This zone is inside the Free Zone.
a_zone = patches.Circle((0, 0), r_a,
                        facecolor='white', edgecolor='navy', linestyle='--', linewidth=1.5)
ax.add_patch(a_zone)

# 3. Plot the Robot (Red Circle at the center)
robot = patches.Circle((0, 0), 0.2,
                       color='red', zorder=5)  # zorder to ensure it's on top
ax.add_patch(robot)

# 4. Add Text Labels
# Using the original Vietnamese labels from the image
ax.text(-r_a / 2, r_a / 3, 'Robot', ha='center',
        va='center', color='k', fontsize=11)
ax.text(0, -r_a * 0.3, 'Vùng tránh \nvật cản $S_{a_i}$',
        ha='center', va='top', color='k', fontsize=9)
ax.text(0, -r_f * 0.7, 'Vùng tự do $S_{f_i}$',
        ha='center', va='top', color='k', fontsize=11)
ax.text(0, -r_c * 0.8, 'Vùng quan trọng $S_{c_i}$',
        ha='center', va='top', color='k', fontsize=11)

# 5. Add Radii Arrows (Annotations)
arrow_style = dict(arrowstyle="<->", color="k", linewidth=1.5)

# r_a arrow
ax.annotate('', xy=(0, 0.0), xytext=(r_a, 0.0), arrowprops=arrow_style)
ax.text(r_a / 2, r_a * 0.2, '$r_a$', ha='left', va='center', fontsize=12)

# r_f arrow
ax.annotate('', xy=(r_f * 0.707, r_f * 0.707),
            xytext=(0, 0), arrowprops=arrow_style)
ax.text(r_f * 0.5, r_f * 0.7, '$r_f$', ha='left', va='center', fontsize=12)

# r_c arrow (vertical)
ax.annotate('', xy=(0, r_c), xytext=(0, 0), arrowprops=arrow_style)
ax.text(0.1, r_c / 4 * 3, '$r_c$', ha='left', va='center', fontsize=12)

# epsilon (ε) arrow (horizontal on the side)
# Horizontal line section for epsilon
y_epsilon = 0
ax.plot([-r_c, -r_f], [y_epsilon, y_epsilon],
        color='k', linestyle='-', linewidth=1.5)
# Vertical ticks
ax.plot([-r_c, -r_c], [y_epsilon - 0.1, y_epsilon + 0.1],
        color='k', linewidth=1.5)
ax.plot([-r_f, -r_f], [y_epsilon - 0.1, y_epsilon + 0.1],
        color='k', linewidth=1.5)
# Epsilon text
ax.text(-r_c + epsilon / 2, y_epsilon - 0.3,
        '$\epsilon$', ha='center', va='top', fontsize=12)


# 6. Set limits and aspect
limit = r_c * 1.2
ax.set_xlim(-limit, limit)
ax.set_ylim(-limit, limit)
ax.set_aspect('equal', adjustable='box')

# 7. Remove axis ticks and labels for a cleaner diagram look
ax.axis('off')

# 8. Display the plot
save_dir = "results/figures"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
plt.savefig(os.path.join(save_dir, "sensing_area.png"))
plt.show()

import matplotlib.pyplot as plt
import matplotlib.patches as patches

# --- Configuration ---
# Radii for the zones (same as the previous example for consistency)
r_a = 1.0  # Avoidance zone radius
r_f = 3.0  # Free zone radius
r_c = 4.0  # Consideration zone radius

# Centers for the two entities i and j
center_i = (-3.5, 0)
center_j = (3.5, 0)

# Coordinates for points k and h
# These points are placed for visual representation as in the original image.
point_k = (0.5, 2.0)
point_h = (0.73, -2.0)

# --- Plotting Function ---


def plot_zones(ax, center, r_c, r_f, r_a, label):
    """Adds the three concentric zones and the central entity to the plot."""
    x, y = center

    # Vùng quan trọng (S_c_i) - Consideration Zone (Dark Gray Outer Ring)
    c_zone = patches.Circle(center, r_c,
                            facecolor='dimgray', edgecolor='navy', linewidth=1)
    ax.add_patch(c_zone)

    # Vùng tự do (S_f_i) - Free Zone (Light Gray Inner Ring)
    f_zone = patches.Circle(center, r_f,
                            facecolor='lightgray', edgecolor='navy', linewidth=1)
    ax.add_patch(f_zone)

    # Vùng tránh vật cản (S_a_i) - Avoidance Zone (White Dashed Circle)
    a_zone = patches.Circle(center, r_a,
                            facecolor='white', edgecolor='navy', linestyle='--', linewidth=1.5)
    ax.add_patch(a_zone)

    # Central Entity (i or j)
    entity = patches.Circle(center, 0.2,
                            color='red', zorder=5)  # zorder to ensure it's on top
    ax.add_patch(entity)

    # Labels for the zones and entity
    ax.text(x, y + 0.25, label, ha='center', va='bottom',
            color='k', fontsize=9)
    ax.text(x, y - 0.25, '$S_{a_i}$', ha='center',
            va='top', color='k', fontsize=9)
    ax.text(x, y - 1.8, '$S_{f_i}$', ha='center',
            va='top', color='k', fontsize=9)
    ax.text(x, y - 3, '$S_{c_i}$', ha='center',
            va='top', color='k', fontsize=9)


# --- Create Plot ---
fig, ax = plt.subplots(figsize=(8, 8))

# 1. Plot the zones for entity 'i'
plot_zones(ax, center_i, r_c, r_f, r_a, 'i')

# 2. Plot the zones for entity 'j'
plot_zones(ax, center_j, r_c, r_f, r_a, 'j')

# 3. Plot points k and h
# Point k
ax.plot(point_k[0], point_k[1], 'ro', markersize=7, zorder=6)
ax.text(point_k[0] + 0.2, point_k[1], 'k', ha='left',
        va='center', fontsize=11)
# Point h
ax.plot(point_h[0], point_h[1], 'ro', markersize=7, zorder=6)
ax.text(point_h[0] + 0.2, point_h[1], 'h', ha='left',
        va='center', fontsize=11)


# 4. Add connecting lines between entities and points
line_color = 'k'
line_style = '-'
line_width = 1

# Lines from i
ax.plot([center_i[0], point_k[0]], [center_i[1], point_k[1]],
        color=line_color, linestyle=line_style, linewidth=line_width)
ax.plot([center_i[0], point_h[0]], [center_i[1], point_h[1]],
        color=line_color, linestyle=line_style, linewidth=line_width)

# Lines from j
ax.plot([center_j[0], point_k[0]], [center_j[1], point_k[1]],
        color=line_color, linestyle=line_style, linewidth=line_width)
ax.plot([center_j[0], point_h[0]], [center_j[1], point_h[1]],
        color=line_color, linestyle=line_style, linewidth=line_width)

# 5. Set limits and aspect
limit_x = r_c * 2.0
limit_y = r_c
ax.set_xlim(-limit_x, limit_x)
ax.set_ylim(-limit_y, limit_y)
ax.set_aspect('equal', adjustable='box')

# 6. Remove axis ticks and labels
ax.axis('off')

# 7. Display the plot
# plt.title('Mô hình tương tác vùng quan trọng giữa hai thực thể i và j', fontsize=14)
plt.savefig("results/figures/area_btw_i_and_j.png")
plt.show()

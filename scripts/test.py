import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from shapely.geometry import Polygon, Point
from adaptive_coverage.utils.utils import ray_intersects_aabb


def density_func(q):
    return 1


def centroid_region(agent_pos, sensing_range, vertices, obstacles, resolution=100):
    """
    Compute the centroid of the polygon using vectorized Trapezoidal rule on a grid.

    Args:
        agent_pos (numpy.ndarray): current agent's position.
        vertices (numpy.ndarray): vertices of the current agent's voronoi partition.
        resolution (int): number of grid points used to compute integral.

    Returns:
        numpy.ndarray: centroid of current agent's voronoi partition.
    """
    polygon = Polygon(vertices)
    xmax = np.max(vertices[:, 0])
    xmin = np.min(vertices[:, 0])
    ymax = np.max(vertices[:, 1])
    ymin = np.min(vertices[:, 1])
    n = resolution
    m = resolution
    hx = (xmax - xmin) / n
    hy = (ymax - ymin) / m

    x_coords = np.linspace(xmin, xmax, n + 1)
    y_coords = np.linspace(ymin, ymax, m + 1)
    xx, yy = np.meshgrid(x_coords, y_coords)
    grid_points = np.stack([xx.ravel(), yy.ravel()], axis=-1)

    # Vectorized point-in-polygon check
    shapely_points = [Point(p) for p in grid_points]
    mask_polygon = np.array([polygon.contains(sp) for sp in shapely_points]).reshape(
        xx.shape
    )

    # Vectorized sensing range check
    distances = np.linalg.norm(grid_points - agent_pos, axis=1).reshape(xx.shape)
    mask_range = distances < sensing_range

    if len(obstacles) > 0:
        mask_obstacles = np.ones(xx.shape, dtype=bool)
        for x, y, w, h in obstacles:
            in_x = (xx >= x) & (xx <= x + w)
            in_y = (yy >= y) & (yy <= y + h)
            mask_obstacles &= ~(in_x & in_y)

        visibility_mask = np.array(
            [
                not ray_intersects_aabb(agent_pos, point, obstacles)
                for point in grid_points
            ]
        ).reshape(xx.shape)
        mask = mask_polygon & mask_range & mask_obstacles & visibility_mask

    else:
        mask = mask_polygon & mask_range

    # Vectorized weight calculation
    wx = np.ones(n + 1)
    wx[1:-1] = 2
    wy = np.ones(m + 1)
    wy[1:-1] = 2
    weights = wx[:, np.newaxis] * wy[np.newaxis, :]

    # Evaluate the function f at all grid points (iterating because f returns a scalar)
    func_values = np.array([density_func(point) for point in grid_points]).reshape(
        xx.shape
    )

    # Apply the mask (only consider points inside the polygon)
    masked_func_values = func_values * mask
    masked_weights = weights * mask

    # Calculate mass
    total_mass = np.sum(masked_weights * masked_func_values) * (hx * hy) / 4

    # Calculate weighted centroid components
    weighted_x = np.sum(masked_weights * masked_func_values * xx) * (hx * hy) / 4
    weighted_y = np.sum(masked_weights * masked_func_values * yy) * (hx * hy) / 4

    if total_mass > 1e-8:
        centroid_x = weighted_x / total_mass
        centroid_y = weighted_y / total_mass
        centroid = np.array([centroid_x, centroid_y])
    else:
        centroid = agent_pos

    return centroid, grid_points[mask.ravel()]


fig, ax = plt.subplots()

agent_pos = np.array([5, 5])
sensing_range = 2.5
agent = Circle(agent_pos, radius=sensing_range, fill=False, color="black")

vertices = np.array([[0, 0], [0, 10], [10, 10], [10, 0]])

obstacles = np.array([[6, 6, 2, 3], [4, 4, 0.5, 0.5], [3, 6, 2, 2]])
if len(obstacles) > 0:
    for obstacle in obstacles:
        obs_rect = Rectangle(obstacle[:2], obstacle[2], obstacle[3])
        ax.add_patch(obs_rect)

ax.add_patch(agent)

ax.set_xlim(0, 10)
ax.set_ylim(0, 10)

centroid, points = centroid_region(agent_pos, sensing_range, vertices, obstacles)

print(f"Found centroid: {centroid}")

ax.scatter(points[:, 0], points[:, 1], c="red", s=5)
ax.scatter(centroid[0], centroid[1], c="green", s=20)

plt.show()

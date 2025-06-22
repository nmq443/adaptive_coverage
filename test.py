import pygame
import numpy as np
from configs import *
from utils import ray_intersects_aabb
from shapely.geometry import Polygon, Point


def density_func(q: np.ndarray):
    """
    Distribution density function. Right now I'm using uniform distribution.

    Args:
        q (numpy.ndarray): 2D position.

    Returns:
        float: f(q) represents the importance of q.
    """
    return 1


def centroid_region(
    agent_pos: np.ndarray,
    vertices: np.ndarray,
    obstacles: np.ndarray = [],
    resolution: int = 20,
):
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
    mask_range = distances <= VALID_RANGE

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
        print(visibility_mask)
        mask = mask_polygon & mask_range & mask_obstacles & visibility_mask

    else:
        mask = mask_polygon & mask_range

    valid_points = grid_points[mask.ravel()]

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

    centroid = CENTER

    if total_mass > 1e-8:
        centroid_x = weighted_x / total_mass
        centroid_y = weighted_y / total_mass
        centroid = np.array([centroid_x, centroid_y])

    return centroid, valid_points


def is_valid_particle(
    v1,
    v2,
    agent_pos,
    position: np.ndarray,
) -> bool:
    def cross_product(v1: np.ndarray, v2: np.ndarray):
        return v1[0] * v2[1] - v1[1] * v2[0]

    # Compute cross products
    v1 = v1 - agent_pos
    v2 = v2 - agent_pos
    cross1 = cross_product(v1, position)
    cross2 = cross_product(v2, position)
    cross12 = cross_product(v1, v2)

    # Check if P is inside the angle
    if cross12 > 0:
        return cross1 >= 0 and cross2 <= 0  # Counterclockwise
    else:
        return cross1 <= 0 and cross2 >= 0  # Clockwise


pygame.init()
screen = pygame.display.set_mode((1600, 900))
running = True

agent_pos = np.array([1600 / 2, 900 / 2])
# vertices = VERTICES
# obstacles = np.array(
#     [
#         [1600 / 2 + SIZE * 2, 900 / 2 + SIZE * 2, 1 * SCALE, 1 * SCALE],
#         [1600 / 3 + 200, 900 / 3 - SIZE * 2, 1 * SCALE, 1 * SCALE],
#     ]
# )
# centroid, valid_points = centroid_region(agent_pos, vertices, obstacles)
virtual_points = []
for i in range(6):
    phi = 2 * np.pi * i / 6
    x = agent_pos[0] + HEXAGON_RANGE * np.cos(phi)
    y = agent_pos[1] + HEXAGON_RANGE * np.sin(phi)
    virtual_points.append([x, y])

v1_idx = np.random.randint(0, 5)
space = [-1, 1]
v2_idx = v1_idx + space[np.random.randint(0, 1)] if v1_idx < 5 else 0
v1 = [virtual_points[v1_idx], v1_idx]
v2 = [virtual_points[v2_idx], v2_idx]
print(f"V1's idx: {v1_idx}, V2's idx: {v2_idx}")

phi_v1 = 2 * np.pi * v1[1] / 6
phi_v2 = 2 * np.pi * v2[1] / 6
if v1[1] > v2[1]:
    v1, v2 = v2, v1
v1x = agent_pos[0] + HEXAGON_RANGE * np.cos(phi_v1 + np.deg2rad(SWEEP_ANGLE_OFFSET))
v1y = agent_pos[1] + HEXAGON_RANGE * np.sin(phi_v1 + np.deg2rad(SWEEP_ANGLE_OFFSET))
v2x = agent_pos[0] + HEXAGON_RANGE * np.cos(phi_v2 - np.deg2rad(SWEEP_ANGLE_OFFSET))
v2y = agent_pos[1] + HEXAGON_RANGE * np.sin(phi_v2 - np.deg2rad(SWEEP_ANGLE_OFFSET))
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill("white")
    # for pt in virtual_points:
    #     pygame.draw.circle(screen, COLOR, pt, 5)
    pygame.draw.circle(screen, "blue", v1[0], 5)
    pygame.draw.circle(screen, "blue", v2[0], 5)
    pygame.draw.circle(screen, "green", (v1x, v1y), 5)
    pygame.draw.circle(screen, "green", (v2x, v2y), 5)
    # for obs in obstacles:
    #     pygame.draw.rect(screen, "black", obs)
    pygame.draw.circle(screen, COLOR, agent_pos, 10)
    pygame.draw.circle(screen, SENSING_COLOR, agent_pos, SENSING_RANGE, width=1)

    pygame.display.flip()

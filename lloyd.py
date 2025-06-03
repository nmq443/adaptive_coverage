import numpy as np
from shapely.geometry import Polygon, Point, LineString
from configs import *
from voronoi import *


def density_func(q):
    q = q - CENTER
    '''
    exponent = q[0] ** 2 + q[1] ** 2
    k = -0.0005
    return np.exp(k * exponent)
    '''
    return 1


def centroid_region(agent_pos, vertices, resolution=20):
    """
    Compute the centroid of the polygon using vectorized Trapezoidal rule on a grid.
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
    mask_polygon = np.array([polygon.contains(sp) for sp in shapely_points]).reshape(xx.shape)

    # Vectorized sensing range check
    distances = np.linalg.norm(grid_points - agent_pos, axis=1).reshape(xx.shape)
    mask_range = distances <= SENSING_RANGE

    mask = mask_polygon & mask_range

    # Vectorized weight calculation
    wx = np.ones(n + 1)
    wx[1:-1] = 2
    wy = np.ones(m + 1)
    wy[1:-1] = 2
    weights = wx[:, np.newaxis] * wy[np.newaxis, :]

    # Evaluate the function f at all grid points (iterating because f returns a scalar)
    func_values = np.array([density_func(point) for point in grid_points]).reshape(xx.shape)

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

    return centroid

def lloyd(agent, agents, env):
    """
    Continuous lloyd algorithm to find centroidal voronoi diagrams.

    Parameters
    ----------
    points : np.ndarray
        The generators array.
    polygon : shapely.Polygon
        The bounding polygon.
    iterations : int
        Number of iterations

    Returns
    -------
    new_points : np.ndarray
        New generators position.
    trajectories: np.ndarray
        The trajectories of all generators.
    """
    generators = [agent.index]
    for other in agents:
        distance = np.linalg.norm(other.pos - agent.pos)
        if other.index != agent.index and distance <= VALID_RANGE - EPS:
        # if other.index != agent.index and distance <= SENSING_RANGE and SENSING_RANGE - distance > EPS:
            generators.append(other.index)

    # Step 1: compute voronoi diagrams
    generators_positions = np.array([agents[index].pos for index in generators])
    vor = compute_voronoi_diagrams(generators_positions, env)

    # Step 2: compute centroidal voronoi diagrams
    centroids = np.zeros_like(generators_positions)
    for i, generator_pos in enumerate(generators_positions):
        for region in vor.filtered_regions:
            current_vertices = vor.vertices[region + [region[0]], :]
            current_polygon = Polygon(current_vertices)
            if current_polygon.contains(Point(generator_pos)):
                centroids[i] = centroid_region(agent.pos, current_vertices)

    # Step 3: move points to centroids
    goal = centroids[0]
    goal = handle_goal(goal, agent.pos, env)
    while not agent.terminated(goal):
        agent.move_to_goal(goal)


def handle_goal(goal, agent_pos, env):
    for obs in env.obstacles:
        x, y, w, h = obs
        edges = np.array([
            [[x, y], [x + w, y]],
            [[x + w, y], [x + w, y + h]],
            [[x + w, y + h], [x, y + h]],
            [[x, y + h], [x, y]]
        ])
        if x <= goal[0] <= x + w and y <= goal[1] <= y + h: # is inside an obstacle
            agent_to_goal = LineString(np.array([agent_pos, goal]))
            intersect = None
            for edge in edges:
                obs_edge = LineString(edge)
                if agent_to_goal.intersects(obs_edge):
                    intersect = agent_to_goal.intersection(obs_edge)
            if intersect is not None: 
                goal = np.array([intersect.x, intersect.y])

    return goal
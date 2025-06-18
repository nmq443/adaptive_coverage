import numpy as np
from environment import Environment
from shapely.geometry import Polygon, Point, LineString
from configs import *
from utils import perpendicular
from scipy.spatial import Voronoi


def density_func(q: np.ndarray):
    """
    Distribution density function. Right now I'm using uniform distribution.

    Args:
        q (numpy.ndarray): 2D position.

    Returns:
        float: f(q) represents the importance of q.
    """
    return 1


def centroid_region(agent_pos: np.ndarray, vertices: np.ndarray, resolution: int = 20):
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

    centroid = CENTER

    if total_mass > 1e-8:
        centroid_x = weighted_x / total_mass
        centroid_y = weighted_y / total_mass
        centroid = np.array([centroid_x, centroid_y])

    return centroid


def lloyd(agent, agents: list, env: Environment):
    """
    Continuous lloyd algorithm to find centroidal voronoi diagrams.

    Args:
        agent (Agent): current agent.
        agents (list): list of all agents.
        env (Environment): simulation environment.
    Returns:
        numpy.ndarray: goal for current agent.
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
    # while not agent.terminated(goal):
    # agent.move_to_goal(goal)
    # agent.move_to_goal(goal)
    return goal


def handle_goal(goal: np.ndarray, agent_pos: np.ndarray, env: Environment):
    """
    Project virtual goal to simulation environment.

    Args:
        goal (numpy.ndarray): virtual goal.
        agent_pos (numpy.ndarray): current agent's position.
        env (Environment): simulation environment.

    Returns:
        numpy.ndarray: projected goal onto the simulation environment.
    """
    original_goal = goal

    for obs in env.obstacles:
        x, y, w, h = obs
        edges = np.array(
            [
                [[x, y], [x + w, y]],
                [[x + w, y], [x + w, y + h]],
                [[x + w, y + h], [x, y + h]],
                [[x, y + h], [x, y]],
            ]
        )
        agent_to_goal = LineString(np.array([agent_pos, goal]))
        intersect = None
        for edge in edges:
            obs_edge = LineString(edge)
            if agent_to_goal.intersects(obs_edge):
                intersect = agent_to_goal.intersection(obs_edge)
        if intersect is not None:
            goal = np.array([intersect.x, intersect.y])

    if np.linalg.norm(goal - original_goal) > EPS:
        dir = goal - agent_pos
        dist = np.linalg.norm(dir)
        new_dir = (dist - SIZE) * dir / dist
        goal = new_dir + agent_pos

    return goal


def compute_voronoi_diagrams(generators: np.ndarray, env: Environment):
    """
    Compute bounded voronoi diagrams inside a polygon.

    Args:
        generators (numpy.ndarray): the generators array for computing polygon.
        env (Environment): simulation environment.

    Returns:
        vor (scipy.spatial.Voronoi): the resulting voronoi
    """
    mirroreds = []
    # mirror over edges
    for generator in generators:
        for edge in env.edges:
            projected_on_edge = perpendicular(generator, edge[0], edge[1])
            mirrored = 2 * projected_on_edge - generator
            mirroreds.append(mirrored)
    mirroreds = np.array(mirroreds)
    new_generators = np.vstack((generators, mirroreds))
    vor = Voronoi(new_generators)

    # Filter regions
    regions = []
    for region in vor.regions:
        flag = True
        for vertex_idx in region:
            if vertex_idx == -1:
                flag = False
                break
            vertex = vor.vertices[vertex_idx]
            if not (env.contains(vertex)):
                flag = False
                break
        if region and flag:
            regions.append(region)
    vor.filtered_points = generators
    vor.filtered_regions = regions
    return vor

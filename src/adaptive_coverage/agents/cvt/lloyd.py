import numpy as np
import pygame
from scipy.spatial import Voronoi
from shapely.geometry import Polygon, Point, LineString
from adaptive_coverage.utils.utils import perpendicular, ray_intersects_aabb


def density_func(q):
    """
    Distribution density function. Right now I'm using uniform distribution.

    Args:
        q (numpy.ndarray): 2D position.

    Returns:
        float: f(q) represents the importance of q.
    """
    return 1


def centroid_region(agent, vertices, env, resolution=20):
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
    distances = np.linalg.norm(grid_points - agent.pos, axis=1).reshape(xx.shape)
    mask_range = distances < agent.valid_range

    if len(env.obstacles) > 0:
        mask_obstacles = np.ones(xx.shape, dtype=bool)
        for x, y, w, h in env.obstacles:
            in_x = (xx >= x) & (xx <= x + w)
            in_y = (yy >= y) & (yy <= y + h)
            mask_obstacles &= ~(in_x & in_y)

        visibility_mask = np.array(
            [
                not ray_intersects_aabb(agent.pos, point, env.obstacles)
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
        centroid = agent.pos

    return centroid


def lloyd(agent, agents, env):
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
        if (
            other.index != agent.index
            and distance <= agent.valid_range - agent.tolerance
        ):
            if not ray_intersects_aabb(agent.pos, other.pos, env.obstacles):
                generators.append(other.index)

    if len(generators) < 2:
        return agent.pos

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
                centroids[i] = centroid_region(agent, current_vertices, env)

    # Step 3: move points to centroids
    goal = centroids[0]
    # goal = handle_goal(goal, agent, env)
    return goal


def handle_goal(goal, agent, env):
    """
    Project virtual goal to simulation environment.

    Args:
        goal (numpy.ndarray): virtual goal.
        agent_pos (numpy.ndarray): current agent's position.
        env (Environment): simulation environment.

    Returns:
        numpy.ndarray: projected goal onto the simulation environment.
    """
    in_obs = False
    for obs in env.obstacles:
        x, y, w, h = obs
        if x <= goal[0] <= x + w and y <= goal[1] <= y + h:
            in_obs = True
        if x <= goal[0] - agent.size <= x + w and y <= goal[1] - agent.size <= y + h:
            in_obs = True
        if x <= goal[0] + agent.size <= x + w and y <= goal[1] + agent.size <= y + h:
            in_obs = True
        if x <= goal[0] - agent.size <= x + w and y <= goal[1] + agent.size <= y + h:
            in_obs = True
        if x <= goal[0] + agent.size <= x + w and y <= goal[1] - agent.size <= y + h:
            in_obs = True
    if not in_obs:
        return goal
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
        agent_to_goal = LineString(np.array([agent.pos, goal]))
        intersect = None
        for edge in edges:
            obs_edge = LineString(edge)
            if agent_to_goal.intersects(obs_edge):
                intersect = agent_to_goal.intersection(obs_edge)
        if intersect is not None:
            goal = np.array([intersect.x, intersect.y])

    # in_obs = False
    # for obs in env.obstacles:
    #     x, y, w, h = obs
    #     if x <= goal[0] <= x + w and y <= goal[1] <= y + h:
    #         in_obs = True
    #     if x <= goal[0] - agent.size <= x + w and y <= goal[1] - agent.size <= y + h:
    #         in_obs = True
    #     if x <= goal[0] + agent.size <= x + w and y <= goal[1] + agent.size <= y + h:
    #         in_obs = True
    #     if x <= goal[0] - agent.size <= x + w and y <= goal[1] + agent.size <= y + h:
    #         in_obs = True
    #     if x <= goal[0] + agent.size <= x + w and y <= goal[1] - agent.size <= y + h:
    #         in_obs = True
    # if not in_obs:
    #     return goal
    if in_obs:
        # if np.linalg.norm(goal - original_goal) > 2 * EPS:
        direction = goal - agent.pos
        dist = np.linalg.norm(direction)
        new_dir = (dist - agent.size * 2) * direction / dist
        goal = new_dir + agent.pos

    return goal


def compute_voronoi_diagrams(generators, env):
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
    new_generators = np.where(np.isfinite(new_generators), new_generators, 0)
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

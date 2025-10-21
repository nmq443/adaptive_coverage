import numpy as np
from concurrent.futures import ThreadPoolExecutor
from scipy.spatial import Voronoi
from shapely import points
from shapely.geometry import Polygon, Point, LineString
from adaptive_coverage.utils.utils import perpendicular, ray_intersects_aabb


def density_func(q):
    """
    Distribution density function. Right now I'm using uniform distribution.

    Args:
        q: 2D position.

    Returns:
        f(q) represents the importance of q.
    """
    return 1


def centroid_region(agent, vertices, env, resolution=15):
    """
    Compute the centroid of the polygon using vectorized Trapezoidal rule on a grid.
    """

    polygon = Polygon(vertices)
    xmax, xmin = np.max(vertices[:, 0]), np.min(vertices[:, 0])
    ymax, ymin = np.max(vertices[:, 1]), np.min(vertices[:, 1])
    n = m = resolution
    hx = (xmax - xmin) / n
    hy = (ymax - ymin) / m

    x_coords = np.linspace(xmin, xmax, n + 1)
    y_coords = np.linspace(ymin, ymax, m + 1)
    xx, yy = np.meshgrid(x_coords, y_coords)
    grid_points = np.stack([xx.ravel(), yy.ravel()], axis=-1)

    # Vectorized point-in-polygon check (Shapely 2.1+)
    mask_polygon = polygon.contains(points(grid_points)).reshape(xx.shape)

    # Vectorized sensing range check
    distances = np.linalg.norm(
        grid_points - agent.pos, axis=1).reshape(xx.shape)
    mask_range = distances < agent.critical_range

    if len(env.obstacles) > 0:
        # Obstacles mask (axis-aligned bounding boxes)
        mask_obstacles = np.ones(xx.shape, dtype=bool)
        for x, y, w, h in env.obstacles:
            in_x = (xx >= x) & (xx <= x + w)
            in_y = (yy >= y) & (yy <= y + h)
            mask_obstacles &= ~(in_x & in_y)

        # Parallelized visibility check
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(
                lambda p: not ray_intersects_aabb(agent.pos, p, env.obstacles),
                grid_points
            ))
        visibility_mask = np.array(results).reshape(xx.shape)
        '''
        visibility_mask = np.array(
            [
                not ray_intersects_aabb(agent.pos, point, env.obstacles)
                for point in grid_points
            ]
        ).reshape(xx.shape)
        '''

        mask = mask_polygon & mask_range & mask_obstacles & visibility_mask
    else:
        mask = mask_polygon & mask_range

    # Vectorized weights for trapezoidal rule
    wx = np.ones(n + 1)
    wx[1:-1] = 2
    wy = np.ones(m + 1)
    wy[1:-1] = 2
    weights = wx[:, np.newaxis] * wy[np.newaxis, :]

    # Density function is constant 1
    func_values = np.ones(xx.shape)

    # Apply masks
    masked_func_values = func_values * mask
    masked_weights = weights * mask

    # Calculate mass
    total_mass = np.sum(masked_weights * masked_func_values) * (hx * hy) / 4

    # Calculate weighted centroid components
    weighted_x = np.sum(
        masked_weights * masked_func_values * xx) * (hx * hy) / 4
    weighted_y = np.sum(
        masked_weights * masked_func_values * yy) * (hx * hy) / 4

    if total_mass > 1e-8:
        centroid = np.array([weighted_x / total_mass, weighted_y / total_mass])
    else:
        centroid = agent.pos

    return centroid


def lloyd(agent, agents, env):
    """
    Continuous lloyd algorithm to find centroidal voronoi diagrams.

    Args:
        agent: current agent.
        agents: list of all agents.
        env: simulation environment.
    Returns:
        numpy.ndarray: goal for current agent.
    """
    generators = [agent.index]
    for other in agents:
        distance = np.linalg.norm(other.pos - agent.pos)
        if other.index != agent.index and distance < agent.critical_range:
            if not ray_intersects_aabb(agent.pos, other.pos, env.obstacles):
                generators.append(other.index)

    if len(generators) < 2:
        return agent.pos

    # Step 1: compute voronoi diagrams
    generators_positions = np.array(
        [agents[index].pos for index in generators])
    vor = compute_voronoi_diagrams(generators_positions, env)

    # Step 2: compute centroidal voronoi diagrams
    centroids = np.zeros_like(generators_positions)
    for i, generator_pos in enumerate(generators_positions):
        # for region in vor.filtered_regions:
        for region in getattr(vor, "filtered_regions"):
            current_vertices = vor.vertices[region + [region[0]], :]
            current_polygon = Polygon(current_vertices)
            if current_polygon.contains(Point(generator_pos)):
                centroids[i] = centroid_region(agent, current_vertices, env)

    # Step 3: move points to centroids
    goal = centroids[0]
    goal = handle_goal(goal, agent, env)
    return goal


def compute_voronoi_diagrams(generators, env):
    """
    Compute bounded voronoi diagrams inside a polygon.

    Args:
        generators: the generators array for computing polygon.
        env: simulation environment.

    Returns:
        vor: the resulting voronoi
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
    setattr(vor, "filtered_points", generators)
    setattr(vor, "filtered_regions", regions)
    return vor


def handle_goal(goal, agent, env):
    """
    Handle goal computed by the agent.

    Args:
        goal: raw agent's goal to be checked.
        agent: current agent.
        env: simulation environment.

    Returns:
        goal: valid goal.
    """
    original_goal = np.array(goal)
    agent_to_goal = LineString([agent.pos, goal])

    # Inflate obstacles by agent.size (buffer creates clearance area)
    inflated_obstacles = []
    for obs in env.obstacles:
        x, y, w, h = obs
        rect = Polygon([
            (x, y), (x + w, y),
            (x + w, y + h), (x, y + h)
        ])
        inflated_obstacles.append(rect.buffer(agent.size))

    closest_intersect = None
    min_dist = float("inf")

    # Check path intersections with inflated obstacles
    for poly in inflated_obstacles:
        if agent_to_goal.intersects(poly):
            inter = agent_to_goal.intersection(poly.boundary)
            if not inter.is_empty:
                if inter.geom_type == "Point":
                    dist = Point(agent.pos).distance(inter)
                    if dist < min_dist:
                        min_dist = dist
                        closest_intersect = np.array([inter.x, inter.y])
                else:
                    # handle LineString intersection, pick nearest point
                    for pt in inter.geoms:
                        dist = Point(agent.pos).distance(pt)
                        if dist < min_dist:
                            min_dist = dist
                            closest_intersect = np.array([pt.x, pt.y])

    # Adjust goal if intersection happens
    if closest_intersect is not None:
        dir_vec = closest_intersect - agent.pos
        dist = np.linalg.norm(dir_vec)
        new_dir = (dist - agent.size) * dir_vec / dist
        goal = new_dir + agent.pos
    else:
        goal = original_goal

    return goal

import numpy as np
import pygame
from scipy.spatial import Voronoi
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


def centroid_region(agent, vertices, env, resolution=20):
    """
    Compute the centroid of the polygon using the Rectangle Rule (midpoint rule) on a grid.

    Args:
        agent: agent object with pos, size, and critical_range attributes.
        vertices: vertices of the current agent's Voronoi partition.
        env: environment containing obstacles.
        resolution: number of grid points per dimension.

    Returns:
        centroid of current agent's Voronoi partition.
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

    # Rectangle rule: sample at midpoints of cells
    x_coords = xmin + (np.arange(n) + 0.5) * hx
    y_coords = ymin + (np.arange(m) + 0.5) * hy
    xx, yy = np.meshgrid(x_coords, y_coords)
    grid_points = np.stack([xx.ravel(), yy.ravel()], axis=-1)

    # Polygon mask
    shapely_points = [Point(p) for p in grid_points]
    mask_polygon = np.array([polygon.contains(sp)
                            for sp in shapely_points]).reshape(xx.shape)

    # Sensing range mask
    distances = np.linalg.norm(
        grid_points - agent.pos, axis=1).reshape(xx.shape)
    mask_range = distances < agent.critical_range

    if len(env.obstacles) > 0:
        mask_obstacles = np.ones(xx.shape, dtype=bool)
        for x, y, w, h in env.obstacles:
            in_x = (xx >= x - agent.size * 4) & (xx <=
                                                 x + w + agent.size * 4)
            in_y = (yy >= y - agent.size * 4) & (yy <=
                                                 y + h + agent.size * 4)
            mask_obstacles &= ~(in_x & in_y)

        visibility_mask = np.array(
            [not ray_intersects_aabb(agent.pos, point, env.obstacles)
             for point in grid_points]
        ).reshape(xx.shape)

        mask = mask_polygon & mask_range & mask_obstacles & visibility_mask
    else:
        mask = mask_polygon & mask_range

    # Function values (density)
    func_values = np.array([density_func(point)
                           for point in grid_points]).reshape(xx.shape)

    # Apply mask
    masked_func_values = func_values * mask

    # Rectangle rule integration
    cell_area = hx * hy
    total_mass = np.sum(masked_func_values) * cell_area
    weighted_x = np.sum(masked_func_values * xx) * cell_area
    weighted_y = np.sum(masked_func_values * yy) * cell_area

    if total_mass > 1e-8:
        centroid_x = weighted_x / total_mass
        centroid_y = weighted_y / total_mass
        centroid = np.array([centroid_x, centroid_y])
        centroid = np.array([centroid_x, centroid_y])
        # Post-check: if centroid lies inside an obstacle, pick nearest valid grid point
        if any(Polygon([(x, y), (x+w, y), (x+w, y+h), (x, y+h)]).contains(Point(centroid))
               for (x, y, w, h) in env.obstacles):
            # fallback: choose closest valid point from sampled grid
            valid_points = grid_points[mask.ravel()]
            if len(valid_points) > 0:
                distances = np.linalg.norm(valid_points - agent.pos, axis=1)
                centroid = valid_points[np.argmin(distances)]
            else:
                centroid = agent.pos
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

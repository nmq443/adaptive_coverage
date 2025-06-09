import os
import numpy as np
from configs import *
from shapely.geometry import Polygon

from shapely.geometry import Polygon, Point
import numpy as np


def compute_coverage_percentage(positions):
    """
    Compute coverage percentage over a polygonal region using mesh grid.

    Args:
        positions (numpy.ndarray): positions of agents, shape (n, 2)
        polygon_vertices (numpy.ndarray): polygon vertices, shape (m, 2)

    Returns:
        float: coverage percentage (0.0 to 1.0)
    """
    resolution = 100
    polygon = Polygon(VERTICES)

    # Get bounds of polygon to limit the grid
    minx, miny, maxx, maxy = polygon.bounds
    x_vals = np.linspace(minx, maxx, resolution)
    y_vals = np.linspace(miny, maxy, resolution)
    xx, yy = np.meshgrid(x_vals, y_vals)
    grid_points = np.column_stack((xx.ravel(), yy.ravel()))  # shape: (N, 2)

    # Step 1: Filter points inside the polygon
    inside_polygon_mask = np.array([
        polygon.contains(Point(p)) for p in grid_points
    ])
    inside_polygon_points = grid_points[inside_polygon_mask]

    # Step 2: Compute coverage mask (robots)
    in_coverage = np.zeros(inside_polygon_points.shape[0], dtype=bool)
    for pos in positions:
        distances = np.linalg.norm(inside_polygon_points - pos, axis=1)
        in_coverage |= distances <= SENSING_RANGE

    # Step 3: Obstacle mask
    in_obs = np.zeros(inside_polygon_points.shape[0], dtype=bool)
    if len(OBSTACLES) > 0:
        for obs in OBSTACLES:
            x_min, y_min, w, h = obs
            x_max, y_max = x_min + w, y_min + h
            inside = (
                (inside_polygon_points[:, 0] >= x_min) & (inside_polygon_points[:, 0] <= x_max) &
                (inside_polygon_points[:, 1] >= y_min) & (
                    inside_polygon_points[:, 1] <= y_max)
            )
            in_obs |= inside

    # Step 4: Final valid coverage
    valid_covered = in_coverage & (~in_obs)

    # Step 5: Return coverage ratio within polygon
    return valid_covered.sum() / inside_polygon_points.shape[0] if inside_polygon_points.shape[0] > 0 else 0.0


def evaluate(controller='voronoi', approach='original'):
    if controller == 'voronoi':
        filename = os.path.join(RES_DIR, controller, ENV_DIR, "swarm_data.npy")
    else:
        if approach == 'original':
            filename = os.path.join(
                RES_DIR, controller, ENV_DIR, approach, "swarm_data.npy")
        else:
            filename = os.path.join(
                RES_DIR, controller, ENV_DIR, approach, "swarm_data.npy")

    with open(filename, 'rb') as f:
        datas = np.load(f)
    final_poses = datas[:, 1]

    area = compute_coverage_percentage(final_poses)

    if controller == 'voronoi':
        print(f"Percent coverage for {controller} method: {area: .2f}")
    else:
        if approach == 'original':
            print(
                f"Percent coverage for {controller} method with original approach: {area: .2f}")
        else:
            print(
                f"Percent coverage for {controller} method with PSO approach: {area: .2f}")
    print("----------")


if __name__ == '__main__':
    controllers = ['voronoi', 'hexagon']
    approaches = ['pso', 'original']
    results = set()
    for controller in controllers:
        if controller == 'hexagon':
            for approach in approaches:
                evaluate(controller, approach)
        else:
            evaluate(controller, approach=None)

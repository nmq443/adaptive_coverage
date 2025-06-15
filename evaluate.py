import os
import numpy as np
from configs import *
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point


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
    inside_polygon_mask = np.array([polygon.contains(Point(p)) for p in grid_points])
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
                (inside_polygon_points[:, 0] >= x_min)
                & (inside_polygon_points[:, 0] <= x_max)
                & (inside_polygon_points[:, 1] >= y_min)
                & (inside_polygon_points[:, 1] <= y_max)
            )
            in_obs |= inside

    # Step 4: Final valid coverage
    valid_covered = in_coverage & (~in_obs)

    # Step 5: Return coverage ratio within polygon
    return (
        valid_covered.sum() / inside_polygon_points.shape[0]
        if inside_polygon_points.shape[0] > 0
        else 0.0
    )


def evaluate(controller="voronoi", approach="original"):
    """
    Compare different methods.

    Args:
        controller (str): name of the methods.
        approach (str): if using hexagonal lattices method, modify approach to use original or PSO based approach.
    """
    if controller == "voronoi":
        filename = os.path.join(RES_DIR, controller, ENV_DIR, "swarm_data.npy")
    else:
        if approach == "original":
            filename = os.path.join(
                RES_DIR, controller, ENV_DIR, approach, "swarm_data.npy"
            )
        else:
            filename = os.path.join(
                RES_DIR, controller, ENV_DIR, approach, "swarm_data.npy"
            )

    with open(filename, "rb") as f:
        datas = np.load(f)
    final_poses = datas[:, 1]

    area = compute_coverage_percentage(final_poses)

    if controller == "voronoi":
        print(f"Percent coverage for {controller} method: {area: .2f}")
    else:
        if approach == "original":
            print(
                f"Percent coverage for {controller} method with original approach: {area: .2f}"
            )
        else:
            print(
                f"Percent coverage for {controller} method with PSO approach: {area: .2f}"
            )
    print("----------")


def laplacian_mat(agents):
    """
    Get Laplacian matrix of a networked multi-agent system.

    Args:
        agents (list): List of all agents.

    Returns:
        numpy.ndarray: A numpy.ndarray with shape (N, N) with N is the number of agents.
    """
    adj_mat = np.zeros((len(agents), len(agents)))  # adjacency matrix
    for i in range(len(agents)):
        for j in range(len(agents)):
            if i == j:
                continue
            dist = np.linalg.norm(agents[i].pos - agents[j].pos)
            if dist <= agents[i].rs:
                adj_mat[i][j] = adj_mat[j][i] = 1
    deg_mat = np.zeros_like(adj_mat)  # degree matrix
    for i in range(len(agents)):
        deg_mat[i][i] = np.sum(adj_mat[i])

    laplacian_mat = deg_mat - adj_mat

    return laplacian_mat


def lamda2(agents):
    """
    Get second smallest eigen value of laplacian matrix.

    Args:
        agents (list): List of all agents in networked multi-robot system.

    Returns:
        float: Second largest eigen value of laplacian matrix.
    """
    l_mat = laplacian_mat(agents)
    eigen_vals = np.linalg.eigvalsh(l_mat)

    # 1 because np.linalg.eigvalsh() return an array of eigen values in ascending order
    return eigen_vals[1]


def plot_travel_distances(distances, save_dir=""):
    """
    Plot travel distances of all agents.

    Args:
        distances (list): Travel distances of all agents.
        save_dir (str): Where to save the plot.
    """
    fig = plt.figure()
    ax = plt.gca()

    ax.bar(np.arange(len(distances)), distances)
    # ax.set_title("Travel distance of all agents.")
    ax.set_xlabel("Agent")
    ax.set_ylabel("Distances (m)")

    if save_dir != "":
        plt.savefig(os.path.join(save_dir, "travel_distances.png"))


def plot_ld2(ld2s, logger, save_dir=""):
    """
    Plot lambda2 value of the swarm over time.

    Args:
        ld2s (list): ld2 value over time.
        save_dir (str): Where to save the plot.
    """
    logger.info(f"Final lambda2 value: {ld2s[-1]: .2f}")
    logger.info("-----")
    fig = plt.figure()
    ax = plt.gca()

    ax.plot(np.arange(len(ld2s)), ld2s)
    # ax.set_title("Lamda2 value over time.")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Lambda2")

    if save_dir != "":
        plt.savefig(os.path.join(save_dir, "ld2.png"))


if __name__ == "__main__":
    controllers = ["voronoi", "hexagon"]
    approaches = ["pso", "original"]
    results = set()
    for controller in controllers:
        if controller == "hexagon":
            for approach in approaches:
                evaluate(controller, approach)
        else:
            evaluate(controller, approach=None)

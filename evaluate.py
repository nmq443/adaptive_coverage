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
        swarm_data_filename = os.path.join(
            RES_DIR, controller, ENV_DIR, "swarm_data.npy"
        )
        ld2s_filename = os.path.join(RES_DIR, controller, ENV_DIR, "ld2s.npy")
        travel_distances_filename = os.path.join(
            RES_DIR, controller, ENV_DIR, "travel_distances.npy"
        )
        save_dir = os.path.join(RES_DIR, controller, ENV_DIR)
    else:
        if approach == "original":
            swarm_data_filename = os.path.join(
                RES_DIR, controller, ENV_DIR, approach, "swarm_data.npy"
            )
            ld2s_filename = os.path.join(
                RES_DIR, controller, ENV_DIR, approach, "ld2s.npy"
            )
            travel_distances_filename = os.path.join(
                RES_DIR, controller, ENV_DIR, approach, "travel_distances.npy"
            )
            save_dir = os.path.join(RES_DIR, controller, ENV_DIR, approach)
        else:
            swarm_data_filename = os.path.join(
                RES_DIR, controller, ENV_DIR, approach, "swarm_data.npy"
            )
            ld2s_filename = os.path.join(
                RES_DIR, controller, ENV_DIR, approach, "ld2s.npy"
            )
            travel_distances_filename = os.path.join(
                RES_DIR, controller, ENV_DIR, approach, "travel_distances.npy"
            )
            save_dir = os.path.join(RES_DIR, controller, ENV_DIR, approach)

    with open(swarm_data_filename, "rb") as f:
        swarm_data = np.load(f)
    with open(ld2s_filename, "rb") as f:
        ld2s = np.load(f)
    with open(travel_distances_filename, "rb") as f:
        distances = np.load(f)

    final_poses = swarm_data[:, 1]

    area = compute_coverage_percentage(final_poses)
    plot_ld2(ld2s, save_dir)
    plot_travel_distances(distances, save_dir="")

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
            if dist <= SENSING_RANGE:
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


def plot_travel_distances(distances, agent_labels=None, save_dir=""):
    """
    Plots the total travel distances for each robot as a bar chart.

    Args:
        distances (list or np.ndarray): A list or NumPy array of total travel distances
                                         for each robot. Each element represents one robot.
        agent_labels (list, optional): A list of strings representing the labels for each agent/robot.
                                       If None, default labels (e.g., "Agent 0", "Agent 1") will be used.
        save_dir (str, optional): The directory where the plot will be saved.
                                  If an empty string, the plot will not be saved.
                                  Defaults to "".
    """
    if not isinstance(distances, (list, np.ndarray)):
        raise TypeError("distances must be a list or NumPy array.")
    if len(distances) == 0:
        print("Warning: No distances provided to plot.")
        return

    num_agents = len(distances)
    x_pos = np.arange(num_agents)

    # Create the figure and axes
    fig, ax = plt.subplots(figsize=(10, 6))  # Adjust figsize for better proportions

    # Plot the bar chart
    bars = ax.bar(
        x_pos, distances, color="skyblue"
    )  # Added a color for better aesthetics

    # Set labels and title
    ax.set_xlabel("Agent", fontsize=12)  # Changed to "Robot", increased font size
    ax.set_ylabel(
        "Total Distance (m)", fontsize=12
    )  # More descriptive label, increased font size
    # Set x-axis ticks and labels
    if agent_labels:
        if len(agent_labels) != num_agents:
            raise ValueError(
                "Length of agent_labels must match the number of distances."
            )
        ax.set_xticks(x_pos)
        ax.set_xticklabels(
            agent_labels, rotation=45, ha="right"
        )  # Rotate labels if many agents
    else:
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f"{i}" for i in range(num_agents)])

    # Add value labels on top of the bars
    for bar in bars:
        yval = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            yval + 0.05 * yval,
            f"{yval:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )  # Format to 2 decimal places

    # Add a grid for better readability
    ax.yaxis.grid(True, linestyle="--", alpha=0.7)

    # Improve layout to prevent labels from overlapping
    plt.tight_layout()

    # Save the figure if save_dir is provided
    if save_dir:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)  # Create directory if it doesn't exist
        save_path = os.path.join(save_dir, "travel_distances.png")
        plt.savefig(
            save_path, dpi=300, bbox_inches="tight"
        )  # High resolution, tight bounding box
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()  # Display the plot if not saving


def plot_ld2(lambda2_values, save_dir=""):
    """
    Plots the Lambda2 value of the swarm over time, indicating cohesion.

    Args:
        lambda2_values (list or np.ndarray): A list or NumPy array of Lambda2 values,
                                            representing swarm cohesion at each timestep.
        save_dir (str, optional): The directory where the plot will be saved.
                                  If an empty string, the plot will be displayed instead.
                                  Defaults to "".
    """
    if not isinstance(lambda2_values, (list, np.ndarray)):
        raise TypeError("lambda2_values must be a list or NumPy array.")
    if len(lambda2_values) == 0:
        print("Warning: No Lambda2 values provided to plot.")
        return

    timesteps = np.arange(len(lambda2_values))

    # Create the figure and axes
    fig, ax = plt.subplots(
        figsize=(12, 7)
    )  # Adjusted figure size for better readability

    # Plot the Lambda2 values
    ax.plot(
        timesteps,
        lambda2_values,
        color="cornflowerblue",
        linewidth=2,
        marker="o",
        markersize=4,
        label="Lambda2 Value",
    )  # Added color, line width, markers, and label

    # Set labels and title
    ax.set_xlabel("Timestep", fontsize=13)
    ax.set_ylabel("Lambda2 Value", fontsize=13)  # More descriptive label

    # Add a grid for better readability
    ax.grid(True, linestyle="--", alpha=0.7)

    # Add a legend
    ax.legend(fontsize=10)

    # Customize ticks
    ax.tick_params(axis="both", which="major", labelsize=10)

    # Improve layout
    plt.tight_layout()

    # Save or display the figure
    if save_dir:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)  # Create directory if it doesn't exist
        save_path = os.path.join(save_dir, "lambda2_over_time.png")
        plt.savefig(
            save_path, dpi=300, bbox_inches="tight"
        )  # High resolution, tight bounding box
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()


'''
def plot_ld2(ld2s, save_dir=""):
    """
    Plot lambda2 value of the swarm over time.

    Args:
        ld2s (list): ld2 value over time.
    """
    fig = plt.figure()
    ax = plt.gca()

    ax.plot(np.arange(len(ld2s)), ld2s)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Lambda2")

    if save_dir != "":
        plt.savefig(os.path.join(save_dir, "ld2.png"))
    else:
        plt.show()
'''

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

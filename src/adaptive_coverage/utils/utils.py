import os
import numpy as np
import networkx as nx
import yaml
import argparse
import matplotlib.pyplot as plt
from scipy.linalg import eigvalsh
from shapely.geometry import Polygon, Point


def meters2pixels(x, scale):
    """
    Convert from meter to pixel.

    Args:
        x: value in meters.
        scale (float): how much to scale.

    Returns:
        float: value in pixels.
    """
    return x * scale


def pixels2meters(x, scale):
    """
    Convert from pixel to meter.

    Args:
        x (float): value in pixel.
        scale (float): how much to scale.

    Returns:
        float: value in meters.
    """
    return x / scale


def perpendicular(x, a, b):
    """
    Finds projection of x on ab.

    Args:
        a: numpy array [ax, ay] representing point A.
        b: numpy array [bx, by] representing point B.
        x: numpy array [cx, cy] representing point X.

    Returns:
        NumPy array [hx, hy] representing point H.
    """
    ab = b - a  # Vector AB
    ac = x - a  # Vector AC
    t = (ac @ ab) / (ab @ ab)
    return a + t * ab


def ray_intersects_aabb(p1, p2, obstacles):
    """
    Test if segment [p1, p2] intersects any AABB in 'obstacles'.

    Args:
        p1 (numpy.ndarray): start position.
        p2 (numpy.ndarray): end position.
        obstacles (numpy.ndarray): list of obstacles.

    Returns:
        bool: if segment [p1, p2] intersects any AABB in 'obstacles'.
    """
    if len(obstacles) <= 0:
        return False
    p1 = np.asarray(p1)
    p2 = np.asarray(p2)

    # Directions
    d = p2 - p1  # shape: (2,)

    # Avoid division by zero
    d = np.where(d == 0, 1e-12, d)

    # Unpack rects: shape (N,)
    x_min = obstacles[:, 0]
    y_min = obstacles[:, 1]
    x_max = x_min + obstacles[:, 2]
    y_max = y_min + obstacles[:, 3]

    # Parametric t values
    tx1 = (x_min - p1[0]) / d[0]
    tx2 = (x_max - p1[0]) / d[0]
    ty1 = (y_min - p1[1]) / d[1]
    ty2 = (y_max - p1[1]) / d[1]

    tmin = np.maximum(np.minimum(tx1, tx2), np.minimum(ty1, ty2))
    tmax = np.minimum(np.maximum(tx1, tx2), np.maximum(ty1, ty2))

    # Segment overlaps if tmin ≤ tmax and tmax ≥ 0 and tmin ≤ 1
    overlaps = (tmax >= 0) & (tmin <= tmax) & (tmin <= 1)

    return np.any(overlaps)


def get_relative_index(cur_agent_pos, target_agent_pos):
    """
    Get relative hexagon index from current agent to target agent.

    Args:
        cur_agent_pos (np.ndarray): position of current agent.
        target_agent_pos (np.ndarray): position of target agent.

    Returns:
        int: relative hexagon index.
    """
    direction = cur_agent_pos - target_agent_pos
    angle = np.arctan2(direction[1], direction[0])
    angles = np.linspace(angle - np.deg2rad(5), angle + np.deg2rad(5), 10)
    return int(np.mean(angles) % (2 * np.pi)) % 6


def nearest_points_on_obstacles(agent_pos, obstacles):
    """
    Vectorized computation of the nearest point on each rectangular obstacle.

    Args:
        agent_pos: np.array of shape (2,)
        obstacles: np.array of shape (N, 4) — [x, y, w, h]

    Returns:
        np.array of shape (N, 2) — nearest points on each obstacle
    """
    obs_x = obstacles[:, 0]
    obs_y = obstacles[:, 1]
    obs_w = obstacles[:, 2]
    obs_h = obstacles[:, 3]

    px = np.clip(agent_pos[0], obs_x, obs_x + obs_w)
    py = np.clip(agent_pos[1], obs_y, obs_y + obs_h)

    return np.stack([px, py], axis=1)


def normalize_angle(angle: float):
    return np.arctan2(np.sin(angle), np.cos(angle))


def lambda2(adj_mat):
    """Get lambda2 value from an adjacency matrix."""
    g = nx.from_numpy_array(adj_mat)

    # Compute the Laplacian matrix (as a NumPy array)
    L = nx.laplacian_matrix(g).toarray()

    # Compute all eigenvalues (ascending order)
    eigenvalues = eigvalsh(L)  # eigvalsh is for symmetric/hermitian matrices

    # Get the second smallest eigenvalue (Fiedler value)
    fiedler_value = eigenvalues[1]
    # print("Second smallest eigenvalue (Fiedler value):", fiedler_value)
    return fiedler_value


def str2bool(v):
    """Convert from string "true" or "false" to bool True or False."""
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def get_args(default_configs):
    with open(default_configs, "r") as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)

    parser = argparse.ArgumentParser()

    # Agent's parameters
    parser.add_argument(
        "--controller",
        type=str,
        help="Which controller to use (hexagon or voronoi)",
        default=configs["agents"]["controller"],
    )
    parser.add_argument(
        "--original_method",
        type=str2bool,
        nargs="?",
        const=True,
        help="If using hexagon, use original method or not",
        default=configs["agents"]["original_method"],
    )
    parser.add_argument(
        "--num_agents",
        help="Number of agents to experiment",
        type=int,
        default=configs["agents"]["num_agents"],
    )
    parser.add_argument(
        "--critical_ratio",
        help="Critical range",
        type=float,
        default=configs["agents"]["critical_ratio"],
    )
    parser.add_argument(
        "--total_time",
        help="Total simulation time",
        type=int,
        default=configs["simulation"]["total_time"],
    )
    parser.add_argument(
        "--timestep",
        help="Timestep",
        type=float,
        default=configs["simulation"]["timestep"],
    )
    parser.add_argument(
        "--agent_size",
        help="Size of an agent (agent is represented as a circle, so size is its radius",
        type=float,
        default=configs["agents"]["agent_size"],
    )
    parser.add_argument(
        "--v_max",
        help="Maximum agent's speed",
        type=float,
        default=configs["agents"]["v_max"],
    )
    parser.add_argument(
        "--tolerance",
        help="Distance threshold",
        type=float,
        default=configs["agents"]["tolerance"],
    )
    parser.add_argument(
        "--sensing_range",
        help="Agent's sensing range",
        type=float,
        default=configs["agents"]["sensing_range"],
    )
    parser.add_argument(
        "--avoidance_range",
        help="Agent's avoidance range",
        type=float,
        default=configs["agents"]["avoidance_range"],
    )

    parser.add_argument(
        "--rho",
        help="RHO parameter when using original " "method",
        type=float,
        default=configs["agents"]["rho"],
    )

    # PSO parameters
    parser.add_argument(
        "--pso_num_iterations",
        help="Number of iterations to run PSO algorithm",
        type=int,
        default=configs["agents"]["pso_num_iterations"],
    )
    parser.add_argument(
        "--pso_num_particles",
        help="Number of particles PSO algorithm uses",
        type=int,
        default=configs["agents"]["pso_num_particles"],
    )
    parser.add_argument(
        "--pso_weights",
        type=float,
        nargs="*",
        help="A list of pso weights (use spaces to separate them)",
        default=configs["agents"]["pso_weights"],
    )

    # Path planning parameters
    parser.add_argument(
        "--goal_factor",
        type=float,
        help="Goal factor for APF path planner",
        default=configs["agents"]["path_planner"]["kg"],
    )
    parser.add_argument(
        "--obstacle_factor",
        type=float,
        help="Obstacle factor for APF path planner",
        default=configs["agents"]["path_planner"]["ko"],
    )
    parser.add_argument(
        "--collision_factor",
        type=float,
        help="Collision factor for APF path planner",
        default=configs["agents"]["path_planner"]["kc"],
    )
    parser.add_argument(
        "--beta_c",
        type=float,
        help="Beta_c coefficient for collision for APF path planner",
        default=configs["agents"]["path_planner"]["beta_c"],
    )

    # Simulation parameters
    parser.add_argument(
        "--screen_size",
        type=float,
        nargs="*",
        help="Screen's size for visualization",
        default=configs["simulation"]["screen_size"],
    )
    parser.add_argument(
        "--scale",
        type=float,
        help="How much to scale",
        default=configs["simulation"]["scale"],
    )
    parser.add_argument(
        "--linewidth",
        type=int,
        help="Linewidth of drawings",
        default=configs["simulation"]["linewidth"],
    )
    parser.add_argument(
        "--show_sensing_range",
        # type=bool,
        type=str2bool,
        nargs="?",
        const=True,
        help="Show sensing range of agents",
        default=configs["simulation"]["show_sensing_range"],
    )
    parser.add_argument(
        "--show_goal",
        # type=bool,
        type=str2bool,
        nargs="?",
        const=True,
        help="Show agent's goal",
        default=configs["simulation"]["show_goal"],
    )
    parser.add_argument(
        "--show_connections",
        type=bool,
        # type=str2bool,
        nargs="?",
        const=True,
        help="Show agent's connections",
        default=configs["simulation"]["show_connections"],
    )
    parser.add_argument(
        "--show_trajectories",
        # type=bool,
        type=str2bool,
        nargs="?",
        const=True,
        help="Show agent's trajectories",
        default=configs["simulation"]["show_trajectories"],
    )

    # Results
    parser.add_argument(
        "--res_dir",
        type=str,
        help="Where to put results",
        default=configs["experiments"]["res_dir"],
    )

    parser.add_argument("--env", type=str,
                        help="Which environment", default="env0")
    parser.add_argument(
        "--area_width", type=float, help="Area width", default=None
    )  # Set a placeholder default
    parser.add_argument("--area_height", help="Area height",
                        type=float, default=None)
    parser.add_argument("--obstacles", type=float, nargs="*", default=None)
    parser.add_argument("--first_agent_pos", type=float,
                        nargs="*", default=None)

    args = parser.parse_args()
    if args.area_width is None:  # If not overridden by command line
        args.area_width = configs["environment"][args.env]["area_width"]
    if args.area_height is None:
        args.area_height = configs["environment"][args.env]["area_height"]
    if args.obstacles is None:
        args.obstacles = configs["environment"][args.env]["obstacles"]
    if args.first_agent_pos is None:
        args.first_agent_pos = configs["environment"][args.env]["first_agent_pos"]
    return args


def save_configs(args, file_path):
    """
    Saves the final configuration from the argparse namespace to a YAML file.

    Args:
        args (argparse.Namespace): The parsed command-line arguments.
        file_path (str): The path to the output YAML file.
    """
    # Convert the argparse namespace to a dictionary
    config_dict = vars(args).copy()

    # Restructure the dictionary to match the YAML format
    output_config = {
        "environment": {},  # We'll fill this in with the specific environment
        "simulation": {},
        "experiments": {},
        "agents": {"path_planner": {}},
    }

    # Populate the restructured dictionary
    output_config["environment"][config_dict["env"]] = {
        "area_width": config_dict.pop("area_width"),
        "area_height": config_dict.pop("area_height"),
        "first_agent_pos": config_dict.pop("first_agent_pos"),
        "obstacles": config_dict.pop("obstacles"),
    }

    output_config["simulation"]["scale"] = config_dict.pop("scale")
    output_config["simulation"]["screen_size"] = config_dict.pop("screen_size")
    output_config["simulation"]["linewidth"] = config_dict.pop("linewidth")
    output_config["simulation"]["total_time"] = config_dict.pop("total_time")
    output_config["simulation"]["timestep"] = config_dict.pop("timestep")
    output_config["simulation"]["show_sensing_range"] = config_dict.pop(
        "show_sensing_range"
    )
    output_config["simulation"]["show_goal"] = config_dict.pop("show_goal")
    output_config["simulation"]["show_connections"] = config_dict.pop(
        "show_connections"
    )
    output_config["simulation"]["show_trajectories"] = config_dict.pop(
        "show_trajectories"
    )

    output_config["experiments"]["res_dir"] = config_dict.pop("res_dir")

    # Path planner configs
    output_config["agents"]["path_planner"]["kg"] = config_dict.pop(
        "goal_factor")
    output_config["agents"]["path_planner"]["ko"] = config_dict.pop(
        "obstacle_factor")
    output_config["agents"]["path_planner"]["kc"] = config_dict.pop(
        "collision_factor")
    output_config["agents"]["path_planner"]["beta_c"] = config_dict.pop(
        "beta_c")

    # Other agent configs
    output_config["agents"]["controller"] = config_dict.pop("controller")
    output_config["agents"]["original_method"] = config_dict.pop(
        "original_method")
    output_config["agents"]["num_agents"] = config_dict.pop("num_agents")
    output_config["agents"]["agent_size"] = config_dict.pop("agent_size")
    output_config["agents"]["v_max"] = config_dict.pop("v_max")
    output_config["agents"]["tolerance"] = config_dict.pop("tolerance")
    output_config["agents"]["sensing_range"] = config_dict.pop("sensing_range")
    output_config["agents"]["critical_ratio"] = config_dict.pop(
        "critical_ratio")
    output_config["agents"]["avoidance_range"] = config_dict.pop(
        "avoidance_range")
    output_config["agents"]["rho"] = config_dict.pop("rho")
    output_config["agents"]["pso_num_iterations"] = config_dict.pop(
        "pso_num_iterations"
    )
    output_config["agents"]["pso_num_particles"] = config_dict.pop(
        "pso_num_particles")
    output_config["agents"]["pso_weights"] = config_dict.pop("pso_weights")

    # The remaining key is 'env' which we've already used.
    # The `pop` method removes the key from the dictionary.

    with open(file_path, "w") as f:
        yaml.dump(output_config, f, sort_keys=False)


def compute_coverage_percentage(positions, env, sensing_range):
    """
    Compute coverage percentage over a polygonal region using mesh grid.

    Args:
        positions (numpy.ndarray): positions of agents, shape (n, 2)
        polygon_vertices (numpy.ndarray): polygon vertices, shape (m, 2)

    Returns:
        float: coverage percentage (0.0 to 1.0)
    """
    resolution = 100
    polygon = Polygon(env.vertices)

    # Get bounds of polygon to limit the grid
    minx, miny, maxx, maxy = polygon.bounds
    x_vals = np.linspace(minx, maxx, resolution)
    y_vals = np.linspace(miny, maxy, resolution)
    xx, yy = np.meshgrid(x_vals, y_vals)
    grid_points = np.column_stack((xx.ravel(), yy.ravel()))  # shape: (N, 2)

    # Step 1: Filter points inside the polygon
    inside_polygon_mask = np.array(
        [polygon.contains(Point(p)) for p in grid_points])
    inside_polygon_points = grid_points[inside_polygon_mask]

    # Step 2: Compute coverage mask (robots)
    in_coverage = np.zeros(inside_polygon_points.shape[0], dtype=bool)
    for pos in positions:
        distances = np.linalg.norm(inside_polygon_points - pos, axis=1)
        in_coverage |= distances <= sensing_range

    # Step 3: Obstacle mask
    in_obs = np.zeros(inside_polygon_points.shape[0], dtype=bool)
    if len(env.obstacles) > 0:
        for obs in env.obstacles:
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
        valid_covered.sum() / (inside_polygon_points.shape[0] - in_obs.sum())
        if inside_polygon_points.shape[0] > 0
        else 0.0
    )


def plot_travel_distances(distances, log, agent_labels=None, save_dir=""):
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
    # Adjust figsize for better proportions
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the bar chart
    bars = ax.bar(
        x_pos, distances, color="skyblue"
    )  # Added a color for better aesthetics

    # Set labels and title
    # Changed to "Robot", increased font size
    ax.set_xlabel("Agent", fontsize=12)
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
        log.log(f"Plot saved to: {save_path}")
    else:
        plt.show()  # Display the plot if not saving


def plot_ld2(lambda2_values, log, save_dir=""):
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
        log.log(f"Plot saved to: {save_path}")
    else:
        plt.show()

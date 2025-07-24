import argparse
import yaml


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
        type=bool,
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
        type=bool,
        help="Show sensing range of agents",
        default=configs["simulation"]["show_sensing_range"],
    )
    parser.add_argument(
        "--show_goal",
        type=bool,
        help="Show agent's goal",
        default=configs["simulation"]["show_goal"],
    )
    parser.add_argument(
        "--show_connections",
        type=bool,
        help="Show agent's connections",
        default=configs["simulation"]["show_connections"],
    )
    parser.add_argument(
        "--show_trajectories",
        type=bool,
        help="Show agent's trajectories",
        default=configs["simulation"]["show_trajectories"],
    )

    # Results and logs
    parser.add_argument(
        "--res_dir",
        type=str,
        help="Where to put results",
        default=configs["experiments"]["res_dir"],
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        help="Where to put log files",
        default=configs["experiments"]["log_dir"],
    )
    parser.add_argument("--env", type=str, help="Which environment", default="env0")
    parser.add_argument(
        "--area_width", type=float, help="Area width", default=None
    )  # Set a placeholder default
    parser.add_argument("--area_height", help="Area height", type=float, default=None)
    parser.add_argument("--obstacles", type=float, nargs="*", default=None)
    parser.add_argument("--first_agent_pos", type=float, nargs="*", default=None)

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
    config_dict = vars(args)

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
    output_config["experiments"]["log_dir"] = config_dict.pop("log_dir")

    # Path planner configs
    output_config["agents"]["path_planner"]["kg"] = config_dict.pop("goal_factor")
    output_config["agents"]["path_planner"]["ko"] = config_dict.pop("obstacle_factor")
    output_config["agents"]["path_planner"]["kc"] = config_dict.pop("collision_factor")
    output_config["agents"]["path_planner"]["beta_c"] = config_dict.pop("beta_c")

    # Other agent configs
    output_config["agents"]["controller"] = config_dict.pop("controller")
    output_config["agents"]["original_method"] = config_dict.pop("original_method")
    output_config["agents"]["num_agents"] = config_dict.pop("num_agents")
    output_config["agents"]["agent_size"] = config_dict.pop("agent_size")
    output_config["agents"]["v_max"] = config_dict.pop("v_max")
    output_config["agents"]["tolerance"] = config_dict.pop("tolerance")
    output_config["agents"]["sensing_range"] = config_dict.pop("sensing_range")
    output_config["agents"]["avoidance_range"] = config_dict.pop("avoidance_range")
    output_config["agents"]["rho"] = config_dict.pop("rho")
    output_config["agents"]["pso_num_iterations"] = config_dict.pop(
        "pso_num_iterations"
    )
    output_config["agents"]["pso_num_particles"] = config_dict.pop("pso_num_particles")
    output_config["agents"]["pso_weights"] = config_dict.pop("pso_weights")

    # The remaining key is 'env' which we've already used.
    # The `pop` method removes the key from the dictionary.

    with open(file_path, "w") as f:
        yaml.dump(output_config, f, sort_keys=False)

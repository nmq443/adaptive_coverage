import argparse
import yaml


def get_args(default_configs):
    with open(default_configs, "r") as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--render_mode",
        type=str,
        help="Real-time visualization or not?",
        default="none",
    )

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
        "--timesteps",
        help="Number of timesteps to simulate",
        type=int,
        default=configs["simulation"]["timesteps"],
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
        "--pso_iterations",
        help="Number of iterations to run PSO algorithm",
        type=int,
        default=configs["agents"]["pso_iterations"],
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
        "--goal_factor", type=float, default=configs["agents"]["path_planner"]["kg"]
    )
    parser.add_argument(
        "--obstacle_factor", type=float, default=configs["agents"]["path_planner"]["ko"]
    )
    parser.add_argument(
        "--collision_factor",
        type=float,
        default=configs["agents"]["path_planner"]["kc"],
    )
    parser.add_argument(
        "--beta_c", type=float, default=configs["agents"]["path_planner"]["beta_c"]
    )

    # Simulation parameters
    parser.add_argument(
        "--screen_size",
        type=float,
        nargs="*",
        default=configs["simulation"]["screen_size"],
    )
    parser.add_argument("--scale", type=float, default=configs["simulation"]["scale"])
    parser.add_argument(
        "--linewidth", type=int, default=configs["simulation"]["linewidth"]
    )
    parser.add_argument(
        "--show_sensing_range",
        type=bool,
        default=configs["simulation"]["show_sensing_range"],
    )
    parser.add_argument(
        "--show_goal", type=bool, default=configs["simulation"]["show_goal"]
    )
    parser.add_argument(
        "--show_connections",
        type=bool,
        default=configs["simulation"]["show_connections"],
    )
    parser.add_argument(
        "--show_trajectories",
        type=bool,
        default=configs["simulation"]["show_trajectories"],
    )

    # Results and logs
    parser.add_argument(
        "--res_dir", type=str, default=configs["experiments"]["res_dir"]
    )
    parser.add_argument(
        "--log_dir", type=str, default=configs["experiments"]["log_dir"]
    )
    parser.add_argument("--env", type=str, default="env0")
    parser.add_argument(
        "--area_width", type=float, default=None
    )  # Set a placeholder default
    parser.add_argument("--area_height", type=float, default=None)
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

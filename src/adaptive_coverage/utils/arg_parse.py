import argparse

def get_args():
    parser = argparse.ArgumentParser()

    # Agent's parameters
    parser.add_argument("--controller", type=str, help='Which controller to use (hexagon or voronoi)', default='hexagon')
    parser.add_argument("--original_method", type=bool, help='If using hexagon, use original method or not', default=False)
    parser.add_argument("--num_agents", type=int, default=10)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--agent_size", type=float, default=0.2)
    parser.add_argument("--v_max", type=float, default=0.05)
    parser.add_argument("--tolerance", type=float, default=0.001)
    parser.add_argument("--sensing_range", type=float, default=2.5)
    parser.add_argument("--avoidance_range", type=float, default=0.05)

    parser.add_argument("--rho", type=float, default=1.0)

    # PSO parameters
    parser.add_argument("--pso_iterations", type=int, default=100)
    parser.add_argument("--pso_num_particles", type=int, default=10)
    parser.add_argument("--pso_weights", type=float, nargs='*', help='A list of pso weights (use spaces to separate them)',
                        default=[0.45, 0.3, 0.15, 0.1])

    # Path planning parameters
    parser.add_argument("--goal_factor", type=float, default=1.0)
    parser.add_argument("--obstacle_factor", type=float, default=1.5)
    parser.add_argument("--collision_factor", type=float, default=1.5)
    parser.add_argument("--beta_c", type=float, default=1.0)

    # Simulation parameters
    parser.add_argument("--screen_size", type=tuple, default=(1600, 900))
    parser.add_argument("--scale", type=float, default=40.0)
    parser.add_argument("--linewidth", type=int, default=2)
    parser.add_argument("--show_sensing_range", type=bool, default=False)
    parser.add_argument("--show_goal", type=bool, default=False)
    parser.add_argument("--show_connections", type=bool, default=False)
    parser.add_argument("--show_trajectories", type=bool, default=False)

    # Results and logs
    parser.add_argument("--res_dir", type=str, default='results')
    parser.add_argument("--log_dir", type=str, default='logs')
    parser.add_argument("--env", type=str, default='env0')

    args = parser.parse_args()
    return args

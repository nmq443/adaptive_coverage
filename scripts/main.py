import numpy as np

from adaptive_coverage.utils.arg_parse import get_args
from adaptive_coverage.simulator.data_manager import LogManager, ResultManager
from adaptive_coverage.swarms.hexagon_swarm import HexagonSwarm
from adaptive_coverage.swarms.voronoi_swarm import VoronoiSwarm
from adaptive_coverage.path_planner.apf import ArtificialPotentialField
from adaptive_coverage.environment.environment import Environment
from adaptive_coverage.simulator.simulator import Simulator
from adaptive_coverage.simulator.renderer import Renderer


def run():
    args = get_args("../configs/default_args.yaml")
    num_agents = args.num_agents
    agent_size = args.agent_size
    v_max = args.v_max
    tolerance = args.tolerance
    avoidance_range = args.avoidance_range
    sensing_range = args.sensing_range

    first_agent_pos = args.first_agent_pos

    kg = args.goal_factor
    ko = args.obstacle_factor
    kc = args.collision_factor
    beta_c = args.beta_c
    scale = args.scale
    rho = args.rho
    pso_weights = np.array(args.pso_weights)
    controller = args.controller
    original_method = args.original_method

    path_planner = ArtificialPotentialField(
        kg, ko, kc, beta_c, sensing_range, avoidance_range, agent_size
    )

    screen_size = args.screen_size

    area_width = args.area_width
    area_height = args.area_height
    obstacles = np.array(args.obstacles)

    timesteps = args.timesteps

    res_dir = args.res_dir
    log_dir = args.log_dir
    env_dir = args.env
    log_manager = LogManager(
        num_agents=num_agents,
        log_dir=log_dir,
        env_dir=env_dir,
        controller=controller,
        original_method=original_method,
    )
    result_manager = ResultManager(
        num_agents=num_agents,
        res_dir=res_dir,
        env_dir=env_dir,
        controller=controller,
        original_method=original_method,
    )

    if controller == "hexagon":
        swarm = HexagonSwarm(
            num_agents=num_agents,
            original_method=original_method,
            v_max=v_max,
            avoidance_range=avoidance_range,
            tolerance=tolerance,
            agent_size=agent_size,
            path_planner=path_planner,
            sensing_range=sensing_range,
            first_agent_pos=first_agent_pos,
            rho=rho,
            pso_weights=pso_weights,
            result_manager=result_manager,
            log_manager=log_manager,
        )
    else:
        swarm = VoronoiSwarm(
            num_agents=num_agents,
            v_max=v_max,
            avoidance_range=avoidance_range,
            tolerance=tolerance,
            agent_size=agent_size,
            path_planner=path_planner,
            sensing_range=sensing_range,
            first_agent_pos=first_agent_pos,
            result_manager=result_manager,
            log_manager=log_manager,
        )

    env = Environment(area_width, area_height, obstacles, offset=1)

    show_connections = args.show_connections
    show_goal = args.show_goal
    show_trajectories = args.show_trajectories
    show_sensing_range = args.show_sensing_range
    renderer = Renderer(
        controller=controller,
        swarm=swarm,
        env=env,
        scale=scale,
        linewidth=args.linewidth,
        show_connections=show_connections,
        show_goal=show_goal,
        show_sensing_range=show_sensing_range,
        show_trajectories=show_trajectories,
    )
    sim = Simulator(
        screen_size=screen_size,
        swarm=swarm,
        env=env,
        result_manager=result_manager,
        log_manager=log_manager,
        renderer=renderer,
        scale=scale,
        timesteps=timesteps,
    )
    sim.execute()


if __name__ == "__main__":
    run()

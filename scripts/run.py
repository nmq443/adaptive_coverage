import os
from adaptive_coverage.utils.utils import get_args, save_configs
from adaptive_coverage.simulator.data_manager import LogManager, ResultManager
from adaptive_coverage.swarms.hexagon_swarm import HexagonSwarm
from adaptive_coverage.swarms.voronoi_swarm import VoronoiSwarm
from adaptive_coverage.path_planner.apf import ArtificialPotentialField
from adaptive_coverage.environment.environment import Environment
from adaptive_coverage.simulator.simulator import Simulator
from adaptive_coverage.simulator.renderer import Renderer


def run():
    args = get_args("configs/default_args.yaml")

    path_planner = ArtificialPotentialField(
        kg=args.goal_factor,
        ko=args.obstacle_factor,
        kc=args.collision_factor,
        beta_c=args.beta_c,
        sensing_range=args.sensing_range,
        avoidance_range=args.sensing_range,
        agent_size=args.agent_size,
    )

    result_manager = ResultManager(
        num_agents=args.num_agents,
        res_dir=args.res_dir,
        env_dir=args.env,
        controller=args.controller,
        original_method=args.original_method,
    )
    log_manager = LogManager(result_manager)

    if args.controller == "hexagon":
        swarm = HexagonSwarm(
            num_agents=args.num_agents,
            original_method=args.original_method,
            v_max=args.v_max,
            avoidance_range=args.avoidance_range,
            tolerance=args.tolerance,
            agent_size=args.agent_size,
            path_planner=path_planner,
            sensing_range=args.sensing_range,
            first_agent_pos=args.first_agent_pos,
            rho=args.rho,
            timestep=args.timestep,
            total_time=args.total_time,
            pso_weights=args.pso_weights,
            pso_num_particles=args.pso_num_particles,
            pso_num_iterations=args.pso_num_iterations,
            result_manager=result_manager,
            log_manager=log_manager,
        )
    else:
        swarm = VoronoiSwarm(
            num_agents=args.num_agents,
            v_max=args.v_max,
            avoidance_range=args.avoidance_range,
            tolerance=args.tolerance,
            agent_size=args.agent_size,
            path_planner=path_planner,
            timestep=args.timestep,
            total_time=args.total_time,
            sensing_range=args.sensing_range,
            first_agent_pos=args.first_agent_pos,
            result_manager=result_manager,
            log_manager=log_manager,
        )

    env = Environment(args.area_width, args.area_height, args.obstacles, offset=1)

    sim = Simulator(
        screen_size=args.screen_size,
        swarm=swarm,
        env=env,
        result_manager=result_manager,
        log_manager=log_manager,
        scale=args.scale,
        total_time=args.total_time,
        timestep=args.timestep,
    )

    sim.execute()
    save_configs(args, os.path.join(result_manager.res_dir, "configs.yaml"))
    render(args, log_manager, result_manager)


def render(args, log_manager, result_manager):
    env = Environment(args.area_width, args.area_height, args.obstacles, offset=1)
    renderer = Renderer(
        screen_size=args.screen_size,
        trajectories_filepath=result_manager.swarm_data_filepath,
        controller=args.controller,
        agent_size=args.agent_size,
        sensing_range=args.sensing_range,
        result_manager=result_manager,
        log_manager=log_manager,
        env=env,
        scale=args.scale,
        linewidth=args.linewidth,
        show_connections=args.show_connections,
        show_goal=args.show_goal,
        show_sensing_range=args.show_sensing_range,
        show_trajectories=args.show_trajectories,
    )
    renderer.run()


if __name__ == "__main__":
    run()

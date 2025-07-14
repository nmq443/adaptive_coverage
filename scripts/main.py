import argparse
import numpy as np

from adaptive_coverage.simulator.data_manager import LogManager, ResultManager
from adaptive_coverage.swarms.hexagon_swarm import HexagonSwarm
from adaptive_coverage.path_planner.apf import ArtificialPotentialField
from adaptive_coverage.environment.environment import Environment
from adaptive_coverage.simulator.simulator import Simulator
from adaptive_coverage.simulator.renderer import Renderer
from adaptive_coverage.swarms.voronoi_swarm import VoronoiSwarm


def run_hexagon():
    num_agents = 20
    agent_size = 0.2
    v_max = 0.05
    tolerance = 0.001
    avoidance_range = 0.05
    sensing_range = 2.0
    first_agent_pos = np.array([5, 5])
    kg = 1.
    ko = 0.
    kc = 0.
    beta_c = 1.
    scale = 40
    rho = 0.5
    pso_weights = np.array([0.45, 0.3, 0.15, 0.1])
    controller = 'hexagon'
    original_method = True

    path_planner = ArtificialPotentialField(kg, ko, kc, beta_c, sensing_range, avoidance_range, agent_size)

    screen_size = (1600, 900)

    area_width = 20
    area_height = 10
    offset = 0.1
    vertices = np.array(
        [
            [0 + offset, 0 + offset],
            [area_width - offset, 0 + offset],
            [area_width - offset, area_height - offset],
            [0 + offset, area_height - offset],
        ],
        dtype=float,
    )

    obstacles = np.array([])

    timesteps = 2000

    res_dir = "results"
    log_dir = "log"
    env_dir = "env0"
    log_manager = LogManager(
        num_agents=num_agents,
        log_dir=log_dir,
        env_dir=env_dir,
        controller=controller,
        original_method=original_method
    )
    result_manager = ResultManager(
        num_agents=num_agents,
        res_dir=res_dir,
        env_dir=env_dir,
        controller=controller,
        original_method=original_method
    )

    parser = argparse.ArgumentParser()
    parser.parse_args()
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
    env = Environment(vertices, obstacles)
    renderer = Renderer(
        swarm=swarm,
        env=env,
        scale=scale,
        show_goal=True
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

def run_voronoi():
    screen_size = (1600, 900)
    area_width = 30
    area_height = 20
    offset = 0.1
    vertices = np.array(
        [
            [0 + offset, 0 + offset],
            [area_width - offset, 0 + offset],
            [area_width - offset, area_height - offset],
            [0 + offset, area_height - offset],
        ],
        dtype=float,
    )
    obstacles = np.array([])
    scale = 50

    timesteps = 1000
    num_agents = 20
    agent_size = 0.2
    v_max = 0.05
    tolerance = 0.001
    avoidance_range = 0.05
    sensing_range = 2.5
    first_agent_pos = np.array([5, 2.5])
    kg = 1.
    ko = 0.
    kc = 0.
    beta_c = 1.
    controller = 'voronoi'
    original_method = True

    path_planner = ArtificialPotentialField(kg, ko, kc, beta_c, sensing_range, avoidance_range, agent_size)

    res_dir = "results"
    log_dir = "log"
    env_dir = "env0"
    log_manager = LogManager(
        num_agents=num_agents,
        log_dir=log_dir,
        env_dir=env_dir,
        controller=controller,
        original_method=original_method
    )
    result_manager = ResultManager(
        num_agents=num_agents,
        res_dir=res_dir,
        env_dir=env_dir,
        controller=controller,
        original_method=original_method
    )

    parser = argparse.ArgumentParser()
    parser.parse_args()
    swarm = VoronoiSwarm(
        num_agents=num_agents,
        agent_size=agent_size,
        path_planner=path_planner,
        sensing_range=sensing_range,
        v_max=v_max,
        avoidance_range=avoidance_range,
        tolerance=tolerance,
        first_agent_pos=first_agent_pos,
        result_manager=result_manager,
        log_manager=log_manager
    )
    env = Environment(vertices, obstacles)
    renderer = Renderer(swarm, env, scale, controller=controller)
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

# run_hexagon()
run_voronoi()
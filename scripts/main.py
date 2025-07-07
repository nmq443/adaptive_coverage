import argparse
import numpy as np
from adaptive_coverage.swarms.hexagon_swarm import HexagonSwarm
from adaptive_coverage.path_planner.apf import ArtificialPotentialField
from adaptive_coverage.environment.environment import Environment
from adaptive_coverage.simulator.simulator import Simulator

if __name__ == "__main__":
    num_agents = 10
    agent_size = 0.5
    avoidance_range = 0.05
    sensing_range = 2.0
    first_agent_pos = np.array([5, 5])
    kg = 1.
    ko = 2.
    kc = 2.
    path_planner = ArtificialPotentialField(kg, ko, kc, avoidance_range, agent_size)

    screen_size = (1600, 900)
    offset = 0.1
    vertices = np.array(
        [
            [0 + offset, 0 + offset],
            [screen_size[0] - offset, 0 + offset],
            [screen_size[0] - offset, screen_size[1] - offset],
            [0 + offset, screen_size[1] - offset],
        ],
        dtype=float,
    )
    obstacles = np.array([])

    parser = argparse.ArgumentParser()
    parser.parse_args()
    swarm = HexagonSwarm(
        num_agents=num_agents,
        agent_size=agent_size,
        path_planner=path_planner,
        sensing_range=sensing_range,
        first_agent_pos=first_agent_pos,
    )
    env = Environment(vertices, obstacles)
    sim = Simulator(swarm, env)
    sim.execute()

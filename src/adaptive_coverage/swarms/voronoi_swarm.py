import numpy as np
from adaptive_coverage.swarms.swarm import Swarm
from adaptive_coverage.agents.cvt.voronoi_agent import VoronoiAgent
from adaptive_coverage.utils.utils import lambda2
from typing import Union, Optional


class VoronoiSwarm(Swarm):
    def __init__(self, *args, critical_ratio, **kwargs):
        super().__init__(*args, **kwargs)
        self.critical_ratio = critical_ratio
        self.generators: Union[list, np.ndarray] = []

    def init_agents(self):
        """Initialize all agents in a grid-like formation or in random positions."""
        if self.random_init:
            pos = np.random.rand(self.num_agents, 2) + self.first_agent_pos
            for i, p in enumerate(pos):
                self.agents.append(
                    VoronoiAgent(
                        index=i,
                        init_pos=p,
                        size=self.agent_size,
                        critical_ratio=self.critical_ratio,
                        sensing_range=self.sensing_range,
                        timestep=self.timestep,
                        path_planner=self.path_planner,
                    )
                )
        else:
            for i in range(self.num_rows):
                for j in range(self.num_cols):
                    x = self.first_agent_pos[0] + j * self.dist_btw_agents
                    y = (
                        self.first_agent_pos[1]
                        + (self.num_rows - i - 1) * self.dist_btw_agents
                    )
                    init_pos = np.array([x, y])
                    self.agents.append(
                        VoronoiAgent(
                            index=i * self.num_cols + j,
                            init_pos=init_pos,
                            size=self.agent_size,
                            critical_ratio=self.critical_ratio,
                            sensing_range=self.sensing_range,
                            timestep=self.timestep,
                            path_planner=self.path_planner,
                        )
                    )
        self.generators = np.array([agent.pos for agent in self.agents])

    def step(self, env, current_step, penalty_flag=0):
        super().step(env, current_step, penalty_flag)

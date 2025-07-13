import numpy as np
from collections import deque
from adaptive_coverage.swarms.swarm import Swarm
from adaptive_coverage.agents.hexagon.hexagon_agent import HexagonAgent
from adaptive_coverage.utils.evaluate import lamda2


class HexagonSwarm(Swarm):
    def __init__(self, *args, rho, pso_weights, original_method=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.landmarks = deque([])
        self.original_method = original_method
        self.rho = rho
        self.pso_weights = pso_weights

    def init_agents(self):
        if self.random_init:
            random_positions = (
                    np.random.rand(self.num_agents, 2) * self.agent_spread + self.first_agent_pos
            )
            for index in range(self.num_agents):
                self.agents.append(
                    HexagonAgent(
                        index=index,
                        init_pos=random_positions[index],
                        size=self.agent_size,
                        path_planner=self.path_planner,
                        sensing_range=self.sensing_range,
                        original_method=self.original_method,
                        v_max=self.v_max,
                        avoidance_range=self.avoidance_range,
                        tolerance=self.tolerance,
                        rho=self.rho,
                        pso_weights=self.pso_weights,
                        result_manager=self.result_manager,
                    )
                )
        else:
            for i in range(self.num_rows):
                for j in range(self.num_cols):
                    x = self.first_agent_pos[0] + j * self.dist_btw_agents
                    y = self.first_agent_pos[1] + (self.num_rows - i - 1) * self.dist_btw_agents
                    init_pos = np.array([x, y])
                    self.agents.append(HexagonAgent(
                        index=i * self.num_cols + j,
                        init_pos=init_pos,
                        size=self.agent_size,
                        v_max=self.v_max,
                        avoidance_range=self.avoidance_range,
                        tolerance=self.tolerance,
                        sensing_range=self.sensing_range,
                        path_planner=self.path_planner,
                        rho=self.rho,
                        pso_weights=self.pso_weights,
                        result_manager=self.result_manager,
                    ))
        self.determine_root(-1, self.agents[-1].pos)

    def determine_root(self, agent_id, agent_goal):
        self.agents[agent_id].set_state("occupied")

    def step(self, env):
        if len(self.agents) > 0:
            order = np.random.permutation(len(self.agents))
            for i in order:
                self.agents[i].step(self.landmarks, self.agents, env)
            ld2 = lamda2(self.agents)
            self.ld2s.append(ld2)
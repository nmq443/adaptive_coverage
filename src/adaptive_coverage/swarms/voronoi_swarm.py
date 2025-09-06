import numpy as np
from adaptive_coverage.swarms.swarm import Swarm
from adaptive_coverage.agents.cvt.voronoi_agent import VoronoiAgent
from adaptive_coverage.utils.utils import lambda2


class VoronoiSwarm(Swarm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.generators: list = []

    def init_agents(self, ref_pos=None):
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
                        sensing_range=self.sensing_range,
                        path_planner=self.path_planner,
                    )
                )
        self.generators = [agent.pos for agent in self.agents]
        self.generators = np.array(self.generators)

    def step(self, env, timestep, current_step):
        super().step(env, timestep, current_step)
        # if len(self.agents) > 0:
        #     order = np.random.permutation(len(self.agents))
        #     for i in order:
        #         self.agents[i].step(self.agents, env, timestep)
        #     self.update_adj_mat()
        #     ld2 = lambda2(self.adjacency_matrix)
        #     self.ld2s.append(ld2)

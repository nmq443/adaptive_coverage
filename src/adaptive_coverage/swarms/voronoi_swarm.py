import numpy as np
from adaptive_coverage.swarms.swarm import Swarm
from adaptive_coverage.agents.cvt.voronoi_agent import VoronoiAgent
from adaptive_coverage.agents.cvt.lloyd import compute_voronoi_diagrams
from adaptive_coverage.utils.utils import draw_voronoi


class VoronoiSwarm(Swarm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.generators: list = []

    def init_agents(self, ref_pos=None):
        if self.random_init and ref_pos is not None:
            random_positions = (
                    np.random.rand(self.num_agents, 2) * self.agent_spread + ref_pos
            )
            for index in range(self.num_agents):
                self.agents.append(
                    VoronoiAgent(
                        index=index,
                        init_pos=random_positions[index],
                        size=self.agent_size,
                        path_planner=self.path_planner,
                        sensing_range=self.sensing_range,
                        result_manager=self.result_manager,
                    )
                )
        else:
            for i in range(self.num_rows):
                for j in range(self.num_cols):
                    x = self.first_agent_pos[0] + j * self.dist_btw_agents
                    y = self.first_agent_pos[1] + (self.num_rows - i - 1) * self.dist_btw_agents
                    init_pos = np.array([x, y])
                    self.agents.append(VoronoiAgent(
                        index=i * self.num_cols + j,
                        init_pos=init_pos,
                        size=self.agent_size,
                        sensing_range=self.sensing_range,
                        path_planner=self.path_planner,
                        result_manager=self.result_manager,
                    ))
        self.generators = [agent.pos for agent in self.agents]
        self.generators = np.array(self.generators)

    def render(self, surface, env, font, timestep, scale):
        if len(self.agents) > 0:
            for agent in self.agents:
                agent.render(surface, font, self.agents, timestep, scale)
                self.generators[agent.index] = agent.pos
            vor = compute_voronoi_diagrams(self.generators, env)
            draw_voronoi(vor, surface, scale)

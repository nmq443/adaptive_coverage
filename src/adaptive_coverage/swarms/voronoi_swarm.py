import numpy as np
from swarm import Swarm
from adaptive_coverage.agents.cvt.voronoi_agent import VoronoiAgent
from adaptive_coverage.agents.cvt.lloyd import compute_voronoi_diagrams
from adaptive_coverage.utils.utils import draw_voronoi


class VoronoiSwarm(Swarm):
    def __init__(self, num_agents, agent_size, path_planner, sensing_range, first_agent_pos, random_init,
                 dist_btw_agents, agent_spread):
        super().__init__(self, num_agents, agent_size, path_planner, sensing_range, first_agent_pos, random_init,
                         dist_btw_agents, agent_spread)
        self.generators: list = []
        assert num_agents % 5 == 0  # for now we hardcode
        self.num_rows = num_agents / 5
        self.num_cols = num_agents / self.num_cols

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
                    )
                )
        else:
            for i in range(self.num_rows):
                for j in range(self.num_cols):
                    x = 0 + j * self.dist_btw_agents
                    y = 0 + (self.num_rows - i - 1) * self.dist_btw_agents
                    init_pos = np.array([x, y])
                    self.agents.append(VoronoiAgent(
                        index=i * self.num_cols + j,
                        init_pos=init_pos,
                        size=self.agent_size,
                        sensing_range=self.sensing_range,
                        path_planner=self.path_planner,
                    ))
        self.generators = [agent.pos for agent in self.agents]
        self.generators = np.array(self.generators)

    def render(self, surface, env, font, timestep):
        if len(self.agents) > 0:
            for agent in self.agents:
                agent.render(surface, font, self.agents, timestep)
                self.generators[agent.index] = agent.pos
            vor = compute_voronoi_diagrams(self.generators, env)
            draw_voronoi(vor, surface)

import os
import numpy as np
from adaptive_coverage.utils.evaluate import lamda2
from adaptive_coverage.agents.cvt.lloyd import compute_voronoi_diagrams
from adaptive_coverage.utils.utils import draw_voronoi


class Swarm:
    def __init__(self, num_agents, agent_size, path_planner, sensing_range, first_agent_pos, random_init=False, dist_btw_agents=0.7, agent_spread=0.05):
        self.num_agents = num_agents
        self.agent_size = agent_size
        self.first_agent_pos = first_agent_pos
        self.dist_btw_agents = dist_btw_agents
        self.path_planner = path_planner
        self.agent_spread = agent_spread
        self.sensing_range = sensing_range
        self.agents = []
        self.ld2s = []
        self.random_init = random_init
        assert num_agents % 5 == 0  # for now we hardcode
        self.num_rows = int(num_agents / 5)
        self.num_cols = int(num_agents / self.num_rows)

    def init_agents(self, ref_pos=None):
        pass

    def step(self, env):
        pass
        if len(self.agents) > 0:
            order = np.random.permutation(len(self.agents))
            for i in order:
                self.agents[i].step(self.agents, env)
            ld2 = lamda2(self.agents)
            self.ld2s.append(ld2)

    def get_travel_distance(self):
        """Save travel distance of all agents."""
        distances = []
        for agent in self.agents:
            distance = agent.get_travel_distance()
            distances.append(distance)
        return np.array(distances)

    def save_data(self, res_dir: str):
        # save poses
        datas = []
        for agent in self.agents:
            data = []
            data.append(agent.traj[0])  # first pose
            data.append(agent.pos)  # last pose
            datas.append(data)
        datas = np.array(datas)
        save_file = os.path.join(res_dir, "swarm_data.npy")
        with open(save_file, "wb") as f:
            np.save(f, datas)

        # save travel distances
        distances = self.get_travel_distance()
        save_file = os.path.join(res_dir, "travel_distances.npy")
        with open(save_file, "wb") as f:
            np.save(f, distances)

        # save ld2s
        ld2s = np.array(self.ld2s)
        save_file = os.path.join(res_dir, "ld2s.npy")
        with open(save_file, "wb") as f:
            np.save(f, ld2s)

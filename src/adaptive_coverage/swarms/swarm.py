import numpy as np


class Swarm:
    def __init__(
        self,
        num_agents,
        agent_size,
        path_planner,
        sensing_range,
        first_agent_pos,
        result_manager,
        log_manager,
        v_max,
        avoidance_range,
        tolerance,
        random_init=False,
        dist_btw_agents=0.7,
        agent_spread=0.05,
    ):
        self.num_agents = num_agents
        self.agent_size = agent_size
        self.first_agent_pos = first_agent_pos
        self.dist_btw_agents = dist_btw_agents
        self.path_planner = path_planner
        self.agent_spread = agent_spread
        self.sensing_range = sensing_range
        self.v_max = v_max
        self.avoidance_range = avoidance_range
        self.tolerance = tolerance
        self.agents = []
        self.ld2s = []
        self.adjacency_matrix = np.zeros((self.num_agents, self.num_agents))
        self.random_init = random_init
        self.result_manager = result_manager
        self.log_manager = log_manager
        assert num_agents % 5 == 0  # for now we hardcode
        self.num_rows = int(num_agents / 5)
        self.num_cols = int(num_agents / self.num_rows)

    def init_agents(self, ref_pos=None):
        pass

    def update_adj_mat(self):
        self.adjacency_matrix.fill(0)
        sr2 = self.agents[0].sensing_range ** 2
        for i in range(self.num_agents):
            for j in range(i + 1, self.num_agents):
                if i == j:
                    continue
                diff = self.agents[i].pos - self.agents[j].pos
                dist2 = np.dot(diff, diff)
                if dist2 <= sr2:
                    self.adjacency_matrix[i][j] = self.adjacency_matrix[j][i] = np.sqrt(
                        dist2
                    )

    def step(self):
        pass

    def get_travel_distance(self):
        """Save travel distance of all agents."""
        distances = []
        for agent in self.agents:
            distance = agent.get_travel_distance()
            distances.append(distance)
        return np.array(distances)

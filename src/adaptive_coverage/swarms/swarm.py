import numpy as np
from adaptive_coverage.utils.utils import lambda2, ray_intersects_aabb, compute_coverage_percentage


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
        total_time,
        timestep,
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

        self.total_time = total_time
        self.timestep = timestep
        # (num_agents, num_timesteps, (x, y, theta, goal_x, goal_y, dx, dy, speed, penalty_flag)
        self.state = np.zeros(
            (self.num_agents, int(self.total_time / self.timestep), 11)
        )

    def init_agents(self):
        pass

    def update_adj_mat(self, env):
        """
        Update adjacency matrix.
        """
        self.adjacency_matrix.fill(0)
        sr2 = self.agents[0].sensing_range ** 2
        for i in range(self.num_agents):
            for j in range(i + 1, self.num_agents):
                if i == j:
                    continue
                if ray_intersects_aabb(self.agents[i].pos, self.agents[j].pos, env.obstacles):
                    continue
                diff = self.agents[i].pos - self.agents[j].pos
                dist2 = np.dot(diff, diff)
                if dist2 <= sr2:
                    self.adjacency_matrix[i][j] = self.adjacency_matrix[j][i] = 1

    def update_state(self, agent_index, current_step, state):
        """
        Update current agent's state.

        Args:
            agent_index: index of agent to be updated.
            current_step: current time step.
            state: updated state.
        """
        self.state[agent_index, current_step] = state

    def step(self, env, current_step, penalty_flag=0):
        if len(self.agents) > 0:
            # order = np.random.permutation(len(self.agents))
            order = np.arange(len(self.agents))
            for i in order:
                self.agents[i].step(self.agents, env)
                pos = self.agents[i].get_pos()
                vel = self.agents[i].get_vel()
                next_pos = pos + vel * self.timestep
                speed = self.agents[i].get_speed()
                theta = self.agents[i].get_theta()
                goal = self.agents[i].get_goal()
                state = np.array(
                    [pos[0], pos[1], theta, goal[0], goal[1],
                        vel[0], vel[1], speed, penalty_flag, next_pos[0], next_pos[1]]
                )
                self.update_state(
                    agent_index=i, current_step=current_step, state=state)
            self.update_adj_mat(env)
            ld2 = lambda2(self.adjacency_matrix)
            self.ld2s.append(ld2)

    def get_travel_distance(self):
        """Save travel distance of all agents."""
        distances = []
        for agent in self.agents:
            distance = agent.get_travel_distance(self.state)
            distances.append(distance)
        return np.array(distances)

    def get_coverage_percentage(self, env):
        positions = self.state[:, -1, :2]
        sensing_range = self.sensing_range
        return compute_coverage_percentage(positions, env, sensing_range)

    def compute_environment_area(self, env):
        """
        Compute polygon area of the environment using the Shoelace formula.
        This supports both convex and non-convex polygons.
        """
        vertices = np.array(env.vertices)
        x, y = vertices[:, 0], vertices[:, 1]
        return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    def compute_total_obstacle_area(self, env):
        """
        Compute the total rectangular obstacle area.
        Each obstacle is given as (x, y, width, height).
        """
        if len(env.obstacles) == 0:
            return 0.0
        return float(np.sum(env.obstacles[:, 2] * env.obstacles[:, 3]))

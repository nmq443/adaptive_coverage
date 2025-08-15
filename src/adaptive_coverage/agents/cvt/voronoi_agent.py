import numpy as np
from shapely.geometry import Point, LineString
from adaptive_coverage.agents.agent import Agent
from adaptive_coverage.agents.cvt.lloyd import lloyd
from adaptive_coverage.utils.utils import (
    ray_intersects_aabb,
)


class VoronoiAgent(Agent):
    def __init__(self, *args, valid_ratio=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.valid_range = self.sensing_range * valid_ratio
        self.critical_range = self.sensing_range * 0.75
        self.eps = self.sensing_range - self.critical_range

    def get_critical_agents(self, agents, env):
        critical_agents = []
        for agent in agents:
            if agent.index != self.index:
                if self.is_critical_agent(agent, agents, env):
                    critical_agents.append(agent.index)
        # agents_positions = np.array(
        #     [agent.pos for agent in agents if agent.index != self.index]
        # )
        # distances = np.linalg.norm(agents_positions - self.pos, axis=1)
        # critical_agents = (self.critical_range <= distances) & (
        #     distances <= self.sensing_range
        # )
        return critical_agents

    def is_critical_agent(self, agent, agents, env):
        if agent.index == self.index:
            return False
        rij = np.linalg.norm(agent.pos - self.pos)
        if rij < self.critical_range or rij > self.sensing_range - self.size:
            return False
        if ray_intersects_aabb(self.pos, agent.pos, env.obstacles):
            return False
        for other_agent in agents:
            if agent.index != other_agent.index and other_agent.index != self.index:
                if ray_intersects_aabb(self.pos, agent.pos, env.obstacles):
                    continue
                if ray_intersects_aabb(other_agent.pos, agent.pos, env.obstacles):
                    continue
                di = np.linalg.norm(agent.pos - self.pos)
                dj = np.linalg.norm(agent.pos - other_agent.pos)
                if (
                    di < self.critical_range - self.size
                    and dj < self.critical_range - self.size
                ):
                    return False
        return True

    def network_maintainence(self, next_pos, timestep, env, agents):
        if len(env.obstacles) <= 0:
            return True
        neighbors = []
        for agent in agents:
            if agent.index != self.index:
                d = np.linalg.norm(agent.pos - self.pos)
                if d <= self.sensing_range and not ray_intersects_aabb(
                    agent.pos, self.pos, env.obstacles
                ):
                    neighbors.append(agent.index)
        if len(neighbors) > 0:
            can_connect = False
            can_connects = [False for i in range(len(neighbors))]
            for idx, neighbor in enumerate(neighbors):
                p_i_new = next_pos
                p_j = agents[neighbor].pos
                d = timestep * self.v_max

                # 3 dangerous points
                #            j1
                #   p_i  j3  pj  j0
                #            j2
                v_ij = p_j - p_i_new
                u_ij = v_ij / np.linalg.norm(v_ij)
                j0 = p_j + d * u_ij
                j3 = p_j - d * u_ij
                u_perp = np.array([-u_ij[1], u_ij[0]])
                # Position of j1 and j2
                j1 = p_j + d * u_perp
                j2 = p_j - d * u_perp
                dangerous_points = [j0, j1, j2, j3]

                for point in dangerous_points:
                    d = np.linalg.norm(point - p_i_new)
                    if d <= self.sensing_range - self.size and not ray_intersects_aabb(
                        p_i_new, point, env.obstacles
                    ):
                        can_connect = True
                can_connects[idx] = can_connect
            can_connects = np.array(can_connects)
            return can_connects.all()
        return False

    def step(self, agents, env, timestep):
        super().step(timestep=timestep)
        if self.goal is not None and not self.terminated(self.goal):
            critical_agents = self.get_critical_agents(agents, env)
            if len(critical_agents) > 0:
                distances = []
                for agent in critical_agents:
                    d = np.linalg.norm(agents[agent].pos - self.pos)
                    distances.append(self.sensing_range - d)
                epsi = min(distances)
                desired_v = self.v_max * (epsi / (2 * timestep))
            else:
                desired_v = self.v_max * (self.eps / (2 * timestep))
            vel = self.path_planner.total_force(
                self.pos, self.goal, self.index, agents, env.obstacles
            )
            v = np.linalg.norm(vel)
            vel = vel / v * desired_v
            next_pos = self.pos + vel * timestep
            if not self.network_maintainence(
                next_pos=next_pos, timestep=timestep, env=env, agents=agents
            ):
                desired_v = 0
            self.move_to_goal(
                self.goal, agents, env.obstacles, timestep, desired_v=desired_v
            )
        else:
            self.goal = lloyd(self, agents, env)

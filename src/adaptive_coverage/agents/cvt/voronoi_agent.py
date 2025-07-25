import numpy as np
from adaptive_coverage.agents.agent import Agent
from adaptive_coverage.agents.cvt.lloyd import lloyd
from adaptive_coverage.utils.utils import ray_intersects_aabb


class VoronoiAgent(Agent):
    def __init__(self, *args, valid_ratio=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.valid_range = self.sensing_range * valid_ratio
        self.critical_range = self.sensing_range * 0.75
        self.eps = self.sensing_range - self.critical_range

    def get_critical_agents(self, agents):
        critical_agents = []
        for agent in agents:
            if agent.index != self.index:
                if self.is_critical_agent(agent, agents):
                    critical_agents.append(agent.index)
        # agents_positions = np.array(
        #     [agent.pos for agent in agents if agent.index != self.index]
        # )
        # distances = np.linalg.norm(agents_positions - self.pos, axis=1)
        # critical_agents = (self.critical_range <= distances) & (
        #     distances <= self.sensing_range
        # )
        return critical_agents

    def is_critical_agent(self, other, agents, env):
        if other.index == self.index:
            return False
        rij = np.linalg.norm(other.pos - self.pos)
        if rij < self.critical_range or rij > self.sensing_range:
            return False
        if ray_intersects_aabb(self.pos, other.pos, env.obstacles):
            return False
        for agent in agents:
            if agent.index != other.index and agent.index != self.index:
                if ray_intersects_aabb(self.pos, agent.pos, env.obstacles):
                    continue
                if ray_intersects_aabb(other.pos, agent.pos, env.obstacles):
                    continue
                di = np.linalg.norm(agent.pos - self.pos)
                dj = np.linalg.norm(agent.pos - other.pos)
                if di <= self.critical_range and dj <= self.critical_range:
                    return False
        return True

    def step(self, agents, env, timestep):
        super().step(timestep=timestep)
        if self.goal is not None and not self.terminated(self.goal):
            critical_agents = self.get_critical_agents(agents)
            if len(critical_agents) > 0:
                distances = []
                for agent in critical_agents:
                    d = np.linalg.norm(agents[agent].pos - self.pos)
                    distances.append(self.sensing_range - d)
                epsi = min(distances)
                desired_v = self.v_max * (epsi / (2 * timestep))
            else:
                desired_v = self.v_max * (self.eps / (2 * timestep))
            self.move_to_goal(
                self.goal, agents, env.obstacles, timestep, desired_v=desired_v
            )
        else:
            self.goal = lloyd(self, agents, env)

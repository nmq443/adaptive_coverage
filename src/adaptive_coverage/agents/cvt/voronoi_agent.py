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
        self.critical_range = self.sensing_range * 0.6
        self.eps = self.sensing_range - self.critical_range

    def get_critical_agents(self, agents, env):
        """
        Get critical agents of current agent.

        Args:
            agents: list of all agents
            env: simulation environment

        Returns:
            list[int]: list of critical agents' indices
        """
        critical_agents = []
        for agent in agents:
            if agent.index != self.index:
                if self.is_critical_agent(agent, agents, env):
                    critical_agents.append(agent.index)
        return critical_agents

    def is_critical_agent(self, agent, agents, env):
        """
        Check if an agent is a critical agent of this agent.

        Args:
            agent: agent to check
            agents: list of all agents
            env: simulation environment

        Returns:
            bool: if the agent is a critical agent.
        """
        if agent.index == self.index:
            return False
        rij = np.linalg.norm(agent.pos - self.pos)
        if rij <= self.critical_range or rij >= self.sensing_range:
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
                    di < self.critical_range
                    and dj < self.critical_range
                ):
                    return False
        return True

    def mobility_constraint(self, critical_agents, agents, env, timestep):
        if len(critical_agents) <= 0:
            return self.v_max * self.eps / (2 * timestep)

        desired_velocity_vector = self.goal - self.pos
        next_pos = self.pos + timestep * desired_velocity_vector
        p_i_new = next_pos
        p_j = agents[critical_agents[0]].pos
        # d = self.v_max * timestep * 1.5
        d = (self.sensing_range - np.linalg.norm(self.pos - p_j)) / 2

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
        can_connect = False

        for point in dangerous_points:
            dist = np.linalg.norm(point - p_i_new)
            if dist < self.sensing_range and not ray_intersects_aabb(
                p_i_new, point, env.obstacles
            ):
                can_connect = True
        if not can_connect:
            return 0

        epsi = []
        for critical_id in critical_agents:
            distance = np.linalg.norm(self.pos - agents[critical_id].pos)
            epsi.append(self.sensing_range - distance)

        return self.v_max * min(epsi) / (2 * timestep)

    def find_best_connectivity(self, agents, env, timestep):
        """
        Finds the best neighbor to maintain a critical connection with,
        based on the minimum angle to the desired velocity vector.
        """
        critical_agents = self.get_critical_agents(agents, env)
        desired_velocity_vector = self.goal - self.pos

        if len(critical_agents) <= 0:
            return None

        min_angle = np.pi * 2
        best_neighbor_id = -1

        for neighbor_id in critical_agents:
            neighbor = agents[neighbor_id]
            relative_position_vector = neighbor.pos - self.pos

            # Use dot product and magnitudes to calculate angle (more robust than arctan2)
            dot_product = np.dot(desired_velocity_vector,
                                 relative_position_vector)
            mag_velocity = np.linalg.norm(desired_velocity_vector)
            mag_relative_pos = np.linalg.norm(relative_position_vector)

            # Avoid division by zero
            if mag_velocity > 1e-6 and mag_relative_pos > 1e-6:
                angle = np.arccos(
                    np.clip(dot_product / (mag_velocity *
                            mag_relative_pos), -1, 1)
                )
                if angle < min_angle:
                    min_angle = angle
                    best_neighbor_id = neighbor_id

        return best_neighbor_id

    def step(self, agents, env, timestep):
        super().step(timestep=timestep)
        if self.goal is not None and not self.terminated(self.goal):
            critical_agents = self.get_critical_agents(agents, env)
            desired_v = self.v_max
            if len(critical_agents) > 0:
                best_connect = self.find_best_connectivity(
                    agents, env, timestep)
                if best_connect is not None:
                    desired_v = self.mobility_constraint(
                        critical_agents=[
                            best_connect], agents=agents, env=env, timestep=timestep
                    )

            self.move_to_goal(
                self.goal, agents, env.obstacles, timestep, desired_v=desired_v
            )
        else:
            self.goal = lloyd(self, agents, env)

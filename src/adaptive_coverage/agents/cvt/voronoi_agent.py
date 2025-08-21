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

    def mobility_constraint(self, agents, env, timestep):
        critical_agents = self.get_critical_agents(agents, env)

        if len(critical_agents) <= 0:
            return self.v_max * self.eps / (2 * timestep)

        epsi = []
        for critical_id in critical_agents:
            distance = np.linalg.norm(self.pos - agents[critical_id].pos)
            epsi.append(self.sensing_range - distance)

        # epsilon_i = max(epsilon_i - self.eps, 0)
        return self.v_max * min(epsi) / (2 * timestep)

    def find_best_connectivity(self, agents, env, desired_velocity_vector, timestep):
        """
        Finds the best neighbor to maintain a critical connection with,
        based on the minimum angle to the desired velocity vector.
        """
        critical_agents = self.get_critical_agents(agents, env)

        if len(critical_agents) <= 0:
            return None

        min_angle = np.pi * 2
        best_neighbor_id = -1

        for neighbor_id in critical_agents:
            neighbor = agents[neighbor_id]
            relative_position_vector = neighbor.pos - self.pos

            # Use dot product and magnitudes to calculate angle (more robust than arctan2)
            dot_product = np.dot(desired_velocity_vector, relative_position_vector)
            mag_velocity = np.linalg.norm(desired_velocity_vector)
            mag_relative_pos = np.linalg.norm(relative_position_vector)

            # Avoid division by zero
            if mag_velocity > 1e-6 and mag_relative_pos > 1e-6:
                angle = np.arccos(
                    np.clip(dot_product / (mag_velocity * mag_relative_pos), -1, 1)
                )
                if angle < min_angle:
                    min_angle = angle
                    best_neighbor_id = neighbor_id

        if best_neighbor_id != -1:
            next_pos = self.pos + timestep * desired_velocity_vector
            p_i_new = next_pos
            p_j = agents[best_neighbor_id].pos
            desired_v = self.mobility_constraint(agents, env, timestep)
            d = desired_v * timestep / 2

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
                if dist <= self.sensing_range - self.size and not ray_intersects_aabb(
                    p_i_new, point, env.obstacles
                ):
                    can_connect = True
            if can_connect:
                return best_neighbor_id
        return None

    def get_triangle_topologies(self, agents, env):
        critical_agents = self.get_critical_agents(agents, env)
        if len(critical_agents) < 2:
            return
        triangle_groups = []
        for i in range(len(critical_agents)):
            for j in range(i + 1, len(critical_agents)):
                robot_j = next(r for r in agents if r.index == critical_agents[i])
                robot_k = next(r for r in agents if r.index == critical_agents[j])

                distance_jk = np.linalg.norm(robot_j.pos - robot_k.pos)

                if distance_jk <= self.sensing_range:
                    triangle_groups.append([critical_agents[i], critical_agents[j]])
        return triangle_groups

    def minimize_triangle_topologies(self, agents, env, timestep):
        redundant = set()
        triangle_groups = self.get_triangle_topologies(agents, env)
        if triangle_groups is None:
            return
        desired_target_direction = self.goal - self.pos
        for group in triangle_groups:
            best_agent = self.find_best_connectivity(
                agents,
                env,
                desired_velocity_vector=desired_target_direction,
                timestep=timestep,
            )
            for id in group:
                if id != best_agent:
                    redundant.add(id)
        return redundant

    def step(self, agents, env, timestep):
        super().step(timestep=timestep)
        if self.goal is not None and not self.terminated(self.goal):
            # level 1: behavioural control
            desired_velocity = self.path_planner.total_force(
                self.pos, self.goal, self.index, agents, env.obstacles
            )

            # level 2: movement constraint
            desired_v = self.mobility_constraint(
                agents=agents, env=env, timestep=timestep
            )

            # level 3, 4: minimize topologies
            redundant = self.minimize_triangle_topologies(agents, env, timestep)
            critical_agents = self.get_critical_agents(agents, env)
            epsi = []
            if redundant is not None:
                for critical_agent in critical_agents:
                    if critical_agent in redundant:
                        continue
                    distance = np.linalg.norm(self.pos - agents[critical_agent].pos)
                    epsi.append(self.sensing_range - distance)

                if len(epsi) > 0:
                    desired_v = self.v_max * min(epsi) / (2 * timestep)
            self.move_to_goal(
                self.goal, agents, env.obstacles, timestep, desired_v=desired_v
            )
        else:
            self.goal = lloyd(self, agents, env)

import numpy as np
from shapely.geometry import Point, LineString
from adaptive_coverage.agents.agent import Agent
from adaptive_coverage.agents.cvt.lloyd import lloyd
from adaptive_coverage.utils.utils import ray_intersects_aabb
from collections import deque


class VoronoiAgent(Agent):
    def __init__(self, *args, critical_ratio=0.75, **kwargs):
        super().__init__(*args, **kwargs)
        self.critical_range = self.sensing_range * critical_ratio
        self.eps = self.sensing_range - self.critical_range
        self.tolerance = self.size

    def get_critical_agents(self, agents, env):
        """
        Get critical agents with definition in *Hierarchical Distributed Control for Global Network Integrity Preservation in Multi-Robot Systems*.

        Args: 
            agents: list of all agents.
            env: simulation environment.

        Return:
            list of all critical agents.
        """
        critical_agents = []
        for agent in agents:
            if agent.index != self.index:
                if self.is_critical_agent(agent, agents, env):
                    critical_agents.append(agent.index)
        return critical_agents

    def is_critical_agent(self, agent, agents, env):
        """
        Check if an agent is this agent's critical agent:
        - agent must be in the annulus (critical area): distance in (critical_range, sensing_range]
        - There must not exist any agent k that belongs to the union of current agent's non-critical area and agent.j's non-critical area (S^n_i union S^n_j = empty).

        Args:
            agent: agent to be checked.
            agents: list of all agents.
            env: simulation environment.

        Return:
            True if agent is critical.
        """
        if agent.index == self.index:
            return False
        rij = np.linalg.norm(agent.pos - self.pos)

        # must be inside annulus (critical area)
        if not (self.critical_range < rij < self.sensing_range):
            return False

        # check line-of-sight (if blocked -> not considered critical)
        if ray_intersects_aabb(self.pos, agent.pos, env.obstacles):
            return False

        # define noncritical area as inside critical_range (Sn_i)
        for other in agents:
            if other.index in (self.index, agent.index):
                continue
            # other in our noncritical area?
            d_other_i = np.linalg.norm(other.pos - self.pos)
            d_other_j = np.linalg.norm(other.pos - agent.pos)
            if ray_intersects_aabb(self.pos, other.pos, env.obstacles):
                continue
            if ray_intersects_aabb(agent.pos, other.pos, env.obstacles):
                continue
            in_Sn_i = (d_other_i < self.critical_range)
            in_Sn_j = (d_other_j < agent.critical_range)

            # if there exists a k in Sn_i union Sn_j, then agent is critical
            if in_Sn_i or in_Sn_j:
                return False

        # if no such k exists, it's noncritical by Definition 1
        return True

    def mobility_constraint(self, critical_agents, agents, env):
        if len(critical_agents) <= 0:
            return self.v_max * (self.eps / (2 * self.timestep))

        epsi = []
        for critical_id in critical_agents:
            neighbor = agents[critical_id]
            distance = np.linalg.norm(self.pos - neighbor.pos)
            if distance >= self.sensing_range:
                continue
            if ray_intersects_aabb(self.pos, neighbor.pos, env.obstacles):
                continue
            epsi.append(self.sensing_range - distance)

        if len(epsi) == 0:
            return self.v_max * (self.eps / (2 * self.timestep))

        min_eps = min(epsi)
        v = self.v_max * min_eps / (2 * self.timestep)

        desired_vec = self.goal - self.pos
        norm_desired = np.linalg.norm(desired_vec)
        desired_dir = desired_vec / norm_desired

        next_pos = self.pos + desired_dir * v * self.timestep

        all_ok = True
        for critical_id in critical_agents:
            neighbor = agents[critical_id]
            # if not self.check_future_connectivity(next_pos, neighbor, env):
            if not self.check_future_connectivity_sample(next_pos, neighbor, env):
                all_ok = False
                break

        if all_ok:
            return v

        # if none of the reduced steps are safe, stay still
        return 0.0

    def angle_to_goal(self, neighbor_pos):
        """
        Compute angle to desired goal.
        Return:
            angle to desired goal.
        """
        v_des = self.goal - self.pos
        v_rel = neighbor_pos - self.pos
        mv = np.linalg.norm(v_des)
        mr = np.linalg.norm(v_rel)
        if mv < 1e-9 or mr < 1e-9:
            return np.pi  # large angle if degenerate
        cosang = np.clip(np.dot(v_des, v_rel) / (mv * mr), -1.0, 1.0)
        return np.arccos(cosang)

    def get_triangle_topologies(self, critical_agents, agents, env):
        """
        Finds all local triangle topologies of current agent.

        Args:
            critical_agents: list of current agent's critical agents.

        Returns:
            A list of triangle topologies.
        """
        if len(critical_agents) <= 0:
            return []
        topos = []
        # sort the local critical agents index based on distance to current agent
        critical_agent_positions = np.array(
            [agents[index].pos for index in critical_agents])
        distances = np.linalg.norm(critical_agent_positions - self.pos, axis=1)
        sorted_indices = np.argsort(distances)
        critical_agents = np.array(critical_agents)[sorted_indices]
        for i in range(len(critical_agents) - 1):
            j = i + 1
            dist_ij = np.linalg.norm(agents[j].pos - agents[i].pos)
            if dist_ij < self.sensing_range and not ray_intersects_aabb(agents[i].pos, agents[j].pos, env.obstacles):
                topos.append([i, j])
        return topos

    def find_best_connectivity(self, topos, critical_agents, agents):
        """
        Finds the best neighbor to maintain a critical connection with,
        based on the minimum angle to the desired velocity vector.
        """
        # critical_agents = self.get_critical_agents(agents, env)
        if len(topos) <= 0:
            return None

        desired_velocity_vector = self.goal - self.pos

        if len(critical_agents) <= 0:
            return None

        min_angle = np.pi * 2
        best_neighbor_id = None

        for neighbor_id in critical_agents:
            in_any_topo = False
            for topo in topos:
                if neighbor_id in topo:
                    in_any_topo = True
                    break
            if not in_any_topo:
                continue

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

    def step(self, agents, env):
        super().step()

        # If we have a goal, attempt to minimize local connectivity first (levels 3/4)
        if self.goal is not None and not self.terminated(self.goal):
            # compute critical agents (level 2 constraint)
            critical_agents = self.get_critical_agents(agents, env)

            topos = self.get_triangle_topologies(critical_agents, agents, env)

            best_connectivity = self.find_best_connectivity(
                topos, critical_agents, agents)

            if best_connectivity is not None:
                real_critical_agents = [best_connectivity]
            else:
                real_critical_agents = critical_agents

            desired_v = self.mobility_constraint(
                real_critical_agents, agents, env)

            self.move_to_goal(
                self.goal, agents, env.obstacles, desired_v=desired_v
            )
        else:
            self.goal = lloyd(self, agents, env)

    def check_future_connectivity(self, next_pos, neighbor, env):
        """
        Check if next_pos maintains connectivity with neighbor under 3 worst-case
        neighbor positions (j0, j1, j2).
        """
        d = self.timestep * self.v_max

        v_ij = next_pos - neighbor.pos
        norm_v = np.linalg.norm(v_ij)
        if norm_v < 1e-9:
            return False
        u_ij = v_ij / norm_v
        u_perp = np.array([-u_ij[1], u_ij[0]])

        j0 = neighbor.pos
        j1 = neighbor.pos + d * u_perp
        j2 = neighbor.pos - d * u_perp
        j3 = neighbor.pos + d * u_ij
        j4 = neighbor.pos - d * u_ij
        worst_cases = [j0, j1, j2, j3, j4]

        for wpos in worst_cases:
            dist = np.linalg.norm(next_pos - wpos)
            if dist < self.sensing_range:
                if ray_intersects_aabb(next_pos, wpos, env.obstacles):
                    return False
            else:
                return False
        return True

    def check_future_connectivity_sample(self, next_pos, neighbor, env, N=24, eps=1e-9):
        d = self.timestep * self.v_max

        p = neighbor.pos
        v = next_pos - p
        r = np.linalg.norm(v)

        # Sample the boundary of the neighbor's motion disk
        thetas = np.linspace(0, 2*np.pi, N, endpoint=False)
        for th in thetas:
            wpos = p + d * np.array([np.cos(th), np.sin(th)])
            # distance check (should be redundant given r + d < R, but keep for numerical safety)
            if np.linalg.norm(next_pos - wpos) > self.sensing_range:
                return False
            # LOS check
            if ray_intersects_aabb(next_pos, wpos, env.obstacles):
                return False
        return True

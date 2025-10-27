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
        self.non_redundant_agents = []

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

        # NOTE: no need for this condition any more
        # must be inside annulus (critical area)
        # but still need to be inside sensing area
        if not (rij < self.sensing_range):
            return False

        # if not (self.critical_range < rij < self.sensing_range):
        # return False

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
            if in_Sn_i and in_Sn_j:
                return False

        # if no such k exists, it's noncritical by Definition 1
        return True

    def get_non_redundant_agents(self):
        return self.non_redundant_agents

    def mobility_constraint(self, critical_agents, agents, env):
        if len(critical_agents) == 0:
            return self.v_max * (self.eps / (2 * self.timestep))

        self.non_redundant_agents = critical_agents
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
        if norm_desired < 1e-9:
            return 0.0

        desired_dir = desired_vec / norm_desired

        next_pos = self.pos + desired_dir * v * self.timestep

        all_ok = True
        d = min_eps * self.timestep / 2
        for critical_id in critical_agents:
            neighbor = agents[critical_id]
            if not self.check_future_connectivity(next_pos, neighbor, d, env):
                # if not self.check_future_connectivity_sample(next_pos, neighbor, env, N=360):
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
        - critical_agents: list of agent indices (integers)
        Returns a list of topologies, each as [agent_idx_j, agent_idx_k]
        """
        if len(critical_agents) <= 1:
            return []

        # ensure we have numpy array of ints
        crit = np.array(critical_agents, dtype=int)

        # compute angle of each critical agent relative to self
        rel = np.array([agents[idx].pos - self.pos for idx in crit])
        thetas = np.arctan2(rel[:, 1], rel[:, 0])  # angle in [-pi, pi)
        sort_idx = np.argsort(thetas)
        crit_sorted = crit[sort_idx]

        topos = []
        q = len(crit_sorted)
        # cyclic adjacency: each pair (m, m+1) and also (q-1,0)
        for m in range(q):
            a_idx = int(crit_sorted[m])
            b_idx = int(crit_sorted[(m+1) % q])
            # check they are mutually visible and connected to each other (edge between them)
            dist_ab = np.linalg.norm(agents[a_idx].pos - agents[b_idx].pos)
            if dist_ab < self.sensing_range and not ray_intersects_aabb(agents[a_idx].pos, agents[b_idx].pos, env.obstacles):
                # triangle topology with this agent i and neighbors a_idx,b_idx
                topos.append([a_idx, b_idx])
        return topos

    def find_best_connectivity_in_a_topology(self, topo, agents):
        """
        Find the best connectivity in a local topology by finding the link that has minimum angle with reference to goal.

        Args:
            topo: a local topology.
            agents: list of all agents.

        Returns:
            An integer indicates best link to maintain.
        """
        best_neighbor_id = -1
        desired_velocity_vector = self.goal - self.pos
        min_angle = np.pi * 2

        for neighbor_id in topo:
            neighbor = agents[neighbor_id]
            relative_position_vector = neighbor.pos - self.pos
            dot_product = np.dot(desired_velocity_vector,
                                 relative_position_vector)
            mag_velocity = np.linalg.norm(desired_velocity_vector)
            mag_relative_pos = np.linalg.norm(relative_position_vector)
            if mag_relative_pos < 1e-6:
                continue

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

    def remove_redundancy(self, topos, agents):
        """
        Remove redundant links.

        Args:
            topos: list of local topologies.
            critical_agents: list of critical_agents.
            agents: list of all agents.

        Returns:
            A list of non-redundant critical agents.
        """
        new_critical_agents = []

        if len(topos) == 0:
            return []

        for topo in topos:
            best = self.find_best_connectivity_in_a_topology(topo, agents)
            if best != -1:
                new_critical_agents.append(int(best))
            # new_critical_agents.append(
                # self.find_best_connectivity_in_a_topology(topo, agents))

        seen = set()
        out = []
        for x in new_critical_agents:
            if x not in seen:
                seen.add(x)
                out.append(x)

        final = self.find_best_connectivity_in_a_topology(out, agents)

        if final != -1:
            return [final]

        return out

    def step(self, agents, env):
        super().step()

        # If we have a goal, attempt to minimize local connectivity first (levels 3/4)
        if self.goal is not None and not self.terminated(self.goal):
            # compute critical agents (level 2 constraint)
            critical_agents = self.get_critical_agents(agents, env)

            topos = self.get_triangle_topologies(critical_agents, agents, env)

            new_critical_agents = self.remove_redundancy(topos, agents)

            desired_v = self.mobility_constraint(
                new_critical_agents, agents, env)

            if desired_v < 1e-5:
                self.stop()
            else:
                self.move_to_goal(
                    self.goal, agents, env.obstacles, desired_v=desired_v
                )
        else:
            self.goal = lloyd(self, agents, env)

    def check_future_connectivity(self, next_pos, neighbor, d, env):
        """
        Check if next_pos maintains connectivity with neighbor under worst-case
        neighbor positions.
        """
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
        # worst_cases = [j0, j1, j2, j3, j4]
        worst_cases = [j1, j2]

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

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
        # store ignored links (i -> j) that we treat as non-critical (after minimization)
        self.ignored_links: set[int] = set()

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
                    # don't include links we've explicitly ignored
                    # if agent.index not in self.ignored_links:
                    critical_agents.append(agent.index)
        return critical_agents

    def is_critical_agent(self, agent, agents, env):
        """
        Check if an agent is this agent's critical agent:
        - agent must be in the annulus (critical area): distance in (critical_range, sensing_range]
        - There must exist at least one other agent k that belongs to our non-critical area but not to agent.j's non-critical area -> i.e. Sn_i \ Sn_j is non-empty.

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
                return False
            if ray_intersects_aabb(agent.pos, other.pos, env.obstacles):
                return False
            in_Sn_i = (d_other_i < self.critical_range)
            in_Sn_j = (d_other_j < agent.critical_range)

            # if exists k such that x_k in (S^n_i and S^n_j)
            if in_Sn_i and in_Sn_j:
                return False

        # if no such k exists, it's critical by Definition 1
        return True

    def mobility_constraint(self, critical_agents, agents):
        if len(critical_agents) <= 0:
            return self.v_max * self.eps / (2 * self.timestep)

        epsi = []
        for critical_id in critical_agents:
            neighbor = agents[critical_id]
            distance = np.linalg.norm(self.pos - neighbor.pos)
            if distance >= self.sensing_range:
                continue
            epsi.append(self.sensing_range - distance)

        if len(epsi) == 0:
            return self.v_max

        min_eps = min(epsi)
        return self.v_max * min_eps / (2 * self.timestep)

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

    def find_best_connectivity(self, critical_agents, agents, env):
        """
        Finds the best neighbor to maintain a critical connection with,
        based on the minimum angle to the desired velocity vector.
        """
        # critical_agents = self.get_critical_agents(agents, env)
        desired_velocity_vector = self.goal - self.pos

        if len(critical_agents) <= 0:
            return None

        min_angle = np.pi * 2
        best_neighbor_id = 0

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

        # If we have a goal, attempt to minimize local connectivity first (levels 3/4)
        if self.goal is not None and not self.terminated(self.goal):
            # compute critical agents (level 2 constraint)
            critical_agents = self.get_critical_agents(agents, env)

            # attempt minimization of local connectivity (level 3/4) to escape topology traps
            # We perform minimization only if there are >1 critical neighbors (paper's trigger)
            # if len(critical_agents) > 1:
            # _ = self.minimize_local_connectivity(agents, env)

            # recompute critical agents after minimization (some links may be ignored)
            critical_agents = self.get_critical_agents(agents, env)

            # mobility constraint: returns velocity cap
            desired_v = self.v_max
            if len(critical_agents) > 0:
                # choose best_connect to maintain as in your original code
                best_connect = self.find_best_connectivity(
                    critical_agents, agents, env)
                if best_connect is not None:
                    # mobility_constraint now returns a scalar velocity bound
                    desired_v = self.mobility_constraint(
                        [best_connect], agents)

            self.move_to_goal(
                self.goal, agents, env.obstacles, timestep, desired_v=desired_v
            )
        else:
            self.goal = lloyd(self, agents, env)

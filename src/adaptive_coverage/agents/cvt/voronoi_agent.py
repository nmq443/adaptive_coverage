import numpy as np
from adaptive_coverage.agents.agent import Agent
from adaptive_coverage.agents.cvt.lloyd import lloyd
from adaptive_coverage.utils.utils import ray_intersects_aabb, compute_coverage_percentage


class VoronoiAgent(Agent):
    def __init__(self, *args, critical_ratio=0.75, **kwargs):
        super().__init__(*args, **kwargs)
        self.critical_range = self.sensing_range * critical_ratio
        self.eps = self.sensing_range - self.critical_range
        self.tolerance = self.size * 2
        self.non_redundant_agents = []

        # local minima handling
        self.prev_pos = np.copy(self.pos)
        self.stuck_counter = 0
        self.STUCK_THRESHOLD = 15     # number of timesteps before adding noise
        self.JITTER_SCALE = 0.5       # fraction of v_max * dt

        # coverage-based exploration parameters
        self.NUM_COVERAGE_SAMPLES = 4     # number of directions to sample when stuck
        # exploration step: move slightly farther than a single nominal step to try escape
        # multiply v_max * timestep by this for exploration
        self.EXPLORATION_STEP_SCALE = 1

    def _compute_pairwise_info(self, agents, env):
        n = len(agents)
        dists = np.empty(n, dtype=float)
        los = [False] * n
        obs = env.obstacles
        for a in agents:
            idx = a.index
            dists[idx] = np.linalg.norm(a.pos - self.pos)
            los[idx] = ray_intersects_aabb(self.pos, a.pos, obs)
        return dists, los

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
        dists, los = self._compute_pairwise_info(agents, env)
        for agent in agents:
            if agent.index != self.index:
                if self.is_critical_agent(agent, agents, env, dists, los):
                    critical_agents.append(agent.index)
        return critical_agents

    def is_critical_agent(self, agent, agents, env, dists_self=None, los_self=None):
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

        if dists_self is None or los_self is None:
            dists_self, los_self = self._compute_pairwise_info(agents, env)

        rij = dists_self[agent.index]

        # NOTE: no need for this condition any more
        # must be inside annulus (critical area)
        # but still need to be inside sensing area
        if rij >= self.sensing_range:
            return False

        # if not (self.critical_range < rij < self.sensing_range):
        # return False

        # check line-of-sight (if blocked -> not considered critical)
        if los_self[agent.index]:
            return False

        # define noncritical area as inside critical_range (Sn_i)
        for other in agents:
            if other.index == self.index or other.index == agent.index:
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
        - critical_agents: list of agent indices (ints)
        Returns: list of [agent_j, agent_k]
        """
        n = len(critical_agents)
        if n <= 1:
            return []

        # Convert to numpy array
        crit = np.asarray(critical_agents, dtype=int)

        # Vectorized positions of critical agents relative to self
        crit_positions = np.array([agents[i].pos for i in crit])  # shape (n,2)
        rel = crit_positions - self.pos

        # Compute and sort by angle
        thetas = np.arctan2(rel[:, 1], rel[:, 0])
        sorted_idx = np.argsort(thetas)
        crit_sorted = crit[sorted_idx]
        crit_pos_sorted = crit_positions[sorted_idx]

        # Precompute pairwise distances for adjacent (cyclic)
        # Compute all distances except the last-to-first
        diffs = crit_pos_sorted[1:] - crit_pos_sorted[:-1]  # shape (n-1,2)
        dists = np.linalg.norm(diffs, axis=1)  # shape (n-1,)

        topos = []

        # Loop through adjacent pairs
        for m in range(n):
            j = crit_sorted[m]
            k = crit_sorted[(m + 1) % n]

            # Select precomputed distance or compute special wrap-around
            if m < n - 1:
                d = dists[m]
            else:
                # Wrap-around distance between last and first
                d = np.linalg.norm(crit_pos_sorted[-1] - crit_pos_sorted[0])

            # Visibility test
            if d < self.sensing_range:
                if not ray_intersects_aabb(agents[j].pos, agents[k].pos, env.obstacles):
                    topos.append([j, k])

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

    # -------------------------
    # Coverage-aware helper
    # -------------------------
    def find_coverage_improving_goal(self, agents, env):
        """
        Sample NUM_COVERAGE_SAMPLES candidate offsets around current position,
        simulate moving only this agent to that position, and pick the candidate
        that increases coverage the most (using compute_coverage_percentage).

        Returns:
            (best_goal, best_gain) where best_goal is a numpy array or None if no improvement.
        """
        # build positions array for all agents (assumes agents list indexed by agent.index)
        current_positions = np.array([a.pos.copy() for a in agents])
        # baseline coverage
        try:
            current_coverage = compute_coverage_percentage(
                current_positions, env, self.sensing_range)
        except Exception:
            # If the coverage function throws, do not break — fall back to jitter
            return None, 0.0

        best_goal = None
        best_gain = 0.0

        # exploration step magnitude
        exploration_step = self.v_max * self.timestep * self.EXPLORATION_STEP_SCALE

        thetas = np.linspace(
            0, 2 * np.pi, self.NUM_COVERAGE_SAMPLES, endpoint=False)
        for th in thetas:
            offset = exploration_step * np.array([np.cos(th), np.sin(th)])
            candidate_goal = self.pos + offset

            # skip candidate if LOS from current pos to candidate is blocked
            if ray_intersects_aabb(self.pos, candidate_goal, env.obstacles):
                continue

            # simulate this agent at candidate and compute coverage
            sim_positions = current_positions.copy()
            # ensure the agent's entry corresponds to its index
            sim_positions[self.index] = candidate_goal

            try:
                new_coverage = compute_coverage_percentage(
                    sim_positions, env, self.sensing_range)
            except Exception:
                # skip candidate if compute fails
                continue

            gain = new_coverage - current_coverage
            if gain > best_gain:
                best_gain = gain
                best_goal = candidate_goal

        return best_goal, best_gain

    def step(self, agents, env):
        super().step()

        # --- Detect if stuck ---
        moved_dist = np.linalg.norm(self.pos - self.prev_pos)
        self.prev_pos = np.copy(self.pos)

        if moved_dist < 1e-4:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0

        # --- CVT Goal if no current goal ---
        if self.goal is None or self.terminated(self.goal):
            self.goal = lloyd(self, agents, env)

        # --- If stuck, apply coverage-aware local-minima escape ---
        if self.stuck_counter > self.STUCK_THRESHOLD:
            best_goal, gain = self.find_coverage_improving_goal(agents, env)

            if best_goal is not None and gain > 1e-6:
                # adopt coverage-improving candidate
                self.goal = best_goal
            else:
                # fallback: random jitter (existing behaviour)
                jitter_mag = self.JITTER_SCALE * self.v_max * self.timestep
                jitter = jitter_mag * np.random.uniform(-1, 1, 2)
                tentative_goal = self.goal + jitter

                # optional safety check: no obstacles between self and new goal
                if not ray_intersects_aabb(self.pos, tentative_goal, env.obstacles):
                    self.goal = tentative_goal

            # reset stuck counter after attempting escape
            self.stuck_counter = 0

        # --- Standard connectivity-aware movement ---
        critical_agents = self.get_critical_agents(agents, env)
        topos = self.get_triangle_topologies(critical_agents, agents, env)
        new_critical_agents = self.remove_redundancy(topos, agents)

        desired_v = self.mobility_constraint(new_critical_agents, agents, env)

        if desired_v < 1e-5:
            self.stop()
        else:
            self.move_to_goal(
                self.goal, agents, env.obstacles, desired_v=desired_v
            )

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

        j1 = neighbor.pos + d * u_perp
        j2 = neighbor.pos - d * u_perp

        worst_cases = [j1, j2]

        for wpos in worst_cases:
            dist = np.linalg.norm(next_pos - wpos)
            if dist >= self.sensing_range:
                return False
            if ray_intersects_aabb(next_pos, wpos, env.obstacles):
                return False
        return True


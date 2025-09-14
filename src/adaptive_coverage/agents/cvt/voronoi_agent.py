import numpy as np
from shapely.geometry import Point, LineString
from adaptive_coverage.agents.agent import Agent
from adaptive_coverage.agents.cvt.lloyd import lloyd
from adaptive_coverage.utils.utils import ray_intersects_aabb
from collections import deque


class VoronoiAgent(Agent):
    def __init__(self, *args, valid_ratio=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.valid_range = self.sensing_range * valid_ratio
        self.critical_range = self.sensing_range * 0.8
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
                    if agent.index not in self.ignored_links:
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
        if not (self.critical_range < rij <= self.sensing_range):
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
            # if there exists a k in Sn_i but not in Sn_j, then agent is critical
            if in_Sn_i and (not in_Sn_j):
                return True

        # if no such k exists, it's noncritical by Definition 1
        return False

    def mobility_constraint(self, critical_agents, agents, env, timestep):
        if len(critical_agents) <= 0:
            return self.v_max

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
        allowed_step = min_eps / 2.0
        allowed_v = allowed_step / max(timestep, 1e-9)
        v_cap = min(self.v_max, allowed_v)

        desired_vec = self.goal - self.pos
        norm_desired = np.linalg.norm(desired_vec)
        if norm_desired < 1e-9:
            return v_cap
        desired_dir = desired_vec / norm_desired

        # try full step, then gradually smaller steps if unsafe
        fractions = [1.0, 0.75, 0.5, 0.25, 0.1]
        for frac in fractions:
            step_size = v_cap * timestep * frac
            next_pos = self.pos + desired_dir * step_size

            all_ok = True
            for critical_id in critical_agents:
                neighbor = agents[critical_id]
                if not self.check_future_connectivity(next_pos, neighbor, env):
                    all_ok = False
                    break

            if all_ok:
                return (step_size / timestep)  # convert back to velocity

        # if none of the reduced steps are safe, stay still
        return 0.0

    # ---------- helpers for connectivity minimization ----------

    def build_neighbor_graph(self, agents, env):
        """
        Build adjacency dict for neighbours within sensing_range and LOS,
        ignoring self.ignored_links. Returns {id: set(neighbor_ids)} for neighbors of self.
        This graph considers only neighbours of self (N_i), and edges between them if they can
        communicate (within sensing & LOS).
        """
        N_i = []
        idx_map = {}
        for a in agents:
            if a.index == self.index:
                continue
            if np.linalg.norm(a.pos - self.pos) <= self.sensing_range:
                # check line-of-sight between self and a to be considered neighbor (paper assumes)
                if not ray_intersects_aabb(self.pos, a.pos, env.obstacles):
                    N_i.append(a)
                    idx_map[a.index] = a

        # adjacency among N_i (edge exists if two neighbors are within each other's sensing and LOS)
        adj = {a.index: set() for a in N_i}
        for a in N_i:
            for b in N_i:
                if a.index == b.index:
                    continue
                if b.index in self.ignored_links or a.index in self.ignored_links:
                    # if we've ignored the connection, don't count it
                    continue
                d_ab = np.linalg.norm(a.pos - b.pos)
                if d_ab <= self.sensing_range:
                    if not ray_intersects_aabb(a.pos, b.pos, env.obstacles):
                        adj[a.index].add(b.index)
        return adj

    def get_ngroup(self, agents, env):
        """
        Return 
            Ng_i: neighbour indices in cyclic order (by angle around self). 
            Only includes neighbours that are currently connected (and not ignored).
        """
        neighbors = []
        for a in agents:
            if a.index == self.index:
                continue
            if a.index in self.ignored_links:
                continue
            d = np.linalg.norm(a.pos - self.pos)
            if d <= self.sensing_range and not ray_intersects_aabb(self.pos, a.pos, env.obstacles):
                vec = a.pos - self.pos
                angle = np.arctan2(vec[1], vec[0])
                neighbors.append((angle, a.index))
        if not neighbors:
            return []
        neighbors.sort(key=lambda x: x[0])
        return [idx for _, idx in neighbors]

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

    def path_exists_between(self, start_id, goal_id, local_adj):
        """
        BFS on local_adj (adjacency among neighbours) to check if a path exists
        between start_id and goal_id ignoring self node.
        """
        if start_id not in local_adj or goal_id not in local_adj:
            return False
        q = deque([start_id])
        seen = {start_id}
        while q:
            u = q.popleft()
            if u == goal_id:
                return True
            for v in local_adj[u]:
                if v not in seen:
                    seen.add(v)
                    q.append(v)
        return False

    def connectivity_removal_rule(self, agents, env):
        """
        Implements Proposition 8 / Eq.30-31 idea:
        - Build Ng_i (neighbors in cyclic order)
        - For each neighbor compute theta_i^k = angle between v_a and r_ik
        - Candidate redundant Rn sets: those neighbors where theta > min_k theta_k (i.e. not closest)
        - Consensus: only remove if symmetric (both agents agree)
        Returns two sets: Rn_T (triangle) and Rn_K (k-connected), but for simplicity we
        will compute one set Rn (union) which will be split heuristically.
        """
        ngroup = self.get_ngroup(agents, env)
        if len(ngroup) <= 1:
            return set(), set()

        # compute angles
        theta = {}
        for nid in ngroup:
            neighbor = next(a for a in agents if a.index == nid)
            theta[nid] = self.angle_to_goal(neighbor.pos)

        # find minimum theta among group (the preferred neighbor)
        min_theta = min(theta.values())

        # candidate redundant: those strictly larger than min_theta (paper uses >)
        candidates = {nid for nid, th in theta.items() if th > min_theta}

        if not candidates:
            return set(), set()

        # Build local adjacency among neighbors to determine whether the group is triangle (T) or k-connected (K)
        local_adj = self.build_neighbor_graph(agents, env)

        # Rn_T = those candidates that are part of triangle topologies (i.e., neighbor pairs are connected)
        # Rn_K = those candidates that are part of k-connected (connected indirectly) topologies
        Rn_T = set()
        Rn_K = set()

        for cid in list(candidates):
            for other in ngroup:
                if other == cid:
                    continue
                # check direct edge between candidate and other (triangle adjacency)
                if cid in local_adj and other in local_adj[cid] and other in candidates:
                    # candidate belongs to triangle with another candidate — mark Rn_T candidate
                    Rn_T.add(cid)
                else:
                    # check if there exists a path between cid and other through local_adj excluding i
                    if self.path_exists_between(cid, other, local_adj):
                        Rn_K.add(cid)

        # Consensus check: only remove if the neighbor also proposes to remove (symmetric)
        # We'll ask neighbor to evaluate the same rule locally (simulated call)
        symmetric_Rn_T = set()
        symmetric_Rn_K = set()

        for nid in Rn_T:
            neighbor = next(a for a in agents if a.index == nid)
            # neighbor evaluates its preference regarding self.index
            their_candidates_T, their_candidates_K = neighbor._simulate_connectivity_pref(
                self.index, agents, env)
            if self.index in their_candidates_T:
                symmetric_Rn_T.add(nid)

        for nid in Rn_K:
            neighbor = next(a for a in agents if a.index == nid)
            their_candidates_T, their_candidates_K = neighbor._simulate_connectivity_pref(
                self.index, agents, env)
            if self.index in their_candidates_K:
                symmetric_Rn_K.add(nid)

        # Note: we return two sets; the caller will combine them as needed
        return symmetric_Rn_T, symmetric_Rn_K

    # ---------- simulation helper used to simulate neighbor's local decision ----------
    def _simulate_connectivity_pref(self, other_id, agents, env):
        """
        Helper used to simulate the neighbor's perspective *without* performing changes.
        This must be fast and minimal: computes neighbour list, angles, and returns
        sets of candidate removals that include `other_id` if the neighbor would propose removing it.
        """
        # neighbor perspective: we need the neighbor object to run its logic; but since this method
        # is invoked on the neighbor object itself, we can reuse self.* but with 'other_id' as a member.
        # So this method assumes it is called on the neighbor instance (not on this agent).
        ngroup = self.get_ngroup(agents, env)
        if len(ngroup) <= 1:
            return set(), set()
        theta = {}
        for nid in ngroup:
            neighbor = next(a for a in agents if a.index == nid)
            theta[nid] = self.angle_to_goal(neighbor.pos)
        min_theta = min(theta.values())
        candidates = {nid for nid, th in theta.items() if th > min_theta}
        if not candidates:
            return set(), set()
        local_adj = self.build_neighbor_graph(agents, env)
        Rn_T = set()
        Rn_K = set()
        for cid in list(candidates):
            for other in ngroup:
                if other == cid:
                    continue
                if cid in local_adj and other in local_adj[cid] and other in candidates:
                    Rn_T.add(cid)
                else:
                    if self.path_exists_between(cid, other, local_adj):
                        Rn_K.add(cid)
        return Rn_T, Rn_K

    def minimize_local_connectivity(self, agents, env):
        """
        High-level call to compute Rn_T and Rn_K and update self.ignored_links
        when symmetric agreement is found. This function returns True if any removal happened.
        """
        Rn_T, Rn_K = self.connectivity_removal_rule(agents, env)
        removed_any = False

        # When minimization chooses to remove a critical link, both agents must add each other
        # to their ignored_links (symmetry). We can request neighbor to add link if they agree.
        for nid in Rn_T | Rn_K:
            neighbor = next(a for a in agents if a.index == nid)
            # to be safe: demand neighbor also agrees (we already checked symmetric sets), but ensure
            # they won't remove a link that would break connectivity — in a full system you'd run
            # the group-consensus; here we rely on symmetric_Rn sets from connectivity_removal_rule
            # update both sides:
            if nid not in self.ignored_links:
                self.ignored_links.add(nid)
                removed_any = True
            if self.index not in neighbor.ignored_links:
                neighbor.ignored_links.add(self.index)

        return removed_any

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

    # ---------- step: call minimization before moving ----------

    def step(self, agents, env, timestep):
        super().step(timestep=timestep)
        # ensure ignored_links attribute exists for all agents (if code loads older agents)
        for a in agents:
            if not hasattr(a, "ignored_links"):
                a.ignored_links = set()

        # If we have a goal, attempt to minimize local connectivity first (levels 3/4)
        if self.goal is not None and not self.terminated(self.goal):
            # compute critical agents (level 2 constraint)
            critical_agents = self.get_critical_agents(agents, env)

            # attempt minimization of local connectivity (level 3/4) to escape topology traps
            # We perform minimization only if there are >1 critical neighbors (paper's trigger)
            if len(critical_agents) > 1:
                _ = self.minimize_local_connectivity(agents, env)

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
                        [best_connect], agents, env, timestep)

            self.move_to_goal(
                self.goal, agents, env.obstacles, timestep, desired_v=desired_v
            )
        else:
            self.goal = lloyd(self, agents, env)

    def check_future_connectivity(self, next_pos, neighbor, env):
        """
        Check if next_pos maintains connectivity with neighbor under 3 worst-case
        neighbor positions (j0, j1, j2).
        """
        rij = np.linalg.norm(neighbor.pos - self.pos)
        d = (self.sensing_range - rij) / 2.0
        if d <= 0:
            return False

        v_ij = next_pos - neighbor.pos
        norm_v = np.linalg.norm(v_ij)
        if norm_v < 1e-9:
            return False
        u_ij = v_ij / norm_v
        u_perp = np.array([-u_ij[1], u_ij[0]])

        j0 = neighbor.pos + d * u_ij
        j1 = neighbor.pos + d * u_perp
        j2 = neighbor.pos - d * u_perp
        worst_cases = [j0, j1, j2]

        for wpos in worst_cases:
            dist = np.linalg.norm(next_pos - wpos)
            if dist < self.critical_range:
                if not ray_intersects_aabb(next_pos, wpos, env.obstacles):
                    return True
        return False

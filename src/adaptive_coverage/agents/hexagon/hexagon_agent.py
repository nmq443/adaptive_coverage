import numpy as np
from adaptive_coverage.utils.utils import ray_intersects_aabb, nearest_points_on_obstacles, normalize_angle
from adaptive_coverage.agents.agent import Agent


class HexagonAgent(Agent):
    def __init__(self, index, init_pos, size, path_planner, sensing_range):
        super().__init__(index, init_pos, size, path_planner, sensing_range)
        # Hexagon agent parameters
        self.source = -1
        self.virtual_targets = []
        self.occupied_virtual_targets = []
        self.hidden_vertices = []
        self.penalty_nodes = []
        self.assigned_target = None
        self.state = "unassigned"
        self.first_time = True
        self.route_id = 0
        self.route = []
        self.flag = 0
        self.tc = 0
        self.is_penalty_node = False
        self.invalid_targets = []

    def is_occupied(self):
        return self.state == "occupied"

    def is_assigned(self):
        return self.state == "assigned"

    def is_unassigned(self):
        return self.state == "unassigned"

    def set_state(self, state: str):
        self.state = state

    def set_goal(self, goal: np.ndarray):
        self.goal = goal

    def render(
            self,
            screen,
            font,
            agents,
            timestep,
    ):
        super().render(screen, font, agents, timestep)

    def mobility_control(self, agents, env):
        if self.route_id == len(self.route) - 2:
            cur_node = agents[self.route[self.route_id]]
            if np.linalg.norm(cur_node.pos - self.pos) <= self.size * 4 and self.flag == 0:
                self.flag = 1
        if self.flag == 0:
            dest1 = agents[self.route[self.route_id]].pos
            dest2 = None
            if self.route_id + 1 < len(self.route) - 1:
                dest2 = agents[self.route[self.route_id + 1]].pos
            self.move_to_goal(dest1, agents, env.obstacles)
            if dest2 is not None and np.linalg.norm(dest2 - self.pos) <= self.sensing_range:
                if np.linalg.norm(self.pos - dest1) <= self.size * 4:
                    self.route_id += 1

    def get_travel_distance(self):
        """Get total travel distance."""
        traj = np.array(self.traj)
        if len(traj) < 2:
            return 0.0
        displacements = traj[1:] - traj[:-1]
        distances = np.linalg.norm(displacements, axis=1)
        return np.sum(distances)

    def alignment_behaviour(self, dest: np.ndarray):
        return KG * (dest - self.pos)

    def obstacle_behaviour(self):
        if len(OBSTACLES) == 0:
            return np.zeros(2)

        obs_pts = nearest_points_on_obstacles(self.pos, OBSTACLES)
        diff = self.pos - obs_pts
        dist = np.linalg.norm(diff, axis=1) - SIZE
        dist = np.clip(dist, 1e-4, None)  # avoid division by zero

        mask = dist < AVOIDANCE_RANGE
        if not np.any(mask):
            return np.zeros(2)

        diff_valid = diff[mask]
        dist_valid = dist[mask][:, np.newaxis]

        force = (
                KO
                * (1.0 / dist_valid - 1.0 / AVOIDANCE_RANGE)
                * diff_valid
                / (dist_valid ** 2)
        )
        vo = np.sum(force, axis=0)
        return vo

    def separation_behaviour(self, agents: list):
        positions = np.array(
            [agent.pos for agent in agents if agent.index != self.index]
        )
        if positions.size == 0:
            return np.zeros(2)

        directions = positions - self.pos  # Vector from self to others
        # Compute all distances at once
        distances = np.linalg.norm(directions, axis=1)
        mask = distances <= SENSING_RANGE  # Only consider nearby agents

        if not np.any(mask):
            return np.zeros(2)

        # Compute avoidance vector only for close agents
        directions = directions[mask]
        distances = distances[mask][:, np.newaxis]  # Reshape for broadcasting

        va = np.sum(
            (
                    KA
                    * np.exp(-BETA_C * (distances - AVOIDANCE_RANGE))
                    * (directions / distances)
                    / (distances - AVOIDANCE_RANGE)
            ),
            axis=0,
        )
        return va

    def reached_target(self, goal: np.ndarray):
        distance = np.linalg.norm(goal - self.pos)
        distance = np.round(distance, 3)
        return distance <= EPS

    def generate_virtual_targets(self, agents, env):
        if self.source != -1:
            direction = agents[self.source].pos - self.pos
            phi_0 = np.arctan2(direction[1], direction[0])
            phi_0 = normalize_angle(phi_0)
        else:
            phi_0 = 0.0
        virtual_targets = []
        occupied_virtual_targets = []
        hidden_vertices = []
        for i in range(6):
            phi = phi_0 + 2 * np.pi * i / 6
            phi = normalize_angle(phi)
            virtual_target = self.pos + np.array(
                [HEXAGON_RANGE * np.cos(phi), HEXAGON_RANGE * np.sin(phi)]
            )
            virtual_target = np.round(virtual_target, 3)
            is_valid, is_hidden_vertex = self.is_valid_virtual_target(
                virtual_target, agents, env
            )
            if is_valid:  # is a valid virtual target
                virtual_targets.append(virtual_target)
                occupied_virtual_targets.append(False)
            else:
                if is_hidden_vertex:
                    hidden_vertices.append([virtual_target, i])
        if len(virtual_targets) > 0:
            self.virtual_targets.extend(virtual_targets)
            self.occupied_virtual_targets.extend(occupied_virtual_targets)
        if len(hidden_vertices) > 0:
            self.hidden_vertices.extend(hidden_vertices)
        if USE_PENALTY_NODE:
            if len(self.hidden_vertices) > 0:
                for i in range(len(self.hidden_vertices) - 1):
                    self.compute_penalty_node(
                        phi_0=phi_0,
                        v1=hidden_vertices[i],
                        v2=hidden_vertices[i + 1],
                        agents=agents,
                        env=env,
                    )
                self.compute_penalty_node(
                    phi_0=phi_0,
                    v1=hidden_vertices[0],
                    v2=hidden_vertices[-1],
                    agents=agents,
                    env=env,
                )

    def compute_penalty_node(self, v1, v2, phi_0, env, agents):
        """
        Compute penalty node for coverage interior angle.

        Args:
            v1 (list): hidden vertex 1, with v1[0] is 2D position, v1[1] is its index.
            v2 (list): hidden vertex 1, with v2[0] is 2D position, v2[1] is its index.
            phi_0 (float): angle between source agent and current agent (if current agent is not the first agent).
            env (Environment): simulation environment.
            agents (list): list of all agents.
        """
        if ORIGINAL_METHOD:
            phi1 = 2 * np.pi * v1[1] / 6
            phi2 = 2 * np.pi * v2[1] / 6
            phi = phi_0 + RHO * (phi1 + phi2) / 2
            x = HEXAGON_RANGE * np.cos(phi)
            y = HEXAGON_RANGE * np.sin(phi)
            pos = self.pos + np.array([x, y])
        else:
            index_i, index_j = v1[1], v2[1]
            if abs(index_i - index_j) != 1:
                return
            pos = find_penalty_node(
                index=self.index,
                v1=v1,
                v2=v2,
                env=env,
                agents=agents,
            )

        # Then check if the penalty node is valid
        is_valid, _ = self.is_valid_virtual_target(target=pos, agents=agents, env=env)
        if is_valid:
            self.virtual_targets.append(pos)
            self.occupied_virtual_targets.append(False)
            self.penalty_nodes.append(pos)

    def is_valid_virtual_target(self, target, agents, env):
        """
        Check if a virtual target is valid or not.

        Args:
            target (numpy.ndarray): virtual target to check.
            agents (list): list of all agents.
            env (Environment): simulation environment.

        Returns:
            tuple (bool, bool): check if is a valid vertex and if not, is it a hidden vertex.
        """
        # not a hidden vertex
        if not env.point_is_in_environment(target):
            return False, True
        is_in_obs = env.point_is_in_obstacle(target)
        if is_in_obs:
            self.invalid_targets.append(target)
            return False, True
        other_agent_positions = np.array(
            [
                other_agent.pos
                for other_agent in agents
                if (other_agent.is_occupied() and other_agent.index != self.index)
            ]
        )
        if len(other_agent_positions) > 0:
            distances = np.linalg.norm(target - other_agent_positions, axis=1)
            if np.any(distances <= 20 * EPS):  # not occupied by another agent
                return False, False
            if self.is_penalty_node:
                if np.any(distances <= SENSING_RANGE):  # not in a coverage range
                    return False, False

        # not a neighbour's targets
        for other_agent in agents:
            virtual_targets = np.array([pt for pt in other_agent.virtual_targets])
            if len(virtual_targets) > 0:
                distances = np.linalg.norm(virtual_targets - target, axis=1)
                distances = np.round(distances, 5)
                if np.any(distances <= 20 * EPS):
                    return False, False
                if self.is_penalty_node:
                    for i, dist in enumerate(distances):
                        if dist <= SENSING_RANGE and not ray_intersects_aabb(
                                agents[j].pos, target, OBSTACLES
                        ):  # not in a coverage range
                            return False, False

        # If behind an obstacle
        if ray_intersects_aabb(self.pos, target, OBSTACLES):
            return False, True

        return True, False

    def on_occupied(self, landmarks, agents, env):
        """
        Occupation phase.

        Args:
            landmarks (deque): queue of landmarks.
            agents (list): list of all agents.
            env (Environment): simulation environment.
        """
        self.stop()
        if self.first_time:
            self.generate_virtual_targets(agents, env)
            if len(self.virtual_targets) > 0:
                landmarks.append(self.index)
            self.first_time = False
        if (
                len(landmarks) > 0
                and self.index in landmarks
                and landmarks[0] == self.index
        ):
            if self.tc >= len(self.virtual_targets):
                landmarks.popleft()

    def on_assigned(self, agents, landmarks, env):
        """
        Assignment phase.

        Args:
            agents (list): list of all agents.
            landmarks (deque): queue of landmarks.
        """
        self.mobility_control(agents, env)
        if self.flag == 1:
            if self.goal is not None:
                self.set_goal(self.assigned_target)
                if self.reached_target(self.goal):
                    self.stop()
                    self.set_state("occupied")
                else:
                    self.move_to_goal(self.goal, agents, env.obstacles)

    def on_unassigned(self, landmarks, agents):
        """
        Unassignment phase.

        Args:
            agents (list): list of all agents.
            landmarks (deque): queue of landmarks.
        """
        self.stop()
        if len(landmarks) > 0:
            lc = landmarks[0]
            landmark = agents[lc]
            tc = landmark.tc
            if (
                    0 <= tc < len(landmark.virtual_targets)
                    and not landmark.occupied_virtual_targets[tc]
            ):
                is_penalty_node = False
                if len(agents[lc].penalty_nodes) > 0:
                    num_vt = len(agents[lc].virtual_targets)
                    num_pen = len(agents[lc].penalty_nodes)
                    if tc >= num_vt - num_pen:
                        is_penalty_node = True
                target = landmark.virtual_targets[tc]
                self.assigned_target = target
                self.source = landmark.index
                self.is_penalty_node = is_penalty_node
                landmark.occupied_virtual_targets[tc] = True
                landmark.tc += 1
                route = self.get_shortest_path(lc, agents)
                route.append(self.assigned_target)
                if len(route) > 0:
                    self.route = route
                    self.set_state("assigned")

    def build_graph(self, agents):
        """
        Build a graph for current agent. Nodes are occupied agents' ids.

        Args:
            agents (list): list of all agents.

        Returns:
            numpy.ndarray: connectivity graph for current agent.
        """
        n = len(agents)
        graph = np.zeros((n, n))
        for i in range(n):
            if not agents[i].is_occupied():
                continue
            for j in range(n):
                if not agents[j].is_occupied():
                    continue
                dist = np.linalg.norm(agents[i].pos - agents[j].pos)
                if dist <= self.sensing_range:
                    if not ray_intersects_aabb(agents[j].pos, agents[i].pos, OBSTACLES):
                        graph[i][j] = graph[j][i] = dist
        for i in range(n):
            if not agents[i].is_occupied():
                continue
            dist = np.linalg.norm(agents[i].pos - self.pos)
            if dist <= self.sensing_range:
                if not ray_intersects_aabb(agents[i].pos, self.pos, OBSTACLES):
                    graph[self.index][i] = graph[i][self.index] = dist
        return graph

    def get_shortest_path(self, landmark_id: int, agents: list):
        """
        Get shortest path from current agent to current activated landmark.

        Args:
            landmark_id (int): index of current activated landmark.
            agents (list): list of all agents.

        Returns:
            list: shortest path found.
        """
        graph = self.build_graph(agents)
        # bfs shortest path
        visited = set()
        start_node = (self.index, [self.index])
        queue = deque([start_node])

        while queue:
            current, path = queue.popleft()
            visited.add(current)

            for neighbor in range(len(graph[current])):
                neighbor = int(neighbor)
                if neighbor in visited:
                    continue
                if graph[current][neighbor] == 0:
                    continue
                if neighbor == landmark_id:
                    return path + [neighbor]
                queue.append((neighbor, path + [neighbor]))

        return [start_node[0]]  # if there is only one element

    def step(self, landmarks, agents, env):
        """
        Run the distributed control.

        Args:
            env (Environment): simulation environment.
            agents (list): list of all agents.
            landmarks (deque): queue of landmarks.
        """
        if self.is_occupied():
            self.on_occupied(landmarks=landmarks, agents=agents, env=env)
        elif self.is_assigned():
            self.on_assigned(agents=agents, landmarks=landmarks, env=env)
        elif self.is_unassigned():
            self.on_unassigned(agents=agents, landmarks=landmarks)

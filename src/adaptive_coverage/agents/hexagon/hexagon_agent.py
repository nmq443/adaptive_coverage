import numpy as np
from collections import deque
from adaptive_coverage.agents.hexagon.penalty_node_solver import (
    OriginalSolver,
    PSOSolver,
)
from adaptive_coverage.utils.utils import (
    ray_intersects_aabb,
    normalize_angle,
)
from adaptive_coverage.agents.agent import Agent


class HexagonAgent(Agent):
    def __init__(
        self,
        *args,
        rho,
        pso_weights,
        pso_num_iterations,
        pso_num_particles,
        original_method=False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # Hexagon agent parameters
        self.original_method = original_method
        self.hexagon_range = 0.9 * self.sensing_range
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
        self.rho = rho
        self.pso_weights = pso_weights
        self.pso_num_iteration = pso_num_iterations
        self.pso_num_particles = pso_num_particles
        self.fitness_func_hist = None

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

    def mobility_control(self, agents, env):
        if self.route_id == len(self.route) - 2:
            cur_node = agents[self.route[self.route_id]]
            if (
                np.linalg.norm(cur_node.pos - self.pos) <= self.size * 4
                and self.flag == 0
            ):
                self.flag = 1
        if self.flag == 0:
            dest1 = agents[self.route[self.route_id]].pos
            dest2 = None
            if self.route_id + 1 < len(self.route) - 1:
                dest2 = agents[self.route[self.route_id + 1]].pos
            self.move_to_goal(dest1, agents, env.obstacles)
            if (
                dest2 is not None
                and np.linalg.norm(dest2 - self.pos) <= self.sensing_range
            ):
                if np.linalg.norm(self.pos - dest1) <= self.size * 4:
                    self.route_id += 1

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
                [self.hexagon_range * np.cos(phi), self.hexagon_range * np.sin(phi)]
            )
            # virtual_target = np.round(virtual_target, 3)
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
        index_i, index_j = v1[1], v2[1]
        if abs(index_i - index_j) == 1 or abs(index_j - index_i) == 5:
            if self.original_method:
                solver = OriginalSolver(
                    index=self.index,
                    pos=self.pos,
                    sensing_range=self.sensing_range,
                    hexagon_range=self.hexagon_range,
                    avoidance_range=self.avoidance_range,
                    phi_0=phi_0,
                    rho=self.rho,
                    v1=v1,
                    v2=v2,
                )
                pos = solver.solve()
            else:
                solver = PSOSolver(
                    index=self.index,
                    pos=self.pos,
                    sensing_range=self.sensing_range,
                    hexagon_range=self.hexagon_range,
                    avoidance_range=self.avoidance_range,
                    phi_0=phi_0,
                    v1=v1,
                    v2=v2,
                    env=env,
                    agents=agents,
                    pso_weights=self.pso_weights,
                )

                pos = solver.solve(
                    num_iterations=self.pso_num_iteration,
                    num_particles=self.pso_num_particles,
                )

            # Then check if the penalty node is valid
            is_valid, _ = self.is_valid_virtual_target(
                target=pos, agents=agents, env=env
            )
            if is_valid:
                if not self.original_method:
                    self.fitness_func_hist = solver.fitness_func_hist
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
        if not env.point_is_in_environment(target, self.size):
            return False, True
        is_in_obs = env.point_is_in_obstacle(target, self.size)
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
            if np.any(
                distances <= 20 * self.tolerance
            ):  # not occupied by another agent
                return False, False
            if self.is_penalty_node:
                if np.any(distances <= self.tolerance):  # not in a coverage range
                    return False, False

        # not a neighbour's targets
        for other_agent in agents:
            virtual_targets = np.array([pt for pt in other_agent.virtual_targets])
            if len(virtual_targets) > 0:
                distances = np.linalg.norm(virtual_targets - target, axis=1)
                distances = np.round(distances, 5)
                if np.any(distances <= 20 * self.tolerance):
                    return False, False
                if self.is_penalty_node:
                    for i, dist in enumerate(distances):
                        if dist <= self.sensing_range and not ray_intersects_aabb(
                            agents[i].pos, target, env.obstacles
                        ):  # not in a coverage range
                            return False, False

        # If behind an obstacle
        if ray_intersects_aabb(self.pos, target, env.obstacles):
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
        # self.stop()
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

    def on_assigned(self, agents, env):
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
                if self.terminated(self.goal):
                    # self.stop()
                    self.set_state("occupied")
                else:
                    self.move_to_goal(self.goal, agents, env.obstacles)

    def on_unassigned(self, landmarks, agents, env):
        """
        Unassignment phase.

        Args:
            agents (list): list of all agents.
            landmarks (deque): queue of landmarks.
        """
        # self.stop()
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
                route = self.get_shortest_path(lc, agents, env)
                route.append(self.assigned_target)
                if len(route) > 0:
                    self.route = route
                    self.set_state("assigned")

    def build_graph(self, agents, env):
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
                    if not ray_intersects_aabb(
                        agents[j].pos, agents[i].pos, env.obstacles
                    ):
                        graph[i][j] = graph[j][i] = dist
        for i in range(n):
            if not agents[i].is_occupied():
                continue
            dist = np.linalg.norm(agents[i].pos - self.pos)
            if dist <= self.sensing_range:
                if not ray_intersects_aabb(agents[i].pos, self.pos, env.obstacles):
                    graph[self.index][i] = graph[i][self.index] = dist
        return graph

    def get_shortest_path(self, landmark_id, agents, env):
        """
        Get shortest path from current agent to current activated landmark.

        Args:
            landmark_id (int): index of current activated landmark.
            agents (list): list of all agents.

        Returns:
            list: shortest path found.
        """
        graph = self.build_graph(agents, env)
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
        super().step()
        if self.is_occupied():
            self.on_occupied(landmarks, agents, env)
        elif self.is_assigned():
            self.on_assigned(agents, env)
        elif self.is_unassigned():
            self.on_unassigned(landmarks, agents, env)

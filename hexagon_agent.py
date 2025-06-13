import pygame as pg
import numpy as np
from shapely.geometry import LineString
from enum import Enum
from collections import deque
from configs import *
from pso import find_penalty_node
from utils import ray_intersects_aabb, nearest_points_on_obstacles


class State(Enum):
    OCCUPIED = 0
    UNASSIGNED = 1
    ASSIGNED = 2


class Agent:
    def __init__(
        self,
        index,
        init_pos,
    ):
        # Behaviour Control Parameters
        self.index = index
        self.pos = init_pos
        self.vel = np.zeros(2)
        self.trajectory = [self.pos.copy()]

        # Distributed Control Parameters
        self.source: int = -1
        self.virtual_targets: list[np.ndarray] = []
        self.occupied_virtual_targets: list[np.ndarray] = []
        self.hidden_vertices: list = []
        self.penalty_nodes: list[np.ndarray] = []
        self.goal: np.ndarray = None
        self.state: State = State.UNASSIGNED
        self.first_time: bool = True
        self.route_id: int = 0
        self.route: list[int] = []
        self.flag = 0
        self.tc: int = 0
        self.is_penalty_node: bool = False
        self.invalid_targets: list = []

    def is_occupied(self):
        return self.state == State.OCCUPIED

    def is_assigned(self):
        return self.state == State.ASSIGNED

    def is_unassigned(self):
        return self.state == State.UNASSIGNED

    def set_state(self, state: str):
        if state == "occupied":
            self.state = State.OCCUPIED
        if state == "unassigned":
            self.state = State.UNASSIGNED
        if state == "assigned":
            self.state = State.ASSIGNED

    def set_goal(self, goal: np.ndarray):
        self.goal = goal

    def render(self, screen: pg.Surface, font: pg.font.Font, agents: list, timestep):
        color = COLOR
        # if self.is_assigned():
        #     color = ASSIGNED_AGENT_COLOR
        # elif self.is_occupied():
        #     color = OCCUPIED_AGENT_COLOR
        # elif self.is_unassigned():
        #     color = UNASSIGNED_AGENT_COLOR
        if self.is_penalty_node:
            color = PENALTY_AGENT_COLOR
        pg.draw.circle(
            surface=screen,
            center=self.pos,
            color=color,
            radius=SIZE,
        )
        if SHOW_SENSING_RANGE:
            pg.draw.circle(
                surface=screen,
                center=self.pos,
                color=SENSING_COLOR,
                radius=SENSING_RANGE,
                width=2,
            )
        if SHOW_CONNECTIONS and timestep >= 10:
            for other in agents:
                if (
                    other.index != self.index
                    and other.is_occupied()
                    and np.linalg.norm(self.pos - other.pos) < SENSING_RANGE
                ):
                    pg.draw.line(screen, SENSING_COLOR, self.pos, other.pos)
        for i in range(len(self.virtual_targets)):
            if not self.occupied_virtual_targets[i]:
                pg.draw.circle(screen, "green", self.virtual_targets[i], 3)
        text_surface = font.render(str(self.index), True, "black")
        text_rect = text_surface.get_rect(center=(self.pos[0] + 10, self.pos[1] - 10))
        screen.blit(text_surface, text_rect)

    def stop(self):
        self.vel = np.zeros(2)

    def limit_speed(self):
        speed = np.linalg.norm(self.vel)
        if speed > VMAX:
            self.vel = (self.vel / speed) * VMAX

    def mobility_control(self, agents: list):
        if self.route_id >= len(self.route) - 1:
            self.route_id = len(self.route) - 1
            cur_node = agents[self.route[self.route_id]]
            if (
                np.linalg.norm(cur_node.pos - self.pos) <= SENSING_RANGE
                and self.flag == 0
            ):
                self.flag = 1
                return
        if self.flag == 0:
            dest1 = agents[self.route[self.route_id]].pos
            dest2 = None
            if self.route_id + 1 <= len(self.route) - 1:
                dest2 = agents[self.route[self.route_id + 1]].pos
            self.move_to_dest(dest=dest1, agents=agents)
            if dest2 is not None and np.linalg.norm(dest2 - self.pos) <= SENSING_RANGE:
                self.route_id += 1

    def move_to_dest(self, agents, dest: np.ndarray = None, env=None):
        va = self.alignment_behaviour(dest)
        vs = self.separation_behaviour(agents)
        vo = self.obstacle_behaviour()
        self.trajectory.append(self.pos.copy())
        self.vel = va + vs + vo
        self.limit_speed()
        self.pos += self.vel

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
            / (dist_valid**2)
        )
        vo = np.sum(force, axis=0)
        return vo

    def separation_behaviour(self, agents):
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

    def reached_target(self, goal):
        distance = np.linalg.norm(goal - self.pos)
        distance = np.round(distance, 3)
        return distance <= EPS

    def generate_virtual_targets(self, agents, env):
        if self.source != -1:
            direction = agents[self.source].pos - self.pos
            phi_0 = np.arctan2(direction[1], direction[0])
        else:
            phi_0 = 0.0
        virtual_targets = []
        occupied_virtual_targets = []
        hidden_vertices = []
        for i in range(6):
            phi = phi_0 + 2 * np.pi * i / 6
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

    def is_valid_virtual_target(self, target: np.ndarray, agents: list, env):
        # not a hidden vertex
        if not env.point_is_in_environment(target):
            return False, True
        is_in_obs = env.point_is_in_obstacle(target, SIZE)
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
                if np.any(distances <= HEXAGON_RANGE):  # not in a coverage range
                    return False, False

        # not a neighbour's targets
        for other_agent in agents:
            virtual_targets = np.array([pt for pt in other_agent.virtual_targets])
            if len(virtual_targets) > 0:
                distances = np.linalg.norm(virtual_targets - target, axis=1)
                distances = np.round(distances, 5)
                if np.any(distances <= 20 * EPS):
                    return False, False

        # If behind an obstacle
        if ray_intersects_aabb(self.pos, target, env.obstacles):
            return False, True

        return True, False

    def on_occupied(self, landmarks, agents, env):
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

    def on_assigned(self, agents, landmarks):
        """
        Assignment phase.
        """
        self.mobility_control(agents)
        if self.flag == 1:
            if self.goal is not None:
                if self.reached_target(self.goal):
                    self.stop()
                    self.set_state("occupied")
                else:
                    self.move_to_dest(dest=self.goal, agents=agents)
            else:
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
                        self.set_goal(target)
                        self.source = landmark.index
                        self.is_penalty_node = is_penalty_node
                        landmark.occupied_virtual_targets[tc] = True
                        landmark.tc += 1

    def on_unassigned(self, landmarks, agents):
        """
        Unassignment phase.
        """
        self.stop()
        if len(landmarks) > 0:
            lc = landmarks[0]
            route = self.get_shortest_path(lc, agents)
            if len(route) > 0:
                route = route[1:]
                self.route = route
                self.set_state("assigned")

    def build_graph(self, agents):
        """
        Build a graph for current agent. Nodes are occupied agents' ids.

        Args:
            agents (list): list of all agents.

        Returns:
            dict: connectivity graph for current agent.
        """
        graph = {agent.index: [] for agent in agents}
        for agent in agents:
            for other_agent in agents:
                if other_agent.index == agent.index or not other_agent.is_occupied():
                    continue
                graph[agent.index].append(other_agent.index)
                graph[other_agent.index].append(agent.index)
        return graph

    def get_shortest_path(self, landmark_id: int, agents):
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
        # (current_node, path_to_current)
        queue = deque([(self.index, [self.index])])

        while queue:
            current, path = queue.popleft()
            visited.add(current)

            for neighbor in graph.get(current, []):
                if neighbor in visited:
                    continue
                if neighbor == landmark_id:
                    return path + [neighbor]
                queue.append((neighbor, path + [neighbor]))

        return []

    def step(self, landmarks, agents, env):
        if self.is_occupied():
            self.on_occupied(landmarks=landmarks, agents=agents, env=env)
        elif self.is_assigned():
            self.on_assigned(agents=agents, landmarks=landmarks)
        elif self.is_unassigned():
            self.on_unassigned(agents=agents, landmarks=landmarks)

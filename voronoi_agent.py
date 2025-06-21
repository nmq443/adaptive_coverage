import numpy as np
import pygame
from environment import Environment
from configs import *
from lloyd import lloyd
from utils import nearest_points_on_obstacles


class Agent:
    def __init__(
        self,
        index: int,
        init_pos: np.ndarray,
    ):
        self.index: int = index
        self.pos: np.ndarray = init_pos.copy()
        self.vel: np.ndarray = np.zeros(2)
        self.goal: np.ndarray = None
        self.traj: list = [init_pos.copy()]
        self.valid_points = None

    def get_travel_distance(self):
        """Get total travel distance."""
        traj = np.array(self.traj)
        if len(traj) < 2:
            return 0.0
        displacements = traj[1:] - traj[:-1]
        distances = np.linalg.norm(displacements, axis=1)
        return np.sum(distances)

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

    def goal_behaviour(self, goal: np.ndarray):
        return -KG * (self.pos - goal)

    def move_to_goal(self, goal: np.ndarray):
        self.goal = goal
        if self.terminated(goal):
            self.stop()
        else:
            self.traj.append(self.pos.copy())
            vg = self.goal_behaviour(goal)
            vo = self.obstacle_behaviour()
            self.vel = vg + vo
            self.limit_speed()
            self.pos += self.vel

    def terminated(self, goal: np.ndarray):
        return np.linalg.norm(self.pos - goal) <= EPS

    def update(self, vel: np.ndarray):
        self.vel = vel
        self.limit_speed()
        self.pos += self.vel

    def stop(self):
        self.vel = np.zeros(2)

    def limit_speed(self):
        v = np.linalg.norm(self.vel)
        if v >= VMAX:
            self.vel = self.vel / v * VMAX

    def step(self, agents: list, env: Environment):
        if self.goal is not None and not self.terminated(self.goal):
            self.move_to_goal(self.goal)
        else:
            self.goal, self.valid_points = lloyd(self, agents, env)
        # lloyd(self, agents, env)

    def render(
        self,
        surface: pygame.Surface,
        font: pygame.font.Font,
        agents: list,
        timestep: int,
    ):
        # if self.valid_points is not None:
        #     for pt in self.valid_points:
        #         pygame.draw.circle(surface, "purple", pt, 1)
        pygame.draw.circle(surface, COLOR, self.pos, SIZE)
        if self.goal is not None:
            pygame.draw.circle(surface, GOAL_COLOR, self.goal, int(SIZE / 5))
        if SHOW_SENSING_RANGE:
            pygame.draw.circle(
                surface=surface,
                center=self.pos,
                color=SENSING_COLOR,
                radius=SENSING_RANGE,
                width=2,
            )
        if SHOW_TRAJECTORY:
            for pt in self.traj:
                pygame.draw.circle(
                    surface=surface,
                    center=pt,
                    color=COLOR,
                    radius=1,
                )
        if (
            SHOW_CONNECTIONS and timestep >= 10
        ):  # only render connection links from frame 10th
            for other in agents:
                if (
                    other.index != self.index
                    and np.linalg.norm(self.pos - other.pos) < SENSING_RANGE
                ):
                    pygame.draw.line(surface, SENSING_COLOR, self.pos, other.pos)

        text_surface = font.render(str(self.index), True, "black")
        text_rect = text_surface.get_rect(
            center=(self.pos[0] + SIZE, self.pos[1] - SIZE)
        )
        surface.blit(text_surface, text_rect)

import numpy as np
import pygame
from configs import *
from lloyd import lloyd
from utils import nearest_points_on_obstacles


class Agent:
    def __init__(
        self,
        index,
        init_pos,
    ):
        self.index = index
        self.pos = init_pos.copy()
        self.vel = np.zeros(2)
        self.goal = None
        self.trajectory = [init_pos.copy()]

    def obstacle_behaviour(self):
        vo = np.zeros(2)
        if len(OBSTACLES) > 0:
            for obs in OBSTACLES:
                obs_point = nearest_points_on_obstacles(self.pos, OBSTACLES)
                obs_rel = self.pos - obs_point
                obs_dis = np.linalg.norm(obs_rel)
                if obs_dis < SENSING_RANGE:
                    vo += (
                        KO * (1 / obs_dis**2 - 1 / SENSING_RANGE**2) * obs_rel / obs_dis
                    )
        return vo

    def goal_behaviour(self, goal):
        return -KG * (self.pos - goal)

    def move_to_goal(self, goal):
        self.goal = goal
        if self.terminated(goal):
            self.stop()
        else:
            self.trajectory.append(self.pos.copy())
            vg = self.goal_behaviour(goal)
            vo = self.obstacle_behaviour()
            self.vel = vg + vo
            self.limit_speed()
            self.pos += self.vel

    def terminated(self, goal):
        return np.linalg.norm(self.pos - goal) <= EPS

    def update(self, vel):
        self.vel = vel
        self.limit_speed()
        self.pos += self.vel

    def stop(self):
        self.vel = np.zeros(2)

    def limit_speed(self):
        v = np.linalg.norm(self.vel)
        if v >= VMAX:
            self.vel = self.vel / v * VMAX

    def step(self, agents, env):
        if self.goal is not None and not self.terminated(self.goal):
            self.move_to_goal(self.goal)
        else:
            self.goal = lloyd(self, agents, env)
        # lloyd(self, agents, env)

    def render(self, surface, font, agents, timestep):
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
            for pt in self.trajectory:
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

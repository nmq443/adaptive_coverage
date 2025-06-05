import numpy as np
import pygame
from configs import *
from lloyd import lloyd


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

    def move_to_goal(self, goal):
        self.goal = goal
        if self.terminated(goal):
            self.stop()
        else:
            self.trajectory.append(self.pos.copy())
            self.vel = -KG * (self.pos - goal)
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

    def render(self, surface, font, agents):
        pygame.draw.circle(surface, COLOR,
                           self.pos, SIZE)
        if SHOW_SENSING_RANGE:
            pygame.draw.circle(
                surface=surface,
                center=self.pos,
                color=SENSING_COLOR,
                radius=SENSING_RANGE,
                width=2
            )
        if SHOW_TRAJECTORY:
            for pt in self.trajectory:
                pygame.draw.circle(
                    surface=surface,
                    center=pt,
                    color=COLOR,
                    radius=1,
                )
        if SHOW_CONNECTIONS:
            for other in agents:
                if other.index != self.index and np.linalg.norm(self.pos - other.pos) < SENSING_RANGE:
                    pygame.draw.line(surface, SENSING_COLOR, self.pos, other.pos)

        text_surface = font.render(str(self.index), True, 'black')
        text_rect = text_surface.get_rect(center=(self.pos[0] + SIZE, self.pos[1] - SIZE))
        surface.blit(text_surface, text_rect)

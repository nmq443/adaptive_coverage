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
        self.errs = []

    def move_to_goal(self, goal):
        self.goal = goal
        if self.terminated(goal):
            self.stop()
        else:
            vel = -KG * (self.pos - goal)
            err = np.linalg.norm(self.pos - goal)
            self.errs.append(err)
            self.update(vel)

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
        lloyd(self, agents, env)

    def render(self, surface, font):
        pygame.draw.circle(surface, COLOR,
                           self.pos, SIZE, width=2)
        if SHOW_SENSING_RANGE:
            pygame.draw.circle(
                surface=surface,
                center=self.pos,
                color=SENSING_COLOR,
                radius=SENSING_RANGE,
                width=2
            )
        text_surface = font.render(str(self.index), True, 'black')
        text_rect = text_surface.get_rect(center=self.pos)
        surface.blit(text_surface, text_rect)

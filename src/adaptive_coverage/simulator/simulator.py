import os
import numpy as np
import pygame
import shutil
import logging
import imageio


class Simulator:
    def __init__(self, swarm, env, result_manager, log_manager, renderer, timesteps=100, font_size=11, controller='hexagon',
                 original_method=True, screen_size=(1600, 900), fps=144, scale=20):
        # Simulation
        self.swarm = swarm
        self.font_size = font_size
        self.screen_size = screen_size
        self.controller = controller
        self.original_method = original_method
        self.scale = scale
        self.env = env
        self.running: bool = True
        self.fps = fps
        self.font = None
        self.timestep = 0
        self.timesteps = timesteps
        self.clock = pygame.time.Clock()
        self.first_click = True

        # Results and logging managers
        self.log_manager = log_manager
        self.result_manager = result_manager

        # Visualization
        self.renderer = renderer

    def init(self):
        pygame.init()
        pygame.display.set_caption("Distributed Coverage Control")
        self.screen = pygame.display.set_mode(self.screen_size)

        self.font = pygame.font.SysFont("monospace", self.font_size, True)

        # init swarm
        self.swarm.init_agents()
        self.start = True

    def loop(self):
        self.renderer.render(
            surface=self.screen,
            font=self.font,
            timestep=self.timestep
        )
        data = pygame.surfarray.array3d(self.screen)  # shape: (width, height, 3)
        frame = np.transpose(data, (1, 0, 2))  # Convert to (height, width, 3)
        if len(self.swarm.agents) > 0:
            self.result_manager.update_video(frame)
        if len(self.swarm.agents) > 0 and self.timestep == 0:
            self.result_manager.update_frames(frame)

        self.swarm.step(self.env)
        self.clock.tick(self.fps)

    def handle_input(self):
        keys = pygame.key.get_pressed()
        if keys[pygame.K_q]:
            self.running = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            if event.type == pygame.MOUSEBUTTONDOWN and self.swarm.random_init:
                mouse_pos = pygame.mouse.get_pos()
                if self.first_click:
                    self.start = True
                    self.swarm.init_agents(ref_pos=mouse_pos)
                    self.first_click = False

    def save_results(self):
        data = pygame.surfarray.array3d(self.screen)  # shape: (width, height, 3)
        frame = np.transpose(data, (1, 0, 2))  # Convert to (height, width, 3)
        self.result_manager.update_frames(frame)

        self.result_manager.save_video()
        self.result_manager.save_images()
        self.swarm.save_data()
        self.log_manager.log(f"Final lambda2 value: {self.swarm.ld2s[-1]: .2f}")

    def execute(self):
        self.init()
        while self.running:
            if self.start:
                if (self.timestep + 1) % 10 == 0:
                    self.log_manager.log(f"Time step {self.timestep + 1}/{self.timesteps}")
                if self.timestep >= self.timesteps:
                    self.running = False
                self.timestep += 1
            self.handle_input()
            self.screen.fill("white")
            self.loop()
            pygame.display.flip()
        self.save_results()
        pygame.quit()
        self.log_manager.log("Finished")

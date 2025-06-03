import os
import pygame
import imageio
from configs import *


class Simulator:
    def __init__(self, swarm, env):
        self.screen_size = SCREEN_SIZE
        self.swarm = swarm
        self.env = env
        self.running = True
        self.frames = []
        self.fps = FPS
        self.iterations = ITERATIONS
        self.font = None
        self.clock = pygame.time.Clock()
        self.limit_running = LIMIT_RUNNING
        self.first_click = True
        self.start = False

    def init(self):
        pygame.init()
        self.screen = pygame.display.set_mode(self.screen_size)
        self.font = pygame.font.SysFont("monospace", FONT_SIZE, True)
        if not RANDOM_INIT:
            self.swarm.init_agents()

    def loop(self):
        # pygame.draw.circle(self.screen, CENTER_COLOR, CENTER, CENTER_SIZE) # center of density function
        self.env.render(self.screen)
        if CONTROLLER == 'voronoi':
            self.swarm.render(self.screen, self.env, self.font)
        else:
            self.swarm.render(self.screen, self.font)
        data = pygame.surfarray.array3d(self.screen)  # shape: (width, height, 3)
        frame = np.transpose(data, (1, 0, 2))  # Convert to (height, width, 3)
        if len(self.swarm.agents) > 0:
            self.frames.append(frame)
        self.swarm.step(self.env)
        self.clock.tick(self.fps)

    def handle_input(self):
        keys = pygame.key.get_pressed()
        if keys[pygame.K_q]:
            self.running = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            if event.type == pygame.MOUSEBUTTONDOWN and RANDOM_INIT:
                mouse_pos = pygame.mouse.get_pos()
                if self.first_click:
                    self.start = True
                    self.swarm.init_agents(ref_pos=mouse_pos)
                    self.first_click = False


    def save_results(self):
        video_path = os.path.join(RES_DIR, METHOD_DIR, ENV_DIR, VIDEO_NAME)
        start_img_path = os.path.join(RES_DIR, METHOD_DIR, ENV_DIR, START_FIG)
        end_img_path = os.path.join(RES_DIR, METHOD_DIR, ENV_DIR, FINAL_FIG)
        imageio.mimsave(video_path, self.frames, fps=self.fps)
        imageio.imwrite(start_img_path, self.frames[0])
        imageio.imwrite(end_img_path, self.frames[-1])
        # self.swarm.create_plot()
 
    def execute(self):
        self.init()
        i = 0
        while self.running:
            if self.limit_running and self.start:
                if (i + 1) % 10 == 0:
                    print(f"Iteration {i + 1}/{self.iterations}")
                if i >= self.iterations:
                    self.running = False
                i += 1
            self.handle_input()
            self.screen.fill('white')
            self.loop()
            pygame.display.flip()
        if self.limit_running:
            self.save_results()
        pygame.quit()

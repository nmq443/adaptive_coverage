import os
import pygame
import shutil
import logging
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
        self.logger = logging.getLogger(__name__)
        self.first_click = True
        self.start = False
        self.res_dir = None
        self.timestep = 0

    def init(self):
        pygame.init()
        if CONTROLLER == 'hexagon':
            if ORIGINAL_METHOD:
                dir = "original"
            else:
                dir = "pso"
            self.res_dir = os.path.join(RES_DIR, METHOD_DIR, ENV_DIR, dir)
            if os.path.exists(self.res_dir):
                shutil.rmtree(self.res_dir)
            os.makedirs(self.res_dir, exist_ok=True)
            log_dir = os.path.join(LOG_DIR, METHOD_DIR, ENV_DIR, dir)
            if os.path.exists(log_dir):
                shutil.rmtree(log_dir)
            os.makedirs(log_dir, exist_ok=True)
            log_file = os.path.join(log_dir, LOG_FILE)
        else:
            self.res_dir = os.path.join(RES_DIR, METHOD_DIR, ENV_DIR)
            if os.path.exists(self.res_dir):
                shutil.rmtree(self.res_dir)
            os.makedirs(self.res_dir, exist_ok=True)
            log_dir = os.path.join(LOG_DIR, METHOD_DIR, ENV_DIR)
            if os.path.exists(log_dir):
                shutil.rmtree(log_dir)
            os.makedirs(log_dir, exist_ok=True)
            log_file = os.path.join(log_dir, LOG_FILE)
        self.screen = pygame.display.set_mode(self.screen_size)
        self.font = pygame.font.SysFont("monospace", FONT_SIZE, True)
        if not RANDOM_INIT:
            self.swarm.init_agents()
            self.start = True
        logging.basicConfig(
            format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
            datefmt='%m-%d %H:%M',
            filemode='w',
            filename=log_file, 
            level=logging.DEBUG
        )
        # define a Handler which writes INFO messages or higher to the sys.stderr
        self.console = logging.StreamHandler()
        self.console.setLevel(logging.INFO)
        # set a format which is simpler for console use
        formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
        # tell the handler to use this format
        self.console.setFormatter(formatter)
        # add the handler to the root logger
        logging.getLogger().addHandler(self.console)
        self.logger.info('Started')
        self.logger.info(f'Using {CONTROLLER} method')

    def loop(self):
        # pygame.draw.circle(self.screen, CENTER_COLOR, CENTER, CENTER_SIZE) # center of density function
        self.env.render(self.screen)
        if CONTROLLER == 'voronoi':
            self.swarm.render(self.screen, self.env, self.font, self.timestep)
        else:
            self.swarm.render(self.screen, self.font, self.timestep)
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
        video_path = os.path.join(self.res_dir, VIDEO_NAME)
        start_img_path = os.path.join(self.res_dir, START_FIG)
        end_img_path = os.path.join(self.res_dir, FINAL_FIG)
        imageio.mimsave(video_path, self.frames, fps=self.fps)
        imageio.imwrite(start_img_path, self.frames[0])
        imageio.imwrite(end_img_path, self.frames[-1])
        self.swarm.save_data(self.res_dir)
        self.logger.info(f"Final adjacent graph: \n{self.swarm.graph}")
        self.logger.info(f"Number of connection links: {int(np.sum(self.swarm.graph) / 2)}")
 
    def execute(self):
        self.init()
        i = 0
        while self.running:
            if LIMIT_RUNNING and self.start:
                self.timestep = i
                if (i + 1) % 10 == 0:
                    # print(f"Iteration {i + 1}/{self.iterations}")
                    self.logger.info(f"Iteration {i + 1}/{self.iterations}")
                if i >= self.iterations:
                    self.running = False
                i += 1
            self.handle_input()
            self.screen.fill('white')
            self.loop()
            pygame.display.flip()
        if LIMIT_RUNNING and SAVE_VIDEO:
            self.save_results()
        pygame.quit()
        self.logger.info('Finished')

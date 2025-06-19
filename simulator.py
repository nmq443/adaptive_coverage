import os
import pygame
import shutil
import logging
import imageio
from swarm import Swarm
from environment import Environment
from configs import *


class Simulator:
    def __init__(self, swarm: Swarm, env: Environment):
        self.swarm: Swarm = swarm
        self.env: Environment = env
        self.running: bool = True
        self.fps: int = FPS
        self.font: pygame.font.Font = None
        self.clock: pygame.time.Clock = pygame.time.Clock()
        self.logger = logging.getLogger(__name__)
        self.frames: list = []
        self.first_click: bool = True
        self.start: np.ndarray = False
        self.res_dir: str = None
        self.timestep: int = 0
        self.video_writer = None

    def init(self):
        pygame.init()
        pygame.display.set_caption("Distributed Coverage Control")
        self.screen = pygame.display.set_mode(SCREEN_SIZE)

        # saving directories
        if CONTROLLER == "hexagon":
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
        self.font = pygame.font.SysFont("monospace", FONT_SIZE, True)

        # init swarm
        if not RANDOM_INIT:
            self.swarm.init_agents()
            self.start = True

        # video
        if LIMIT_RUNNING and SAVE_VIDEO:
            video_path = os.path.join(self.res_dir, VIDEO_NAME)
            self.video_writer = imageio.get_writer(video_path, fps=self.fps)
            self.logger.info(f"Video writer initialized: {video_path}")

        # logging
        logging.basicConfig(
            format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s",
            datefmt="%m-%d %H:%M",
            filemode="w",
            filename=log_file,
            level=logging.DEBUG,
        )
        # define a Handler which writes INFO messages or higher to the sys.stderr
        self.console = logging.StreamHandler()
        self.console.setLevel(logging.INFO)
        # set a format which is simpler for console use
        formatter = logging.Formatter("%(name)-12s: %(levelname)-8s %(message)s")
        # tell the handler to use this format
        self.console.setFormatter(formatter)
        # add the handler to the root logger
        logging.getLogger().addHandler(self.console)
        self.logger.info("Started")
        self.logger.info(f"Using {CONTROLLER} method")

    def loop(self):
        self.env.render(self.screen)
        if CONTROLLER == "voronoi":
            self.swarm.render(self.screen, self.env, self.font, self.timestep)
        else:
            self.swarm.render(self.screen, self.font, self.timestep)
        data = pygame.surfarray.array3d(self.screen)  # shape: (width, height, 3)
        frame = np.transpose(data, (1, 0, 2))  # Convert to (height, width, 3)
        if len(self.swarm.agents) > 0 and self.video_writer is not None:
            self.video_writer.append_data(frame)
        if len(self.swarm.agents) > 0 and self.timestep == 0:
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
        data = pygame.surfarray.array3d(self.screen)  # shape: (width, height, 3)
        frame = np.transpose(data, (1, 0, 2))  # Convert to (height, width, 3)
        self.frames.append(frame)

        video_path = os.path.join(self.res_dir, VIDEO_NAME)
        start_img_path = os.path.join(self.res_dir, START_FIG)
        end_img_path = os.path.join(self.res_dir, FINAL_FIG)
        if self.video_writer is not None:
            self.video_writer.close()
            self.logger.info(f"Video saved to {video_path}")
        imageio.imwrite(start_img_path, self.frames[0])
        imageio.imwrite(end_img_path, self.frames[-1])
        self.swarm.save_data(self.res_dir)
        self.logger.info(f"Final lambda2 value: {self.swarm.ld2s[-1]: .2f}")

    def execute(self):
        self.init()
        i = 0
        while self.running:
            if LIMIT_RUNNING and self.start:
                self.timestep = i
                if (i + 1) % 10 == 0:
                    self.logger.info(f"Iteration {i + 1}/{ITERATIONS}")
                if i >= ITERATIONS:
                    self.running = False
                i += 1
            self.handle_input()
            self.screen.fill("white")
            self.loop()
            pygame.display.flip()
        if LIMIT_RUNNING and SAVE_VIDEO:
            self.save_results()
        pygame.quit()
        self.logger.info("Finished")

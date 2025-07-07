import os
import numpy as np
import pygame
import shutil
import logging
import imageio


class Simulator:
    def __init__(self, swarm, env, timesteps=100, font_size=11, env_dir="env_0", res_dir='results', log_dir='log', controller='hexagon',
                 original_method=True, screen_size=(1600, 900), fps=144,
                 save_video=False):
        # Simulation
        self.swarm = swarm
        self.font_size = font_size
        self.screen_size = screen_size
        self.controller = controller
        self.original_method = original_method
        self.env = env
        self.running: bool = True
        self.fps = fps
        self.font = None
        self.timestep = 0
        self.timesteps = timesteps
        self.clock = pygame.time.Clock()
        self.first_click = True

        # Results and logging
        self.logger = logging.getLogger(__name__)
        self.res_dir = res_dir
        self.log_dir = log_dir
        self.env_dir = env_dir
        self.frames = []
        self.video_writer = None
        self.video_path = None
        self.save_video = save_video
        self.start = False

    def init(self):
        pygame.init()
        pygame.display.set_caption("Distributed Coverage Control")
        self.screen = pygame.display.set_mode(self.screen_size)

        # saving directories
        if self.controller == "hexagon":
            if self.original_method:
                dir = "original"
            else:
                dir = "pso"
            self.res_dir = os.path.join(
                self.res_dir, self.controller, self.env_dir, f"{self.swarm.num_agents}_agents", dir
            )
            self.log_dir = os.path.join(
                self.log_dir, self.controller, self.env_dir, f"{self.swarm.num_agents}_agents", dir
            )
        else:
            self.res_dir = os.path.join(
                self.res_dir, self.controller, self.env_dir, f"{self.swarm.num_agents}_agents"
            )
            self.log_dir = os.path.join(
                self.log_dir, self.controller, self.env_dir, f"{self.swarm.num_agents}_agents"
            )
        if os.path.exists(self.res_dir):
            shutil.rmtree(self.res_dir)
        os.makedirs(self.res_dir, exist_ok=True)
        if os.path.exists(self.log_dir):
            shutil.rmtree(self.log_dir)
        os.makedirs(self.log_dir, exist_ok=True)
        log_file = os.path.join(self.log_dir, f"{self.controller}_log.log")

        self.font = pygame.font.SysFont("monospace", self.font_size, True)

        # init swarm
        self.swarm.init_agents()
        self.start = True

        # video
        if self.save_video:
            self.video_path = os.path.join(self.res_dir, f"running_video.mp4")
            self.video_writer = imageio.get_writer(self.video_path, fps=self.fps)
            self.logger.info(f"Video writer initialized: {self.video_path}")

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
        self.logger.info(f"Using {self.controller} method")

    def loop(self):
        self.env.render(self.screen)
        if self.controller == "voronoi":
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
            if event.type == pygame.MOUSEBUTTONDOWN and self.swarm.random_init:
                mouse_pos = pygame.mouse.get_pos()
                if self.first_click:
                    self.start = True
                    self.swarm.init_agents(ref_pos=mouse_pos)
                    self.first_click = False

    def save_results(self):
        data = pygame.surfarray.array3d(self.screen)  # shape: (width, height, 3)
        frame = np.transpose(data, (1, 0, 2))  # Convert to (height, width, 3)
        self.frames.append(frame)

        start_img_path = os.path.join(self.res_dir, "start_pose.png")
        end_img_path = os.path.join(self.res_dir, "final_pose.png")
        if self.video_writer is not None:
            self.video_writer.close()
            self.logger.info(f"Video saved to {self.video_path}")
        imageio.imwrite(start_img_path, self.frames[0])
        imageio.imwrite(end_img_path, self.frames[-1])
        self.swarm.save_data(self.res_dir)
        self.logger.info(f"Final lambda2 value: {self.swarm.ld2s[-1]: .2f}")

    def execute(self):
        self.init()
        while self.running:
            if self.start:
                if (self.timestep + 1) % 10 == 0:
                    self.logger.info(f"Time step {i + 1}/{self.timesteps}")
                if self.timestep >= self.timesteps:
                    self.running = False
                self.timestep += 1
            self.handle_input()
            self.screen.fill("white")
            self.loop()
            pygame.display.flip()
        if self.save_video:
            self.save_results()
        pygame.quit()
        self.logger.info("Finished")

import os
import numpy as np
import pygame


class Simulator:
    def __init__(
        self,
        swarm,
        env,
        result_manager,
        log_manager,
        timesteps=100,
        font_size=11,
        controller="hexagon",
        original_method=True,
        dt=0.1,
        screen_size=(1600, 900),
        fps=144,
        scale=20,
    ):
        # Simulation
        self.swarm = swarm
        self.font_size = font_size
        self.screen_size = screen_size
        self.controller = controller
        self.original_method = original_method
        self.scale = scale
        self.env = env
        self.running = True
        self.dt = dt
        self.fps = fps
        self.font = None
        self.timestep = 0
        self.timesteps = timesteps
        self.screen = None
        self.clock = pygame.time.Clock()
        self.first_click = True

        # Results and logging managers
        self.log_manager = log_manager
        self.result_manager = result_manager

    def init(self):
        # pygame.init()
        # pygame.display.set_caption("Distributed Coverage Control")
        # self.screen = pygame.display.set_mode(self.screen_size)
        # self.font = pygame.font.SysFont("monospace", self.font_size, True)

        # init swarm
        self.swarm.init_agents()

    def loop(self):
        # self.renderer.render(
        #     surface=self.screen, font=self.font, timestep=self.timestep
        # )
        # data = pygame.surfarray.array3d(self.screen)  # shape: (width, height, 3)
        # frame = np.transpose(data, (1, 0, 2))  # Convert to (height, width, 3)
        # if len(self.swarm.agents) > 0:
        #     self.result_manager.update_video(frame)
        # if len(self.swarm.agents) > 0 and self.timestep == 1:
        #     self.result_manager.update_frames(frame)
        self.swarm.step(self.env)

    def handle_input(self):
        keys = pygame.key.get_pressed()
        if keys[pygame.K_q]:
            self.running = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

    def save_results(self):
        # data = pygame.surfarray.array3d(self.screen)  # shape: (width, height, 3)
        # frame = np.transpose(data, (1, 0, 2))  # Convert to (height, width, 3)
        # self.result_manager.update_frames(frame)
        # self.result_manager.save_video()
        # self.result_manager.save_images()
        self.save_swarm_datas()
        self.log_manager.log(f"Final lambda2 value: {self.swarm.ld2s[-1]: .2f}")

    def save_swarm_datas(self):
        datas = []
        for agent in self.swarm.agents:
            datas.append(agent.traj)
        datas = np.array(datas)
        print(datas.shape)
        with open(self.result_manager.swarm_data_filepath, "wb") as f:
            np.save(f, datas)

        # save travel distances
        distances = self.swarm.get_travel_distance()
        with open(self.result_manager.travel_distances_filepath, "wb") as f:
            np.save(f, distances)

        # save ld2s
        ld2s = np.array(self.swarm.ld2s)
        with open(self.result_manager.ld2s_filepath, "wb") as f:
            np.save(f, ld2s)

        # save pso fitness function history
        if self.controller == "hexagon" and not self.original_method:
            for agent in self.swarm.agents:
                if agent.fitness_func_hist is not None:
                    filepath = os.path.join(
                        self.result_manager.res_dir, f"agent_no_{agent.index}.npy"
                    )
                    with open(filepath, "wb") as f:
                        np.save(f, agent.fitness_func_hist)

    def execute(self):
        self.init()
        while self.running:
            if (self.timestep + 1) % 10 == 0:
                self.log_manager.log(f"Time step {self.timestep + 1}/{self.timesteps}")
            if self.timestep >= self.timesteps:
                self.running = False
            else:
                self.timestep += 1
                self.loop()
        self.save_results()
        self.log_manager.log("Finished")
        pygame.quit()

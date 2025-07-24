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
        total_time=20,
        timestep=0.01,
        font_size=11,
        controller="hexagon",
        original_method=True,
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
        self.fps = fps
        self.font = None
        self.current_time = 0
        self.timestep = timestep
        self.total_time = total_time
        self.screen = None
        self.step_count = 0

        # Results and logging managers
        self.log_manager = log_manager
        self.result_manager = result_manager

    def init(self):
        # init swarm
        self.swarm.init_agents()

    def loop(self):
        self.swarm.step(self.env, self.timestep)

    def handle_input(self):
        keys = pygame.key.get_pressed()
        if keys[pygame.K_q]:
            self.running = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

    def save_results(self):
        self.save_swarm_datas()
        self.log_manager.log(f"Saving results to {self.result_manager.res_dir}")
        self.log_manager.log(f"Final lambda2 value: {self.swarm.ld2s[-1]: .2f}")

    def save_swarm_datas(self):
        datas = []
        for agent in self.swarm.agents:
            datas.append(agent.traj)
        datas = np.array(datas)
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
            if self.step_count % 10 == 0:
                self.log_manager.log(
                    f"Current time {self.current_time: .2f}/{self.total_time}. Step: {self.step_count}"
                )
            if self.current_time >= self.total_time:
                self.running = False
            else:
                self.current_time += self.timestep
                self.step_count += 1
                self.loop()
        self.save_results()
        self.log_manager.log("Finished")
        pygame.quit()

import os
import numpy as np
import time
from adaptive_coverage.utils.utils import plot_travel_distances, plot_ld2


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
        self.timestep = timestep
        self.total_time = total_time
        self.current_time = 0
        self.total_steps = int(total_time / timestep) - 1
        self.screen = None
        self.step_count = 0

        # Results and logging managers
        self.log_manager = log_manager
        self.result_manager = result_manager

    def init(self):
        # init swarm
        self.swarm.init_agents()

    def loop(self):
        self.swarm.step(
            env=self.env, current_step=self.step_count
        )

    def save_results(self):
        self.save_swarm_datas()
        self.log_manager.log(
            f"Saving results to {self.result_manager.res_dir}")
        self.log_manager.log(
            f"Final lambda2 value: {self.swarm.ld2s[-1]: .2f}")

    def save_swarm_datas(self):
        with open(self.result_manager.swarm_data_filepath, "wb") as f:
            np.save(f, self.swarm.state)

        # save travel distances
        distances = self.swarm.get_travel_distance()
        plot_travel_distances(distances, log=self.log_manager,
                              save_dir=self.result_manager.res_dir)
        with open(self.result_manager.travel_distances_filepath, "wb") as f:
            np.save(f, distances)

        # save ld2s
        ld2s = np.array(self.swarm.ld2s)
        plot_ld2(ld2s, log=self.log_manager,
                 save_dir=self.result_manager.res_dir)
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

        # save critical agents information
        if self.controller == "voronoi":
            with open(self.result_manager.critical_agents_filepath, "wb") as f:
                np.save(f, np.array(self.swarm.critical_agents))

        # save coverage percentage
        percentage = self.swarm.get_coverage_percentage(self.env)

        # Compute areas
        env_area = self.swarm.compute_environment_area(self.env)
        obstacle_area = self.swarm.compute_total_obstacle_area(self.env)
        free_area = max(env_area - obstacle_area, 0.0)

        # Absolute coverage area (useful for comparing different environments)
        covered_area = percentage * free_area

        # Write all to file
        output_path = os.path.join(
            self.result_manager.res_dir, "coverage_area.txt")
        with open(output_path, 'w') as f:
            f.write(f"Environment total area: {env_area:.4f}\n")
            f.write(f"Total obstacle area: {obstacle_area:.4f}\n")
            f.write(f"Total free area: {free_area:.4f}\n")
            f.write(f"Coverage area: {covered_area:.4f}\n")
            f.write(f"Coverage percentage: {percentage:.2f}\n")

        # Log summary
        self.log_manager.log(f"Environment area: {env_area:.2f}")
        self.log_manager.log(f"Obstacle area: {obstacle_area:.2f}")
        self.log_manager.log(f"Free area: {free_area:.2f}")
        self.log_manager.log(f"Coverage area: {covered_area:.2f}")
        self.log_manager.log(f"Coverage percentage: {percentage:.2f}")

    def execute(self):
        self.init()
        start = time.perf_counter()
        while self.running:
            if self.step_count % 10 == 0:
                self.log_manager.log(
                    f"Current time {self.current_time: .2f}/{self.total_time}. Step: {self.step_count}"
                )
            self.loop()
            if self.step_count >= self.total_steps:
                self.running = False
            else:
                self.current_time += self.timestep
                self.step_count += 1
        end = time.perf_counter()
        self.log_manager.log(f"Total time {end - start} seconds.")
        self.save_results()
        self.log_manager.log("Finished")

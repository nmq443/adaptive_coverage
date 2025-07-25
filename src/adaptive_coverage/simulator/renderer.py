import pygame
import numpy as np
import os
from adaptive_coverage.utils.utils import meters2pixels, ray_intersects_aabb
from adaptive_coverage.agents.cvt.lloyd import compute_voronoi_diagrams


class Renderer:
    def __init__(
        self,
        env,
        agent_size,
        sensing_range,
        scale,
        screen_size,
        trajectories_filepath,
        result_manager=None,
        log_manager=None,
        controller="voronoi",
        agent_color="red",
        agent_sensing_color="blue",
        goal_color="green",
        heading_color="green",
        index_color="black",
        obs_color="black",
        occupied_color="red",
        assigned_color="blue",
        unassigned_color="black",
        penalty_color="green",
        linewidth=1,
        fps=30,
        trail_length=100,
        font_size=11,
        show_sensing_range=False,
        show_goal=False,
        show_connections=False,
        show_trajectories=False,
    ):
        self.screen_size = screen_size
        self.sensing_range = sensing_range
        self.trajectories_filepath = trajectories_filepath
        self.env = env
        self.scale = scale
        self.agent_size = agent_size
        self.controller = controller
        self.linewidth = linewidth
        self.show_sensing_range = show_sensing_range
        self.show_goal = show_goal
        self.show_connections = show_connections
        self.show_trajectories = show_trajectories
        self.agent_color = agent_color
        self.agent_sensing_color = agent_sensing_color
        self.goal_color = goal_color
        self.obs_color = obs_color
        self.heading_color = heading_color
        self.index_color = index_color
        if self.controller == "hexagon":
            self.occupied_color = occupied_color
            self.assigned_color = assigned_color
            self.unassigned_color = unassigned_color
            self.penalty_color = penalty_color
        self.num_agents = 0
        self.num_timesteps = 0
        self.current_timestep = 0
        self.screen = None
        self.clock = None
        self.running = False
        self.fps = fps
        self.trail_length = trail_length
        self.font_size = font_size
        self.result_manager = result_manager
        self.log_manager = log_manager

    def load_data(self):
        if not os.path.exists(self.trajectories_filepath):
            print(f"Error: Trajectory file not found at {self.trajectories_filepath}")
            return False
        try:
            self.trajectories_data = np.load(self.trajectories_filepath)
            self.num_agents, self.num_timesteps, _ = self.trajectories_data.shape
            print(
                f"Loaded data: {self.num_agents} agents, {self.num_timesteps} timesteps."
            )
            return True
        except Exception as e:
            print(f"Error loading trajectory data: {e}")
            return False

    def init_pygame(self):
        pygame.init()
        pygame.display.set_caption("Playback")
        self.screen = pygame.display.set_mode(self.screen_size)
        self.clock = pygame.time.Clock()
        self.running = True
        self.font = pygame.font.SysFont("monospace", self.font_size, True)

    def draw(self):
        self.screen.fill("white")

        for i in range(self.num_agents):
            current_pos_sim = self.trajectories_data[i, self.current_timestep, :-1]
            yaw = self.trajectories_data[i, self.current_timestep, -1]

            # Draw agents
            self.draw_agent(i, current_pos_sim)

            # Draw heading
            self.draw_heading(current_pos_sim, yaw)

            # Draw trails
            # self.draw_trails(i)

            # Draw sensing range
            if self.show_sensing_range:
                self.draw_sensing_range(current_pos_sim)

            # Draw connection links
            if self.show_connections:
                self.draw_connections(i, current_pos_sim)

        # Draw environment
        self.draw_environment()

        # Draw voronoi partitions (for voronoi agent)
        if self.controller == "voronoi":
            generators = self.trajectories_data[:, self.current_timestep, :-1]
            vor = compute_voronoi_diagrams(generators, self.env)
            self.draw_voronoi(vor, self.screen)

        if self.result_manager is not None and self.log_manager is not None:
            data = pygame.surfarray.array3d(self.screen)  # shape: (width, height, 3)
            frame = np.transpose(data, (1, 0, 2))  # Convert to (height, width, 3)
            self.result_manager.update_video(frame)
            if self.current_timestep == 0:
                self.result_manager.update_frames(frame)
        pygame.display.flip()

    def draw_heading(self, pos, yaw):
        length = 2 * self.agent_size
        start_pos = meters2pixels(pos, self.scale)
        end_pos = pos + length * np.array([np.cos(yaw), np.sin(yaw)])
        end_pos = meters2pixels(end_pos, self.scale)
        pygame.draw.line(
            self.screen, self.heading_color, start_pos, end_pos, self.linewidth
        )

    def draw_environment(self):
        # Draw environment
        for i, edge in enumerate(self.env.edges):
            start_pos = meters2pixels(edge[0], self.scale)
            end_pos = meters2pixels(edge[1], self.scale)
            pygame.draw.line(
                self.screen, self.obs_color, start_pos, end_pos, self.linewidth
            )

        for obs in self.env.obstacles:
            rect = np.array([obs[0], obs[1], obs[2], obs[3]])
            rect = meters2pixels(rect, self.scale)
            pygame.draw.rect(self.screen, self.obs_color, pygame.rect.Rect(rect))

    def draw_agent(self, index, pos):
        pos = meters2pixels(pos, self.scale)
        agent_size = meters2pixels(self.agent_size, self.scale)
        pygame.draw.circle(self.screen, self.agent_color, pos, agent_size)
        # Render agent's index
        text_surface = self.font.render(str(index), True, self.index_color)
        text_rect = text_surface.get_rect(
            center=(pos[0] + agent_size, pos[1] - agent_size)
        )
        self.screen.blit(text_surface, text_rect)

    def draw_sensing_range(self, pos):
        pos = meters2pixels(pos, self.scale)
        sensing_range = meters2pixels(self.sensing_range, self.scale)
        pygame.draw.circle(
            self.screen, self.agent_sensing_color, pos, sensing_range, self.linewidth
        )

        # If voronoi agent, draw critical range
        # if self.controller == "voronoi":
        #     critical_range = meters2pixels(self.sensing_range * 0.75, self.scale)
        #     pygame.draw.circle(
        #         self.screen,
        #         self.agent_sensing_color,
        #         pos,
        #         critical_range,
        #         self.linewidth,
        #     )

    def draw_trails(self, index):
        start_trail_idx = max(0, self.current_timestep - self.trail_length)
        trail_points_sim = self.trajectories_data[
            index, start_trail_idx : self.current_timestep + 1, :-1
        ]
        trail_points_screen = []
        for j in range(trail_points_sim.shape[0]):
            trail_points_screen.append(meters2pixels(trail_points_sim[j], self.scale))

        if len(trail_points_screen) > 1:
            pygame.draw.lines(
                self.screen,
                self.agent_color,
                False,
                trail_points_screen,
                self.linewidth,
            )

    def draw_connections(self, i, pos):
        for j in range(self.num_agents):
            if i == j:
                continue
            other_pos_sim = self.trajectories_data[j, self.current_timestep, :-1]
            if ray_intersects_aabb(pos, other_pos_sim, self.env.obstacles):
                continue
            dist = np.linalg.norm(pos - other_pos_sim)
            if dist <= self.sensing_range:
                start_pos = meters2pixels(pos, self.scale)
                end_pos = meters2pixels(other_pos_sim, self.scale)
                pygame.draw.line(
                    self.screen,
                    self.agent_color,
                    start_pos,
                    end_pos,
                    self.linewidth,
                )

    def draw_voronoi(self, vor, surface):
        """
        Draw voronoi on screen.

        Args:
            vor (scipy.spatial.Voronoi): voronoi partition.
            surface (pygame.Surface): surface to render on.
        """
        # Plot ridges
        for region in vor.filtered_regions:
            vertices = vor.vertices[region + [region[0]], :]
            for i in range(len(vertices) - 1):
                start_pos = meters2pixels(vertices[i], self.scale)
                end_pos = meters2pixels(vertices[i + 1], self.scale)
                pygame.draw.line(surface, "black", start_pos, end_pos, self.linewidth)
            start_pos = meters2pixels(vertices[-1], self.scale)
            end_pos = meters2pixels(vertices[0], self.scale)
            pygame.draw.line(surface, "black", start_pos, end_pos, self.linewidth)

    def run(self):
        if not self.load_data():
            return

        self.init_pygame()

        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        self.running = False
                    elif event.key == pygame.K_SPACE:  # Pause/unpause with spacebar
                        self.running = False  # Temporary, just to break the loop. A better pause would be implemented.
                        # For a true pause, you'd toggle a 'paused' flag and skip current_timestep increment.
                        # Example: `if not self.paused: self.current_timestep += 1`

            if self.current_timestep < self.num_timesteps:
                self.draw()
                self.current_timestep += 1
            else:
                # Loop playback or stop when finished
                # self.current_timestep = 0 # Uncomment to loop
                # print("Playback finished.")
                self.running = False  # Stop when finished

            self.clock.tick(self.fps)

        self.save()
        pygame.quit()

    def save(self):
        if self.result_manager is not None and self.log_manager is not None:
            data = pygame.surfarray.array3d(self.screen)  # shape: (width, height, 3)
            frame = np.transpose(data, (1, 0, 2))  # Convert to (height, width, 3)
            self.result_manager.update_frames(frame)

            self.result_manager.save_video()
            self.result_manager.save_images()

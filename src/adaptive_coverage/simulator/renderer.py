import pygame
import numpy as np
import os
from adaptive_coverage.utils.utils import meters2pixels, ray_intersects_aabb
from adaptive_coverage.agents.cvt.lloyd import compute_voronoi_diagrams
from adaptive_coverage.environment.environment import Environment
from adaptive_coverage.simulator.data_manager import ResultManager, LogManager
from typing import Union


class Renderer:
    def __init__(
        self,
        env,
        agent_size,
        critical_ratio,
        sensing_range,
        scale,
        screen_size,
        trajectories_filepath,
        result_manager,
        log_manager,
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
        """
        This class renders the result of the simulation.
        Args:
            env: simulation environment.
            agent_size: size of the agent in meters, because an agent is represented as a circle, agent_size is the
            circle's radius.
            sensing_range: sensing range of the agent in meters.
            critical_ratio: used to determine critical range when using voronoi method.
            scale: scaling factor from simulation plane to rendering plane.
            screen_size: size of the rendering screen in pixels.
            trajectories_filepath: filepath to result data.
            result_manager: result manager object (for result tree).
            log_manager: log manager object (for log tree).
            controller: controller used for simulation (voronoi or hexagon).
            agent_color: agent's color.
            agent_sensing_color: agent's sensing range's color.
            goal_color: agent's goal's color.
            heading_color: agent's heading's color.
            index_color: agent's index's color.
            obs_color: obstacle's color.
            occupied_color: agent's color if controller is hexagon and agent's state is occupied.
            assigned_color: agent's color if controller is hexagon and agent's state is assigned.
            unassigned_color: agent's color if controller is hexagon and agent's state is unassigned.
            penalty_color: agent's color if controller is hexagon and agent is a penalty node.
            linewidth: line width to render.
            fps: frames per second.
            trail_length: trail length of the agent's trajectory.
            font_size: font size of the agent's index.
            show_sensing_range: show agent's sensing range or not.
            show_goal: show agent's goal or not.
            show_connections: show agent's connections or not.
            show_trajectories: show agent's trajectory or not.
        """
        self.screen_size: tuple = screen_size
        self.sensing_range: float = sensing_range
        self.critical_range: float = self.sensing_range * critical_ratio
        self.trajectories_filepath: str = trajectories_filepath
        self.env: Environment = env
        self.scale: float = scale
        self.agent_size: float = agent_size
        self.controller: str = controller
        self.linewidth: int = linewidth
        self.show_sensing_range: bool = show_sensing_range
        self.show_goal: bool = show_goal
        self.show_connections: bool = show_connections
        self.show_trajectories: bool = show_trajectories
        self.agent_color: Union[pygame.color.Color, str] = agent_color
        self.agent_sensing_color: Union[pygame.color.Color,
                                        str] = agent_sensing_color
        self.goal_color: Union[pygame.color.Color, str] = goal_color
        self.obs_color: Union[pygame.color.Color, str] = obs_color
        self.heading_color: Union[pygame.color.Color, str] = heading_color
        self.index_color: Union[pygame.color.Color, str] = index_color
        if self.controller == "hexagon":
            self.occupied_color: Union[pygame.color.Color,
                                       str] = occupied_color
            self.assigned_color: Union[pygame.color.Color,
                                       str] = assigned_color
            self.unassigned_color: Union[pygame.color.Color,
                                         str] = unassigned_color
            self.penalty_color: Union[pygame.color.Color, str] = penalty_color
        self.num_agents: int = 0
        self.num_timesteps: int = 0
        self.current_timestep: int = 0
        self.screen: pygame.Surface | None = None
        self.clock: pygame.time.Clock | None = None
        self.running: bool = False
        self.fps: int = fps
        self.trail_length: int = trail_length
        self.font_size: int = font_size
        self.result_manager: ResultManager = result_manager
        self.log_manager: LogManager = log_manager

    def load_data(self):
        """
        Load simulation data. The data is a .npy file, which contains a (num_agents, num_timesteps, 3) numpy.ndarray
        representing the agents' states.
        """
        if not os.path.exists(self.trajectories_filepath):
            print(
                f"Error: Trajectory file not found at {self.trajectories_filepath}")
            return False
        try:
            self.trajectories_data = np.load(self.trajectories_filepath)
            self.num_agents, self.num_timesteps, _ = self.trajectories_data.shape
            return True
        except Exception as e:
            print(f"Error loading trajectory data: {e}")
            return False

    def init_pygame(self):
        """
        Initialize pygame window.
        """
        pygame.init()
        pygame.display.set_caption("Playback")
        self.screen = pygame.display.set_mode(self.screen_size)
        self.clock = pygame.time.Clock()
        self.running = True
        self.font = pygame.font.SysFont("monospace", self.font_size, True)

    def draw(self):
        """
        Rendering at each frame.
        """
        if self.screen is not None:
            self.screen.fill("white")

            for i in range(self.num_agents):
                current_pos_sim = self.trajectories_data[i,
                                                         self.current_timestep, :-2]
                penalty_flag = self.trajectories_data[i,
                                                      self.current_timestep, -1]
                yaw = self.trajectories_data[i, self.current_timestep, -1]

                # Draw agents
                self.draw_agent(i, current_pos_sim, penalty_flag)

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
                generators = self.trajectories_data[:,
                                                    self.current_timestep, :-2]
                vor = compute_voronoi_diagrams(generators, self.env)
                self.draw_voronoi(vor)

            if self.result_manager is not None and self.log_manager is not None:
                data = pygame.surfarray.array3d(
                    self.screen)  # shape: (width, height, 3)
                # Convert to (height, width, 3)
                frame = np.transpose(data, (1, 0, 2))
                self.result_manager.update_video(frame)
                if self.current_timestep == 0:
                    self.result_manager.update_frames(frame)
            pygame.display.flip()

    def draw_heading(self, pos, yaw):
        """
        Render the heading of the agent.

        Args:
            pos: agent's simulation plane's position.
            yaw: agent's simulation plane's heading angle.
        """
        if self.screen is not None:
            length = 2 * self.agent_size
            start_pos = meters2pixels(pos, self.scale)
            end_pos = pos + length * np.array([np.cos(yaw), np.sin(yaw)])
            end_pos = meters2pixels(end_pos, self.scale)
            pygame.draw.line(
                self.screen, self.heading_color, start_pos, end_pos, self.linewidth
            )

    def draw_environment(self):
        """
        Render the environment.
        """
        if self.screen is not None:
            for i, edge in enumerate(self.env.edges):
                start_pos = meters2pixels(edge[0], self.scale)
                end_pos = meters2pixels(edge[1], self.scale)
                pygame.draw.line(
                    self.screen, self.obs_color, start_pos, end_pos, self.linewidth
                )

            for obs in self.env.obstacles:
                rect = np.array([obs[0], obs[1], obs[2], obs[3]])
                rect = meters2pixels(rect, self.scale)
                pygame.draw.rect(self.screen, self.obs_color,
                                 pygame.rect.Rect(rect))

    def draw_agent(self, index, pos, penalty_flag=0):
        """
        Render the agent.

        Args:
            index: index of agent.
            pos: agent's simulation plane's position.
        """
        if self.screen is not None:
            pos = meters2pixels(pos, self.scale)
            agent_size = meters2pixels(self.agent_size, self.scale)
            if penalty_flag == 1:
                pygame.draw.circle(
                    self.screen, self.penalty_color, pos, agent_size)
            else:
                pygame.draw.circle(
                    self.screen, self.agent_color, pos, agent_size)
            # Render agent's index
            text_surface = self.font.render(str(index), True, self.index_color)
            text_rect = text_surface.get_rect(
                center=(pos[0] + agent_size, pos[1] - agent_size)
            )
            self.screen.blit(text_surface, text_rect)

    def draw_sensing_range(self, pos):
        """
        Render agent's sensing range.

        Args:
            pos: simulation plane's position.
        """
        if self.screen is not None:
            pos = meters2pixels(pos, self.scale)
            sensing_range = meters2pixels(self.sensing_range, self.scale)
            pygame.draw.circle(
                self.screen, self.agent_sensing_color, pos, sensing_range, self.linewidth*2
            )

            # If voronoi agent, draw critical range
            if self.controller == "voronoi":
                critical_range = meters2pixels(
                    self.critical_range, self.scale)
                pygame.draw.circle(
                    self.screen,
                    self.agent_sensing_color,
                    pos,
                    critical_range,
                    self.linewidth,
                )

    def draw_trails(self, index):
        """
        Render agent's trajectory trails.

        Args:
            index: agent's index
        """
        if self.screen is not None:
            start_trail_idx = max(0, self.current_timestep - self.trail_length)
            trail_points_sim = self.trajectories_data[
                index, start_trail_idx: self.current_timestep + 1, :-1
            ]
            trail_points_screen = []
            for j in range(trail_points_sim.shape[0]):
                trail_points_screen.append(
                    meters2pixels(trail_points_sim[j], self.scale))

            if len(trail_points_screen) > 1:
                pygame.draw.lines(
                    self.screen,
                    self.agent_color,
                    False,
                    trail_points_screen,
                    self.linewidth,
                )

    def draw_connections(self, i, pos):
        """
        Render agent's connections.

        Args:
            i: agent's index.
            pos: agent's simulation plane's position.
        """
        if self.screen is not None:
            for j in range(self.num_agents):
                if i == j:
                    continue
                other_pos_sim = self.trajectories_data[j,
                                                       self.current_timestep, :-2]
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

    def draw_voronoi(self, vor):
        """
        Draw voronoi on screen.

        Args:
            vor: voronoi partition.
        """
        if self.screen is not None:
            # Plot ridges
            for region in vor.filtered_regions:
                vertices = vor.vertices[region + [region[0]], :]
                for i in range(len(vertices) - 1):
                    start_pos = meters2pixels(vertices[i], self.scale)
                    end_pos = meters2pixels(vertices[i + 1], self.scale)
                    pygame.draw.line(self.screen, "black", start_pos,
                                     end_pos, self.linewidth)
                start_pos = meters2pixels(vertices[-1], self.scale)
                end_pos = meters2pixels(vertices[0], self.scale)
                pygame.draw.line(self.screen, "black", start_pos,
                                 end_pos, self.linewidth)

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
                        # Temporary, just to break the loop. A better pause would be implemented.
                        self.running = False
                        # For a true pause, you'd toggle a 'paused' flag and skip current_timestep increment.
                        # Example: `if not self.paused: self.current_timestep += 1`

            if self.current_timestep < self.num_timesteps:
                self.draw()
                self.current_timestep += 1
            else:
                # Loop playback or stop when finished
                # self.current_timestep = 0 # Uncomment to loop
                self.running = False  # Stop when finished

            if self.clock is not None:
                self.clock.tick(self.fps)

        self.save()
        pygame.quit()

    def save(self):
        """
        Save the playback video.
        """
        if self.result_manager is not None and self.log_manager is not None and self.screen is not None:
            data = pygame.surfarray.array3d(
                self.screen)  # shape: (width, height, 3)
            # Convert to (height, width, 3)
            frame = np.transpose(data, (1, 0, 2))
            self.result_manager.update_frames(frame)

            self.result_manager.save_video()
            self.result_manager.save_images()
            self.log_manager.log(
                f"Saved results to {self.result_manager.res_dir}")

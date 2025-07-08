import numpy as np
import pygame
from adaptive_coverage.utils.utils import meters2pixels


class Agent:
    def __init__(
            self,
            index,
            init_pos,
            size,
            path_planner,
            sensing_range,
            result_manager,
            sensing_color='blue',
            color='red',
            show_goal=False,
            show_connections=False,
            show_sensing_range=False,
            show_trajectory=False,
            v_max=0.05,
            avoidance_range=0.05,
            tolerance=0.05
    ):
        # Basic parameters
        self.index = index
        self.pos = init_pos.copy()
        self.theta = 0
        self.vel = np.zeros(2)
        self.size = size
        self.v_max = v_max
        self.sensing_range = sensing_range

        # Path planning parameters
        self.path_planner = path_planner
        self.avoidance_range = avoidance_range + size * 2
        self.tolerance = tolerance

        # Visualization parameters
        self.sensing_color = sensing_color
        self.color = color
        self.show_goal = show_goal
        self.show_connections = show_connections
        self.show_sensing_range = show_sensing_range
        self.show_trajectory = show_trajectory
        self.goal = None
        self.traj = [init_pos.copy()]

        # Result manager
        self.result_manager = result_manager

    def get_travel_distance(self):
        """Get total travel distance."""
        traj = np.array(self.traj)
        if len(traj) < 2:
            return 0.0
        displacements = traj[1:] - traj[:-1]
        distances = np.linalg.norm(displacements, axis=1)
        return np.sum(distances)

    def move_to_goal(self, goal, agents, obstacles):
        self.goal = goal
        if self.terminated(goal):
            self.stop()
        else:
            self.traj.append(self.pos.copy())
            self.vel = self.path_planner.total_force(self.pos, self.goal, self.index, agents, obstacles)
            self.limit_speed()
            self.pos += self.vel

    def terminated(self, goal):
        return np.linalg.norm(self.pos - goal) <= self.tolerance

    def update(self, vel):
        self.vel = vel
        self.limit_speed()
        self.pos += self.vel

    def stop(self):
        self.vel = np.zeros(2)

    def limit_speed(self):
        v = np.linalg.norm(self.vel)
        if v >= self.v_max:
            self.vel = self.vel / v * self.v_max

    def step(self, *args, **kwargs):
        pass

    def render(
            self,
            surface: pygame.Surface,
            font: pygame.font.Font,
            agents: list,
            timestep: int,
            scale
    ):
        agent_pos = meters2pixels(self.pos, scale)
        agent_size = meters2pixels(self.size, scale)
        sensing_range = meters2pixels(self.sensing_range, scale)

        pygame.draw.circle(surface, self.color, agent_pos, agent_size)
        yaw = np.arctan2(self.vel[1], self.vel[0])
        length = 2 * self.size
        end = self.pos + length * np.array([np.cos(yaw), np.sin(yaw)])

        start_pos = meters2pixels(self.pos, scale)
        end_pos = meters2pixels(end, scale)
        pygame.draw.line(surface, "green", start_pos, end_pos, 2)

        if self.goal is not None and self.show_goal:
            goal = meters2pixels(self.goal, scale)
            goal_size = meters2pixels(self.size / 2, scale)
            pygame.draw.circle(surface, self.show_goal, goal, goal_size)

        if self.show_sensing_range:
            pygame.draw.circle(
                surface=surface,
                center=agent_pos,
                color=self.sensing_color,
                radius=sensing_range,
                width=2,
            )

        if self.show_trajectory:
            for pt in self.traj:
                traj_pt = meters2pixels(pt, scale)
                pygame.draw.circle(
                    surface=surface,
                    center=traj_pt,
                    color=self.color,
                    radius=1,
                )
        if (self.show_connections and timestep >= 10):  # only render connection links from frame 10th
            for other in agents:
                if (
                        other.index != self.index
                        and np.linalg.norm(self.pos - other.pos) < self.sensing_range
                ):
                    start_pos = meters2pixels(other.pos, scale)
                    end_pos = meters2pixels(self.pos, scale)
                    pygame.draw.line(surface, self.sensing_color, start_pos, end_pos)

        text_surface = font.render(str(self.index), True, "black")
        text_rect = text_surface.get_rect(
            center=(agent_pos[0] + agent_size, agent_pos[1] - agent_size)
        )
        surface.blit(text_surface, text_rect)

import pygame
import numpy as np
from adaptive_coverage.utils.utils import meters2pixels
from adaptive_coverage.agents.cvt.lloyd import compute_voronoi_diagrams


class Renderer:
    def __init__(
        self,
        swarm,
        env,
        scale,
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
        show_sensing_range=False,
        show_goal=False,
        show_connections=False,
        show_trajectories=False,
    ):
        self.swarm = swarm
        self.env = env
        self.scale = scale
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

    def render(self, surface, font, timestep):
        # Render the swarm
        for agent in self.swarm.agents:
            agent_pos = meters2pixels(agent.pos, self.scale)
            agent_size = meters2pixels(agent.size, self.scale)
            agent_sensing_range = meters2pixels(agent.sensing_range, self.scale)

            # Render agent
            if self.controller == "voronoi":
                pygame.draw.circle(surface, self.agent_color, agent_pos, agent_size)
            else:
                if agent.is_occupied():
                    color = self.occupied_color
                elif agent.is_assigned():
                    color = self.assigned_color
                elif agent.is_unassigned():
                    color = self.unassigned_color
                if agent.is_penalty_node:
                    color = self.penalty_color
                pygame.draw.circle(surface, color, agent_pos, agent_size)

            # Render heading
            yaw = np.arctan2(agent.vel[1], agent.vel[0])
            length = 2 * agent_size
            start_pos = agent_pos
            end_pos = agent_pos + length * np.array([np.cos(yaw), np.sin(yaw)])
            pygame.draw.line(
                surface, self.heading_color, start_pos, end_pos, self.linewidth
            )

            # Render agent's index
            text_surface = font.render(str(agent.index), True, self.index_color)
            text_rect = text_surface.get_rect(
                center=(agent_pos[0] + agent_size, agent_pos[1] - agent_size)
            )
            surface.blit(text_surface, text_rect)

            # Render connection links (only render from 10th step)
            if self.show_connections and timestep >= 10:
                for other in self.swarm.agents:
                    dist = np.linalg.norm(agent.pos - other.pos)
                    if other.index != agent.index and dist <= agent.sensing_range:
                        start_pos = agent_pos
                        end_pos = meters2pixels(other.pos, self.scale)
                        pygame.draw.line(
                            surface,
                            agent.sensing_color,
                            start_pos,
                            end_pos,
                            self.linewidth,
                        )

            # Render trajectories
            if self.show_trajectories:
                for pt in agent.traj:
                    traj_pt = meters2pixels(pt, self.scale)
                    pygame.draw.circle(
                        surface=surface,
                        center=traj_pt,
                        color=self.agent_color,
                        radius=int(agent_size / 4),
                    )

            # Render goals
            if agent.goal is not None and self.show_goal:
                goal_pos = meters2pixels(agent.goal, self.scale)
                pygame.draw.circle(
                    surface, self.goal_color, goal_pos, int(agent_size / 2)
                )

            # Render sensing range
            if self.show_sensing_range:
                pygame.draw.circle(
                    surface,
                    self.agent_sensing_color,
                    agent_pos,
                    agent_sensing_range,
                    self.linewidth,
                )

        # Render the environment
        for i, edge in enumerate(self.env.edges):
            start_pos = meters2pixels(edge[0], self.scale)
            end_pos = meters2pixels(edge[1], self.scale)
            pygame.draw.line(
                surface, self.obs_color, start_pos, end_pos, self.linewidth
            )

        for obs_rect in self.env.obstacles_rects:
            rect = np.array([obs_rect[0], obs_rect[1], obs_rect[2], obs_rect[3]])
            rect = meters2pixels(rect, self.scale)
            pygame.draw.rect(surface, self.obs_color, pygame.rect.Rect(rect))

        # Render voronoi partitions (for voronoi agent)
        if self.controller == "voronoi":
            generators = np.array([agent.pos for agent in self.swarm.agents])
            vor = compute_voronoi_diagrams(generators, self.env)
            self.draw_voronoi(vor, surface)

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

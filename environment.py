import pygame
from shapely.geometry import Polygon, Point
from configs import *


class Environment:
    def __init__(self):
        self.vertices = VERTICES
        self.x_min = np.min(self.vertices[:, 0])
        self.x_max = np.max(self.vertices[:, 0])
        self.y_min = np.min(self.vertices[:, 1])
        self.y_max = np.max(self.vertices[:, 1])
        self.polygon = Polygon(self.vertices)
        self.edges = []
        self.obstacles = OBSTACLES
        self.obstacles_rects = []
        self.init()

    def init(self):
        for obs in self.obstacles:
            self.obstacles_rects.append(pygame.rect.Rect(obs[0], obs[1], obs[2], obs[3]))
        for i in range(len(self.vertices) - 1):
            edge_start = self.vertices[i]
            edge_end = self.vertices[i + 1]
            self.edges.append([edge_start, edge_end])
        edge_start = self.vertices[-1]
        edge_end = self.vertices[0]
        self.edges.append([edge_start, edge_end])

    def render(self, surface):
        for edge in self.edges:
            pygame.draw.line(surface, 'black', edge[0], edge[1], LINE_WIDTH)
        for obs_rect in self.obstacles_rects:
            pygame.draw.rect(
                surface, 'black', obs_rect)

    def contains(self, point):
        shapely_point = Point(point)
        if self.polygon.contains(shapely_point):
            return True
        if self.polygon.boundary.distance(shapely_point) < meters2pixels(1e-2, SCALE):
            return True
        return False

    # def point_is_in_obstacle(self, point):
    #     for obs in self.obstacles:
    #         x, y, w, h = obs
    #         if x <= point[0] <= x + w and y <= point[1] <= y + h:
    #             return True
    #     return False
    def point_is_in_obstacle(self, point: np.ndarray, agent_radius: float):
        """Obstacle is a rectangle with (x, y, width, height) values"""
        if len(self.obstacles) <= 0:
            return False
        x = self.obstacles[:, 0]
        y = self.obstacles[:, 1]
        w = self.obstacles[:, 2]
        h = self.obstacles[:, 3]

        x_min = x
        y_min = y
        x_max = x + w
        y_max = y + h

        closest_x = np.maximum(x_min, np.minimum(point[0], x_max))
        closest_y = np.maximum(y_min, np.minimum(point[1], y_max))

        # Calculate distance from the closest points to the circle center
        dx = closest_x - point[0]
        dy = closest_y - point[1]
        distance_squared = dx ** 2 + dy ** 2

        # Check if the closest point is within the circle's radius
        intersects = distance_squared <= agent_radius ** 2
        # print(f"Distances squared: {distance_squared}, intersects: {intersects}")

        # Return True if the circle intersects with any obstacle
        return np.any(intersects)

    def point_is_in_environment(self, point, agent_radius):
        if point[0] < self.x_min or point[0] > self.x_max:
            return False
        if point[1] < self.y_min or point[1] > self.y_max:
            return False
        return True
        '''
        if point[0] - agent_radius < self.x_min or point[0] + agent_radius > self.x_max:
            return False
        if point[1] - agent_radius < self.y_min or point[1] + agent_radius > self.y_max:
            return False
        return True
        '''
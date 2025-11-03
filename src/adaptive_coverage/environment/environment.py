import numpy as np
from shapely.geometry import Polygon, Point
from adaptive_coverage.utils.utils import meters2pixels


class Environment:
    def __init__(self, area_width, area_height, vertices, obstacles, offset=0.1):
        self.area_width = area_width
        self.area_height = area_height
        self.offset = offset
        # self.vertices = np.array(
        #     [
        #         [0 + self.offset, 0 + self.offset],
        #         [self.area_width - self.offset, 0 + self.offset],
        #         [self.area_width - self.offset, area_height - offset],
        #         [0 + self.offset, self.area_height - self.offset],
        #     ],
        #     dtype=float,
        # )
        self.vertices = np.array(vertices)
        self.x_min = np.min(self.vertices[:, 0])
        self.x_max = np.max(self.vertices[:, 0])
        self.y_min = np.min(self.vertices[:, 1])
        self.y_max = np.max(self.vertices[:, 1])
        self.polygon = Polygon(self.vertices)
        self.edges = []
        self.obstacles = np.array(obstacles)
        self.obstacles_rects = []
        self.init()

    def init(self):
        """
        Initialize the obstacles.
        """
        for obs in self.obstacles:
            self.obstacles_rects.append(
                (obs[0], obs[1], obs[2], obs[3])
            )
        for i in range(len(self.vertices) - 1):
            edge_start = self.vertices[i]
            edge_end = self.vertices[i + 1]
            self.edges.append([edge_start, edge_end])
        edge_start = self.vertices[-1]
        edge_end = self.vertices[0]
        self.edges.append([edge_start, edge_end])

    def point_is_in_environment(self, point, agent_size):
        """
        Check if an agent with given size is inside the environment.

        Args:
            point: (x, y) position in 2D of the agent.

        Returns:
            True if the entire circle is within or touches the polygon.
        """
        circle = Point(point).buffer(agent_size)

        # Check if circle is entirely within polygon
        if self.polygon.contains(circle):
            return True

        # Check if circle touches or intersects the polygon (i.e., partially in)
        if self.polygon.intersects(circle):
            return False

        return False

    def contains(self, point: np.ndarray):
        """
        Check if the environment contains this point or not.

        Args:
            point (numpy.ndarray): point to check.

        Returns:
            bool: point is contained or not.
        """
        shapely_point = Point(point)
        if self.polygon.contains(shapely_point):
            return True
        if self.polygon.boundary.distance(shapely_point) < 1e-2:
            return True
        return False

    def point_is_in_obstacle(self, point, agent_size):
        """
        Check if point is in any obstacle. Obstacle is a rectangle with (x, y, width, height) values.

        Args:
            point (numpy.ndarray): point to check.

        Returns:
            bool: if point is in any obstacle.
        """
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
        distance_squared = dx**2 + dy**2

        # Check if the closest point is within the circle's radius
        intersects = distance_squared <= agent_size**2

        # Return True if the circle intersects with any obstacle
        return np.any(intersects)

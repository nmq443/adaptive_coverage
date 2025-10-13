import numpy as np
from adaptive_coverage.utils.utils import nearest_points_on_obstacles, normalize


class ArtificialPotentialField:
    def __init__(self, kg, ko, kc, beta_c, sensing_range, avoidance_range, agent_size):
        self.kg = kg
        self.ko = ko
        self.kc = kc
        self.beta_c = beta_c
        self.agent_size = agent_size
        self.sensing_range = sensing_range
        self.avoidance_range = avoidance_range

    def goal_force(self, pos, goal):
        """
        Heading to goal behavior.

        Args:
            pos: current position.
            goal: desired goal.

        Returns:
            Force to desired goal.
        """
        return self.kg * (goal - pos)

    def obstacle_force(self, pos, obstacles):
        """
        Obstacles avoidance behaviors.

        Args:
            pos: current position.
            obstacles: list of obstacles, represented as a (n, 4) array.

        Returns:
            Force to avoid obstacles.
        """
        if len(obstacles) == 0:
            return np.zeros(2)

        obs_pts = nearest_points_on_obstacles(pos, obstacles)
        diff = pos - obs_pts
        dist = np.linalg.norm(diff, axis=1) - self.agent_size
        dist = np.clip(dist, 1e-4, None)  # avoid division by zero

        mask = dist < self.avoidance_range
        if not np.any(mask):
            return np.zeros(2)

        diff_valid = diff[mask]
        dist_valid = dist[mask][:, np.newaxis]

        force = (
            self.ko
            * (1.0 / dist_valid - 1.0 / self.avoidance_range)
            * diff_valid
            / (dist_valid**2)
        )
        fo = np.sum(force, axis=0)
        return fo

    def collision_force(self, pos, agent_index, agents):
        """
        Collision avoidance force.

        Args:
            pos: current position.
            agent_index: current agent's index.
            agents: list of all agents.

        Returns:
            Force to avoid collision.
        """
        positions = np.array(
            [agent.pos for agent in agents if agent.index != agent_index]
        )
        if positions.size == 0:
            return np.zeros(2)

        directions = positions - pos  # Vector from self to others
        # Compute all distances at once
        distances = np.linalg.norm(directions, axis=1)
        mask = distances <= self.sensing_range  # Only consider nearby agents

        if not np.any(mask):
            return np.zeros(2)

        # Compute avoidance vector only for close agents
        directions = directions[mask]
        distances = distances[mask][:, np.newaxis]  # Reshape for broadcasting

        fc = np.sum(
            (
                self.kc
                * np.exp(-self.beta_c * (distances - self.avoidance_range))
                * (directions / distances + 1e-9)
                / (distances - self.avoidance_range + 1e-9)
            ),
            axis=0,
        )
        return fc

    def total_force(self, pos, goal, agent_index, agents, obstacles):
        """Find total force based on 3 behaviors: goal, collision avoidance, obstacle avoidance."""
        fg = self.goal_force(pos, goal)
        fo = self.obstacle_force(pos, obstacles)
        fc = self.collision_force(pos, agent_index, agents)
        return normalize(fg + fc + fo)

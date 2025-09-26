import os
import numpy as np
from adaptive_coverage.utils.utils import (
    nearest_points_on_obstacles,
    ray_intersects_aabb,
)
from adaptive_coverage.utils.utils import lambda2


class PenaltyNodeSolver:
    def __init__(
        self,
        index,
        pos,
        sensing_range,
        hexagon_range,
        avoidance_range,
        phi_0,
        v1,
        v2,
    ):
        self.index = index
        self.phi_0 = phi_0
        self.pos = pos
        self.sensing_range = sensing_range
        self.hexagon_range = hexagon_range
        self.avoidance_range = avoidance_range
        self.v1 = v1[0]
        self.v2 = v2[0]
        self.v1_idx = v1[1]
        self.v2_idx = v2[1]

    def solve(self):
        raise NotImplementedError()


class OriginalSolver(PenaltyNodeSolver):
    def __init__(self, *args, rho, **kwargs):
        super().__init__(*args, **kwargs)
        self.rho = rho

    def solve(self):
        phi1 = 2 * np.pi * self.v1_idx / 6
        phi2 = 2 * np.pi * self.v2_idx / 6
        phi = self.phi_0 + self.rho * (phi1 + phi2) / 2
        x = self.hexagon_range * np.cos(phi)
        y = self.hexagon_range * np.sin(phi)
        pos = self.pos + np.array([x, y])
        return pos


class PSOSolver(PenaltyNodeSolver):
    def __init__(
        self,
        *args,
        env,
        agents,
        pso_weights,
        dim=2,
        w=0.5,
        c1=1.0,
        c2=1.0,
        v_max=0.05,
        spread=0.05,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.dim = dim
        self.max_speed = v_max
        self.spread = spread

        self.agents = agents
        self.env = env
        self.agent = self.agents[self.index]

        self.positions = None
        self.velocities = None
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.pso_weights = pso_weights

        self.pbest = None
        self.pbest_val = None
        self.gbest = None
        self.gbest_val = None

        self.fitness_func_hist = []

    def initialize_particles(self, num_particles):
        self.num_particles = num_particles
        direction = self.v2 - self.v1
        direction /= np.linalg.norm(direction)
        perp_direction = np.array([-direction[1], direction[0]])

        # Random t values for interpolation along the segment
        t = np.random.uniform(0, 1, self.num_particles).reshape(-1, 1)

        # Linear interpolation along the segment
        base_positions = (1 - t) * self.v1 + t * \
            self.v2  # shape: (num_particles, 2)

        # Random perpendicular offsets
        offsets = np.random.uniform(
            -self.spread, self.spread, self.num_particles
        ).reshape(-1, 1)
        offset_vectors = offsets * perp_direction  # shape: (num_particles, 2)

        # Final particle positions
        self.positions = base_positions + offset_vectors
        self.velocities = np.zeros_like(self.positions)

        self.pbest = self.positions.copy()
        self.pbest_val = np.array([self.fitness_func(p)
                                  for p in self.positions])
        self.gbest = self.pbest[np.argmax(self.pbest_val)]
        self.gbest_val = np.max(self.pbest_val)  # minimize fitness function

    def calculate_coverage(self, position: np.ndarray):
        xmin = position[0] - self.sensing_range
        xmax = position[0] + self.sensing_range
        ymin = position[1] - self.sensing_range
        ymax = position[1] + self.sensing_range
        resolution = 100
        x_vals = np.linspace(xmin, xmax, resolution)
        y_vals = np.linspace(ymin, ymax, resolution)
        xx, yy = np.meshgrid(x_vals, y_vals)
        grid_points = np.column_stack((xx.ravel(), yy.ravel()))
        distances = np.linalg.norm(position - grid_points, axis=1)
        in_coverage = distances <= self.sensing_range

        if len(self.env.obstacles) > 0:
            x_in = (self.env.obstacles[:, 0] <= position[0]) & (
                position[0] <= self.env.obstacles[:, 0] +
                self.env.obstacles[:, 2]
            )
            y_in = (self.env.obstacles[:, 1] <= position[1]) & (
                position[1] <= self.env.obstacles[:, 1] +
                self.env.obstacles[:, 3]
            )
            in_obs = x_in & y_in
        else:
            in_obs = np.zeros(2)
        return 1 - (in_obs.sum() / in_coverage.sum())

    def connectivity_metric(self, position):
        has_free_agent = False
        free_agent_index = -1
        for agent in self.agents:
            if agent.is_unassigned():
                # agent.pos = position
                has_free_agent = True
                free_agent_index = agent.index
                break
        if has_free_agent:
            adj_mat = np.zeros((len(self.agents), len(self.agents)))
            agent_positions = []
            for agent in self.agents:
                if agent.index == free_agent_index:
                    agent_positions.append(position)
                else:
                    agent_positions.append(agent.pos.copy())
            for i in range(len(self.agents)):
                for j in range(i + 1, len(self.agents)):
                    if i == j:
                        continue
                    d = np.linalg.norm(agent_positions[i] - agent_positions[j])
                    if d < self.agent.sensing_range:
                        adj_mat[i][j] = adj_mat[j][i] = 1
            return lambda2(adj_mat)
        else:
            return 0

    '''
    def obstacle_avoidance(self, position: np.ndarray):
        if len(self.env.obstacles) > 0:
            obs_pts = nearest_points_on_obstacles(
                self.agent.pos, self.env.obstacles)
            dist_to_obs = np.linalg.norm(self.agent.pos - obs_pts)
            if dist_to_obs <= self.agent.sensing_range:
                diff = position - obs_pts
                return np.linalg.norm(diff)
            else:
                return 0
        else:
            return 0
    '''

    def obstacle_avoidance(self, position: np.ndarray):
        """
        Obstacle avoidance metric.

        Returns a value in [0, 1]:
            - 0   = worst (inside obstacle or blocked line of sight)
            - 1   = safe (far from obstacles and clear line of sight)
        """
        if len(self.env.obstacles) == 0:
            return 1.0  # no obstacles -> fully safe

        # --- Case 1: Candidate inside obstacle -> immediate penalty
        if self.env.point_is_in_obstacle(position, self.agent.size):
            return 0.0

        # --- Case 2: Distance to nearest obstacle surface
        obs_pts = nearest_points_on_obstacles(position, self.env.obstacles)
        dist_to_obs = np.linalg.norm(position - obs_pts)

        # Normalize: within avoidance_range -> scaled [0,1], outside -> safe
        if dist_to_obs < self.avoidance_range:
            dist_score = dist_to_obs / self.avoidance_range  # 0 near -> 1 at limit
        else:
            dist_score = 1.0

        # --- Case 3: Line-of-sight penalty (block between agent and candidate)
        if ray_intersects_aabb(self.agent.pos, position, self.env.obstacles):
            los_score = 0.0
        else:
            los_score = 1.0

        # --- Combine: weighted product (safe only if both distance & LOS are good)
        return dist_score * los_score

    def fitness_func(
        self,
        position: np.ndarray,
    ):
        """
        Fitness function of PSO algorithm.

        Args:
            position (numpy.ndarray): current particle's position.

        Returns:
            float: fitness value of current particle.
        """

        f_coverage_area = self.calculate_coverage(position)
        f_connectivity = self.connectivity_metric(position)
        f_avoidance = self.obstacle_avoidance(position)

        f = np.array([f_coverage_area, f_connectivity, f_avoidance])
        return self.pso_weights @ f.T

    def validate_positions(self):
        """Validate and correct all particle positions."""
        valid_positions = np.array(
            [self.is_valid_particle(p) for p in self.positions])
        in_obstacle = np.array(
            [self.env.point_is_in_obstacle(p) for p in self.positions]
        )
        in_range = np.array(
            [
                np.linalg.norm(p - self.agent.pos) <= self.sensing_range
                for p in self.positions
            ]
        )
        in_fov = np.array(
            [
                not ray_intersects_aabb(self.agent.pos, p, self.env.obstacles)
                for p in self.positions
            ]
        )

        # Identify invalid particles (either outside valid sector, in obstacle, or out of range)
        invalid_mask = (~valid_positions) | in_obstacle | (
            ~in_range) | (~in_fov)

        if np.any(invalid_mask):
            # Calculate mean position of valid particles for projection
            if np.any(~invalid_mask):
                # If we have some valid particles, use their mean
                mean_position = np.mean(self.positions[~invalid_mask], axis=0)
            else:
                # If all particles are invalid, use the midpoint between v1 and v2
                mean_position = (self.v1 + self.v2) / 2

            # Project invalid particles to mean position
            self.positions[invalid_mask] = mean_position

    def is_valid_particle(self, position):
        return True

        def cross_product(v1: np.ndarray, v2: np.ndarray):
            return v1[0] * v2[1] - v1[1] * v2[0]

        # Compute cross products
        v1 = self.v1 - self.agent_pos
        v2 = self.v2 - self.agent_pos
        # v1 = new_v1 - self.agent_pos
        # v2 = new_v2 - self.agent_pos
        cross1 = cross_product(v1, position)
        cross2 = cross_product(v2, position)
        cross12 = cross_product(v1, v2)

        # Check if P is inside the angle
        if cross12 > 0:
            return cross1 >= 0 and cross2 <= 0  # Counterclockwise
        else:
            return cross1 <= 0 and cross2 >= 0  # Clockwise

    def solve(self, num_particles, num_iterations):
        self.initialize_particles(num_particles)
        for i in range(num_iterations):
            r1 = np.random.rand(self.num_particles, self.dim)
            r2 = np.random.rand(self.num_particles, self.dim)

            cognitive = self.c1 * r1 * (self.pbest - self.positions)
            social = self.c2 * r2 * (self.gbest - self.positions)
            self.velocities = self.w * self.velocities + cognitive + social
            speeds = np.linalg.norm(self.velocities, axis=1, keepdims=True)
            too_fast = speeds > self.max_speed
            too_fast = too_fast.flatten()
            self.velocities[too_fast] = (
                self.velocities[too_fast] / speeds[too_fast]
            ) * self.max_speed
            self.positions += self.velocities
            # self.validate_positions()

            # Evaluate
            fitness = np.array([self.fitness_func(p) for p in self.positions])
            improved = fitness < self.pbest_val
            self.pbest[improved] = self.positions[improved]
            self.pbest_val[improved] = fitness[improved]

            if np.max(fitness) > self.gbest_val:
                self.gbest_val = np.max(fitness)
                self.gbest = self.positions[np.argmax(fitness)]

            self.fitness_func_hist.append(np.min(fitness))

        self.fitness_func_hist = np.array(self.fitness_func_hist)
        return self.gbest

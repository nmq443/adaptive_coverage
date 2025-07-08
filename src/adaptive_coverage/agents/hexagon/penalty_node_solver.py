import os
import numpy as np


class PenaltyNodeSolver:
    def __init__(self, index, pos, sensing_range, hexagon_range, avoidance_range, phi_0, v1, v2, result_manager):
        self.index = index
        self.phi_0 = phi_0
        self.pos = pos
        self.sensing_range = sensing_range
        self.hexagon_range = hexagon_range
        self.avoidance_range = avoidance_range
        self.v1 = v1
        self.v2 = v2
        self.result_manager = result_manager

    def solve(self):
        raise NotImplementedError()


class OriginalSolver(PenaltyNodeSolver):
    def __init__(self, *args, rho, **kwargs):
        super().__init__(*args, **kwargs)
        self.rho = rho

    def solve(self):
        phi1 = 2 * np.pi * self.v1[1] / 6
        phi2 = 2 * np.pi * self.v2[1] / 6
        phi = self.phi_0 + self.rho * (phi1 + phi2) / 2
        x = self.hexagon_range * np.cos(phi)
        y = self.hexagon_range * np.sin(phi)
        pos = self.pos + np.array([x, y])
        return pos


class PSOSolver(PenaltyNodeSolver):
    def __init__(
            self, *args, env, agents, pso_weights, res_dir, dim=2, w=0.5, c1=1.0, c2=1.0, num_particles=20, num_iterations=100, v_max=0.05, spread=0.05, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.res_dir = res_dir
        self.num_particles = num_particles
        self.dim = dim
        self.max_speed = v_max
        self.spread = spread
        self.num_iterations: int = num_iterations

        self.agents: list = agents
        self.env = env
        self.agent = self.agents[self.index]
        self.agent_pos: np.ndarray = self.agent.pos

        self.positions = None
        self.velocities = None
        self.w: float = w
        self.c1: float = c1
        self.c2: float = c2
        self.pso_weights: np.ndarray = pso_weights

        self.init_particles()
        self.initial_particles = self.positions.copy()
        self.pbest = self.positions.copy()
        self.pbest_val = np.array([self.fitness_func(p) for p in self.positions])
        self.gbest = self.pbest[np.argmax(self.pbest_val)]
        self.gbest_val = np.max(self.pbest_val)  # minimize fitness function

        self.fitness_func_hist: list = []

    def init_particles(self):
        direction = self.v2 - self.v1
        direction /= np.linalg.norm(direction)
        perp_direction = np.array([-direction[1], direction[0]])

        # Random t values for interpolation along the segment
        t = np.random.uniform(0, 1, self.num_particles).reshape(-1, 1)

        # Linear interpolation along the segment
        base_positions = (1 - t) * self.v1 + t * self.v2  # shape: (num_particles, 2)

        # Random perpendicular offsets
        offsets = np.random.uniform(
            -self.spread, self.spread, self.num_particles
        ).reshape(-1, 1)
        offset_vectors = offsets * perp_direction  # shape: (num_particles, 2)

        # Final particle positions
        self.positions = base_positions + offset_vectors
        self.velocities = np.zeros_like(self.positions)

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
                    position[0] <= self.env.obstacles[:, 0] + self.env.obstacles[:, 2]
            )
            y_in = (self.env.obstacles[:, 1] <= position[1]) & (
                    position[1] <= self.env.obstacles[:, 1] + self.env.obstacles[:, 3]
            )
            in_obs = x_in & y_in
        else:
            in_obs = np.zeros(2)
        return 1 - (in_obs.sum() / in_coverage.sum())

    def connectivity_metric(self, position):
        # Count connected boundary nodes
        connected_nodes = 0
        agent_positions = np.array(
            [agent.pos for agent in self.agents if agent.is_occupied()]
        )
        distances = np.linalg.norm(agent_positions - position, axis=1)
        connected_nodes = (distances <= self.sensing_range).sum()

        # Ensure connection to at least 2 boundary nodes for network integrity
        if connected_nodes >= 2:
            # Additional score for maintaining the minimum required connectivity
            base_score = 1.0
        else:
            # Penalty for insufficient connectivity
            base_score = 0.5 * connected_nodes / 2

        # Bonus for connecting to more boundary nodes (up to a point)
        # This promotes robust connectivity but doesn't overly reward excessive connections
        if connected_nodes > 2:
            additional_score = min(0.5, (connected_nodes - 2) * 0.1)
        else:
            additional_score = 0

        return base_score + additional_score

    def obstacle_avoidance(self, position: np.ndarray, min_clearance: float = 0.1):
        if len(self.env.obstacles) > 0:
            cx = self.env.obstacles[:, 0] + self.env.obstacles[:, 2] / 2
            cy = self.env.obstacles[:, 1] + self.env.obstacles[:, 3] / 2
            obs_pos = np.array([cx, cy]).T
            distances = np.linalg.norm(position - obs_pos, axis=1)
            min_distance = distances[np.argmin(distances)]
            # min_clearance += np.sqrt(2) * 0.5 * self.env.tile_size
            min_clearance += self.avoidance_range
            if min_distance <= min_clearance:
                penalty = np.exp(min_clearance - min_distance) - 1
            else:
                penalty = 0
            return penalty
        else:
            return 0

    def calculate_network_efficiency(self, position: np.ndarray):
        total_distance = 0
        count = 0
        agent_positions = np.array(
            [agent.pos for agent in self.agents if agent.is_occupied()]
        )
        distances = np.linalg.norm(position - agent_positions, axis=1)
        total_distance += np.sum(distances)
        count += len(distances)

        agent_positions = np.array(
            [
                agent.pos
                for agent in self.agents
                if (agent.is_occupied() and agent.is_penalty_node)
            ]
        )
        if len(agent_positions) > 0:
            distances = np.linalg.norm(position - agent_positions, axis=1)
            total_distance += np.sum(distances)
            count += len(distances)

        if count == 0:
            return 1.0

        avg_distance = total_distance / count
        efficiency = max(0, 1 - abs(avg_distance - self.sensing_range) / self.sensing_range)
        return efficiency

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
        f_network_efficiency = self.calculate_network_efficiency(position)

        f = np.array(
            [f_coverage_area, f_connectivity, f_avoidance, f_network_efficiency]
        )
        return self.pso_weights @ f.T

    def validate_positions(self):
        """Validate and correct all particle positions."""
        valid_positions = np.array([self.is_valid_particle(p) for p in self.positions])
        in_obstacle = np.array(
            [self.env.point_is_in_obstacle(p) for p in self.positions]
        )
        in_range = np.array(
            [
                np.linalg.norm(p - self.agent_pos) <= self.sensing_range
                for p in self.positions
            ]
        )

        # Identify invalid particles (either outside valid sector, in obstacle, or out of range)
        invalid_mask = (~valid_positions) | in_obstacle | (~in_range)

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

    def is_valid_particle(
            self,
            position: np.ndarray,
    ) -> bool:
        return True
        if abs(self.v1_idx - self.v2_idx) == 5:
            if self.v1_idx > self.v2_idx:
                self.v1_idx, self.v2_idx = self.v2_idx, self.v1_idx
        else:
            if self.v1_idx < self.v2_idx:
                v1_idx, v2_idx = self.v2_idx, self.v1_idx
            else:
                v1_idx, v2_idx = self.v1_idx, self.v2_idx
        phi_v1 = 2 * np.pi * v1_idx / 6
        phi_v2 = 2 * np.pi * v2_idx / 6
        v1x = self.agent_pos[0] + HEXAGON_RANGE * np.cos(
            phi_v1 + np.deg2rad(SWEEP_ANGLE_OFFSET)
        )
        v1y = self.agent_pos[1] + HEXAGON_RANGE * np.sin(
            phi_v1 + np.deg2rad(SWEEP_ANGLE_OFFSET)
        )
        v2x = self.agent_pos[0] + HEXAGON_RANGE * np.cos(
            phi_v2 - np.deg2rad(SWEEP_ANGLE_OFFSET)
        )
        v2y = self.agent_pos[1] + HEXAGON_RANGE * np.sin(
            phi_v2 - np.deg2rad(SWEEP_ANGLE_OFFSET)
        )
        new_v1 = np.array([v1x, v1y])
        new_v2 = np.array([v2x, v2y])

        def cross_product(v1: np.ndarray, v2: np.ndarray):
            return v1[0] * v2[1] - v1[1] * v2[0]

        # Compute cross products
        # v1 = self.v1 - self.agent_pos
        # v2 = self.v2 - self.agent_pos
        v1 = new_v1 - self.agent_pos
        v2 = new_v2 - self.agent_pos
        cross1 = cross_product(v1, position)
        cross2 = cross_product(v2, position)
        cross12 = cross_product(v1, v2)

        # Check if P is inside the angle
        if cross12 > 0:
            return cross1 >= 0 and cross2 <= 0  # Counterclockwise
        else:
            return cross1 <= 0 and cross2 >= 0  # Clockwise

    def solve(self):
        for i in range(self.num_iterations):
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

        self.save_figures()
        # return self.gbest, self.gbest_val
        return self.gbest

    def save_figures(self):
        # Save data
        fitness_func_hist = np.array(self.fitness_func_hist)
        data_filename = os.path.join(self.result_manager.res_dir, f"agent_no_{self.index}.npy")
        with open(data_filename, "wb") as f:
            np.save(f, fitness_func_hist)
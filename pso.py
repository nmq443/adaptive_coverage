import numpy as np
import os

from environment import Environment
from configs import *


class PSO:
    def __init__(
        self,
        index: int,
        v1: np.ndarray,
        v2: np.ndarray,
        env: Environment,
        agents: list,
        pso_weights: np.ndarray,
        dim: int = 2,
        w: float = 0.5,
        c1: float = 1.0,
        c2: float = 1.0,
    ):
        self.num_particles: int = PSO_PARTICLES
        self.dim = dim
        self.max_speed: float = PSO_VMAX
        self.spread: float = PSO_SPREAD
        self.num_iterations: int = PSO_ITERATIONS

        self.agents: list = agents
        self.env: Environment = env
        self.index: int = index
        self.agent = self.agents[self.index]
        self.agent_pos: np.ndarray = self.agent.pos

        self.positions = None
        self.velocities = None
        self.w: float = w
        self.c1: float = c1
        self.c2: float = c2
        self.v1: np.ndarray = v1[0]
        self.v2: np.ndarray = v2[0]
        self.v1_idx: int = v1[1]
        self.v2_idx: int = v2[1]
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
        xmin = position[0] - SENSING_RANGE
        xmax = position[0] + SENSING_RANGE
        ymin = position[1] - SENSING_RANGE
        ymax = position[1] + SENSING_RANGE
        resolution = 100
        x_vals = np.linspace(xmin, xmax, resolution)
        y_vals = np.linspace(ymin, ymax, resolution)
        xx, yy = np.meshgrid(x_vals, y_vals)
        grid_points = np.column_stack((xx.ravel(), yy.ravel()))
        distances = np.linalg.norm(position - grid_points, axis=1)
        in_coverage = distances <= SENSING_RANGE

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

    def connectivity_metric(self, position: np.ndarray):
        # Count connected boundary nodes
        connected_nodes = 0
        agent_positions = np.array(
            [agent.pos for agent in self.agents if agent.is_occupied()]
        )
        distances = np.linalg.norm(agent_positions - position, axis=1)
        connected_nodes = (distances <= SENSING_RANGE).sum()

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
            min_clearance += AVOIDANCE_RANGE
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
        efficiency = max(0, 1 - abs(avg_distance - SENSING_RANGE) / SENSING_RANGE)
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
                np.linalg.norm(p - self.agent_pos) <= SENSING_RANGE
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

    def optimize(self):
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
            self.validate_positions()

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
        return self.gbest, self.gbest_val

    def save_figures(self):
        # Save data
        fitness_func_hist = np.array(self.fitness_func_hist)
        data_dir = os.path.join(RES_DIR, METHOD_DIR, ENV_DIR, "pso")
        os.makedirs(data_dir, exist_ok=True)
        data_filename = os.path.join(data_dir, f"agent_no_{self.index}.npy")
        with open(data_filename, "wb") as f:
            np.save(f, fitness_func_hist)


def find_penalty_node(
    index: int, v1: np.ndarray, v2: np.ndarray, env: Environment, agents: list
) -> np.ndarray:
    """
    Find penalty node using PSO.

    Args:
        index (int): index of current agent.
        v1 (numpy.ndarray): first hidden vertex.
        v2 (numpy.ndarray): second hidden vertex.
        env (Environment): simulation environment.
        agents (list): list of all agents.

    Returns:
        numpy.ndarray: final solution.
    """
    pso_weights = np.array([0.45, 0.3, 0.15, 0.1])
    pso = PSO(
        index=index,
        v1=v1,
        v2=v2,
        env=env,
        agents=agents,
        pso_weights=pso_weights,
        w=0.75,
        c1=1.5,
        c2=1.5,
    )
    gbest, _ = pso.optimize()
    return gbest

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.sparse import csgraph
import networkx as nx
from typing import List, Dict, Set, Tuple, Optional
import time


class Robot:
    """
    Individual robot class implementing HDC strategy
    """

    def __init__(
        self,
        robot_id: int,
        position: np.ndarray,
        target: np.ndarray,
        r_c: float = 1.0,
        r_a: float = 0.2,
        epsilon: float = 0.2,
    ):
        self.id = robot_id
        self.position = position.copy()
        self.target = target.copy()
        self.velocity = np.zeros(2)

        # System parameters
        self.r_c = r_c  # Communication/sensing range
        self.r_a = r_a  # Obstacle avoidance range
        self.r_n = r_c - epsilon  # Noncritical area radius
        self.epsilon = epsilon  # Critical tolerance

        # Behavioral control parameters
        self.alpha = 0.75  # Cohesion weight
        self.beta = 1.0  # Separation weight
        self.gamma = 1.0  # Alignment weight
        self.zeta = 30  # Separation sharpness
        self.mu = 0.5  # Potential field balance

        # Network connectivity sets
        self.neighbors = set()
        self.critical_neighbors = set()
        self.noncritical_neighbors = set()

        # Topology information
        self.triangle_groups = []
        self.k_connected_groups = []
        self.redundant_critical = set()

        # Control state
        self.v_max = float("inf")
        self.dt = 0.01  # Time step

    def update_neighbors(self, other_robots: List["Robot"]):
        """Update neighbor sets based on distances"""
        self.neighbors.clear()
        self.critical_neighbors.clear()
        self.noncritical_neighbors.clear()

        for other in other_robots:
            if other.id == self.id:
                continue

            distance = np.linalg.norm(self.position - other.position)

            if distance <= self.r_c:
                self.neighbors.add(other.id)

                # Determine if critical or noncritical
                if self._is_critical_neighbor(other, other_robots):
                    self.critical_neighbors.add(other.id)
                else:
                    self.noncritical_neighbors.add(other.id)

    def _is_critical_neighbor(
        self, neighbor: "Robot", all_robots: List["Robot"]
    ) -> bool:
        """Determine if a neighbor is critical based on Definition 1"""
        distance = np.linalg.norm(self.position - neighbor.position)

        # Check if neighbor is in critical area
        if distance > self.r_n:
            # Check if there's no robot in the intersection of noncritical areas
            for other in all_robots:
                if other.id == self.id or other.id == neighbor.id:
                    continue

                dist_to_self = np.linalg.norm(self.position - other.position)
                dist_to_neighbor = np.linalg.norm(neighbor.position - other.position)

                if dist_to_self <= self.r_n and dist_to_neighbor <= neighbor.r_n:
                    return False  # Found robot in intersection, so not critical
            return True
        return False

    def behavioral_control(self, other_robots: List["Robot"]) -> np.ndarray:
        """Level 1: Behavioral control (cohesion, separation, alignment)"""
        v_cohesion = self._cohesion_velocity(other_robots)
        v_separation = self._separation_velocity(other_robots)
        v_alignment = self._alignment_velocity()

        return (
            self.alpha * v_cohesion
            + self.beta * v_separation
            + self.gamma * v_alignment
        )

    def _cohesion_velocity(self, other_robots: List["Robot"]) -> np.ndarray:
        """Cohesion velocity component"""
        v_cohesion = np.zeros(2)

        for other in other_robots:
            if (
                other.id in self.neighbors
                and other.id not in self._get_avoidance_neighbors(other_robots)
            ):
                r_ij = other.position - self.position
                v_cohesion += r_ij

        return v_cohesion

    def _separation_velocity(self, other_robots: List["Robot"]) -> np.ndarray:
        """Separation velocity component"""
        v_separation = np.zeros(2)

        for other in other_robots:
            if other.id in self._get_avoidance_neighbors(other_robots):
                r_ij = other.position - self.position
                distance = np.linalg.norm(r_ij)

                if distance > 0:
                    # Angle-dependent weight
                    heading = self.target - self.position
                    if np.linalg.norm(heading) > 0:
                        heading = heading / np.linalg.norm(heading)
                        r_hat = r_ij / distance
                        cos_phi = np.dot(heading, r_hat)
                        w_ij = self.mu + (1 - self.mu) * (1 + cos_phi) / 2
                    else:
                        w_ij = 1.0

                    # Exponential repulsion
                    repulsion = w_ij * np.exp(self.zeta * (distance - self.r_a))
                    v_separation -= repulsion * (r_ij / distance)

        return v_separation

    def _alignment_velocity(self) -> np.ndarray:
        """Alignment velocity towards target"""
        r_target = self.target - self.position
        distance = np.linalg.norm(r_target)

        if distance > 0:
            return r_target / distance
        return np.zeros(2)

    def _get_avoidance_neighbors(self, other_robots: List["Robot"]) -> set:
        """Get neighbors within avoidance range"""
        avoidance_neighbors = set()
        for other in other_robots:
            distance = np.linalg.norm(self.position - other.position)
            if distance <= self.r_a:
                avoidance_neighbors.add(other.id)
        return avoidance_neighbors

    def mobility_constraint(self, other_robots: List["Robot"]) -> float:
        """Level 2: Calculate mobility constraint for network integrity"""
        if not self.critical_neighbors:
            return self.epsilon / (2 * self.dt)

        epsilon_i = float("inf")
        for neighbor_id in self.critical_neighbors:
            neighbor = next(r for r in other_robots if r.id == neighbor_id)
            distance = np.linalg.norm(self.position - neighbor.position)
            epsilon_i = min(epsilon_i, self.r_c - distance)

        epsilon_i = max(epsilon_i - self.epsilon, 0)
        return epsilon_i / (2 * self.dt)

    def identify_local_topologies(
        self, other_robots: List["Robot"], communication_level: int = 1
    ):
        """Identify triangle and k-connected topologies"""
        self.triangle_groups.clear()
        self.k_connected_groups.clear()

        if len(self.critical_neighbors) < 2:
            return

        # Convert to list for ordering
        critical_list = list(self.critical_neighbors)

        # Check all pairs of critical neighbors
        for i in range(len(critical_list)):
            for j in range(i + 1, len(critical_list)):
                robot_j = next(r for r in other_robots if r.id == critical_list[i])
                robot_k = next(r for r in other_robots if r.id == critical_list[j])

                distance_jk = np.linalg.norm(robot_j.position - robot_k.position)

                if distance_jk <= self.r_c:
                    # Triangle topology
                    self.triangle_groups.append((critical_list[i], critical_list[j]))
                elif self._has_indirect_connection(
                    robot_j, robot_k, other_robots, communication_level
                ):
                    # K-connected topology
                    self.k_connected_groups.append((critical_list[i], critical_list[j]))

    def _has_indirect_connection(
        self,
        robot_j: "Robot",
        robot_k: "Robot",
        other_robots: List["Robot"],
        max_hops: int,
    ) -> bool:
        """Check if two robots have indirect connection within communication level"""
        # Simple BFS to check connectivity
        visited = set()
        queue = [(robot_j.id, 0)]
        visited.add(robot_j.id)

        while queue:
            current_id, hops = queue.pop(0)

            if current_id == robot_k.id:
                return True

            if hops >= max_hops:
                continue

            current_robot = next(r for r in other_robots if r.id == current_id)

            for neighbor_id in current_robot.neighbors:
                if neighbor_id not in visited and neighbor_id != self.id:
                    visited.add(neighbor_id)
                    queue.append((neighbor_id, hops + 1))

        return False

    def minimize_topologies(self, other_robots: List["Robot"]):
        """Level 3 & 4: Minimize redundant critical connections"""
        self.redundant_critical.clear()

        # Level 3: Minimize triangle topologies
        redundant_triangle = self._minimize_triangle_topologies(other_robots)
        self.redundant_critical.update(redundant_triangle)

        # Level 4: Minimize k-connected topologies if needed
        remaining_critical = self.critical_neighbors - self.redundant_critical
        if len(remaining_critical) >= 2:
            redundant_k = self._minimize_k_connected_topologies(other_robots)
            self.redundant_critical.update(redundant_k)

    def _minimize_triangle_topologies(self, other_robots: List["Robot"]) -> set:
        """Minimize triangle topologies using connectivity removal rule"""
        redundant = set()

        for group in self.triangle_groups:
            # Find the robot that allows best movement toward target
            best_robot = self._find_best_connectivity(group, other_robots)

            for robot_id in group:
                if robot_id != best_robot:
                    # Consensus check (simplified - assume agreement)
                    if self._consensus_check(robot_id, other_robots):
                        redundant.add(robot_id)

        return redundant

    def _minimize_k_connected_topologies(self, other_robots: List["Robot"]) -> set:
        """Minimize k-connected topologies"""
        redundant = set()

        for group in self.k_connected_groups:
            best_robot = self._find_best_connectivity(group, other_robots)

            for robot_id in group:
                if robot_id != best_robot:
                    if self._consensus_check(robot_id, other_robots):
                        redundant.add(robot_id)

        return redundant

    def _find_best_connectivity(self, group: tuple, other_robots: List["Robot"]) -> int:
        """Find the best robot to maintain connection with based on Eq. 25"""
        target_direction = self.target - self.position
        if np.linalg.norm(target_direction) == 0:
            return group[0]  # Default choice

        target_direction = target_direction / np.linalg.norm(target_direction)

        best_robot = group[0]
        min_angle = float("inf")

        for robot_id in group:
            robot = next(r for r in other_robots if r.id == robot_id)
            robot_direction = robot.position - self.position

            if np.linalg.norm(robot_direction) > 0:
                robot_direction = robot_direction / np.linalg.norm(robot_direction)
                angle = np.arccos(
                    np.clip(np.dot(target_direction, robot_direction), -1, 1)
                )

                if angle < min_angle:
                    min_angle = angle
                    best_robot = robot_id

        return best_robot

    def _consensus_check(self, robot_id: int, other_robots: List["Robot"]) -> bool:
        """Simplified consensus check - in real implementation, this would involve communication"""
        return True  # Assume consensus for simplification

    def update_velocity_constraint(self, other_robots: List["Robot"]):
        """Update maximum velocity based on mobility constraint"""
        self.v_max = self.mobility_constraint(other_robots)

    def compute_control_input(self, other_robots: List["Robot"]) -> np.ndarray:
        """Compute the final control input using HDC"""
        # Level 1: Behavioral control
        desired_velocity = self.behavioral_control(other_robots)

        # Level 2: Apply mobility constraint
        self.update_velocity_constraint(other_robots)

        # Level 3 & 4: Minimize topologies
        self.identify_local_topologies(other_robots)
        self.minimize_topologies(other_robots)

        # Apply velocity constraint
        speed = np.linalg.norm(desired_velocity)
        if speed > self.v_max:
            desired_velocity = (self.v_max / speed) * desired_velocity

        return desired_velocity

    def update_position(self, control_input: np.ndarray):
        """Update robot position based on control input"""
        self.velocity = control_input
        self.position += self.velocity * self.dt

    def distance_to_target(self) -> float:
        """Calculate distance to target"""
        return np.linalg.norm(self.position - self.target)


class MultiRobotSystem:
    """
    Multi-robot system implementing HDC strategy
    """

    def __init__(self, num_robots: int, area_size: Tuple[float, float] = (10, 10)):
        self.robots = []
        self.area_size = area_size
        self.time = 0
        self.dt = 0.01

        # Performance metrics
        self.connectivity_history = []
        self.success_rate_history = []

    def initialize_random_scenario(self, num_robots: int, seed: Optional[int] = None):
        """Initialize robots and targets randomly"""
        if seed is not None:
            np.random.seed(seed)

        self.robots.clear()

        # Generate robot positions
        positions = np.random.uniform(0, min(self.area_size), (num_robots, 2))

        # Generate targets ensuring connectivity
        targets = self._generate_connected_targets(num_robots)

        # Create robots
        for i in range(num_robots):
            robot = Robot(i, positions[i], targets[i])
            self.robots.append(robot)

    def _generate_connected_targets(self, num_robots: int) -> np.ndarray:
        """Generate targets that form a connected graph"""
        targets = []
        r_c = 1.0  # Communication range

        # Place first target randomly
        first_target = np.random.uniform(1, np.array(self.area_size) - 1)
        targets.append(first_target)

        # Place remaining targets to ensure connectivity
        for i in range(1, num_robots):
            placed = False
            attempts = 0

            while not placed and attempts < 100:
                # Generate candidate position
                candidate = np.random.uniform(1, np.array(self.area_size) - 1)

                # Check if it's connected to existing targets
                for existing_target in targets:
                    if np.linalg.norm(candidate - existing_target) <= r_c:
                        targets.append(candidate)
                        placed = True
                        break

                attempts += 1

            if not placed:
                # Place near a random existing target
                base_target = targets[np.random.randint(len(targets))]
                angle = np.random.uniform(0, 2 * np.pi)
                distance = np.random.uniform(0.3, 0.8) * r_c
                candidate = base_target + distance * np.array(
                    [np.cos(angle), np.sin(angle)]
                )

                # Ensure it's within bounds
                candidate = np.clip(candidate, 1, np.array(self.area_size) - 1)
                targets.append(candidate)

        return np.array(targets)

    def update_all_neighbors(self):
        """Update neighbor information for all robots"""
        for robot in self.robots:
            robot.update_neighbors(self.robots)

    def compute_network_connectivity(self) -> float:
        """Compute the second smallest eigenvalue (algebraic connectivity)"""
        n = len(self.robots)
        if n < 2:
            return 0

        # Build adjacency matrix
        adjacency = np.zeros((n, n))

        for i, robot_i in enumerate(self.robots):
            for j, robot_j in enumerate(self.robots):
                if i != j:
                    distance = np.linalg.norm(robot_i.position - robot_j.position)
                    if distance <= robot_i.r_c:
                        adjacency[i, j] = 1

        # Compute Laplacian matrix
        degree = np.sum(adjacency, axis=1)
        laplacian = np.diag(degree) - adjacency

        # Compute eigenvalues
        eigenvalues = np.linalg.eigvals(laplacian)
        eigenvalues = np.real(eigenvalues)  # Remove tiny imaginary parts
        eigenvalues.sort()

        # Return second smallest eigenvalue
        return eigenvalues[1] if n > 1 else 0

    def compute_success_rate(self, threshold: float = 0.2) -> float:
        """Compute the rate of robots that reached their targets"""
        successful = 0
        for robot in self.robots:
            if robot.distance_to_target() <= threshold:
                successful += 1
        return successful / len(self.robots) if self.robots else 0

    def step(self):
        """Perform one simulation step"""
        self.update_all_neighbors()

        # Compute control inputs for all robots
        control_inputs = []
        for robot in self.robots:
            control_input = robot.compute_control_input(self.robots)
            control_inputs.append(control_input)

        # Update positions
        for robot, control_input in zip(self.robots, control_inputs):
            robot.update_position(control_input)

        # Update time and metrics
        self.time += self.dt
        connectivity = self.compute_network_connectivity()
        success_rate = self.compute_success_rate()

        self.connectivity_history.append(connectivity)
        self.success_rate_history.append(success_rate)

    def run_simulation(
        self, max_steps: int = 5000, target_success_rate: float = 0.95
    ) -> Dict:
        """Run the complete simulation"""
        print(f"Starting simulation with {len(self.robots)} robots...")

        start_time = time.time()

        for step in range(max_steps):
            self.step()

            # Check termination conditions
            if step % 100 == 0:
                success_rate = self.compute_success_rate()
                connectivity = self.compute_network_connectivity()

                print(
                    f"Step {step}: Success rate = {success_rate:.2f}, "
                    f"Connectivity = {connectivity:.4f}"
                )

                if success_rate >= target_success_rate:
                    print(f"Target success rate achieved at step {step}")
                    break

        end_time = time.time()

        final_success_rate = self.compute_success_rate()
        final_connectivity = self.compute_network_connectivity()

        results = {
            "final_success_rate": final_success_rate,
            "final_connectivity": final_connectivity,
            "steps": step + 1,
            "simulation_time": end_time - start_time,
            "connectivity_history": self.connectivity_history.copy(),
            "success_rate_history": self.success_rate_history.copy(),
        }

        print(f"\nSimulation completed:")
        print(f"Final success rate: {final_success_rate:.2f}")
        print(f"Final connectivity: {final_connectivity:.4f}")
        print(f"Total steps: {step + 1}")
        print(f"Simulation time: {end_time - start_time:.2f} seconds")

        return results

    def visualize_system(self, figsize: Tuple[int, int] = (12, 10)):
        """Visualize the current state of the multi-robot system"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)

        # Plot 1: Robot positions and targets
        ax1.set_xlim(0, self.area_size[0])
        ax1.set_ylim(0, self.area_size[1])
        ax1.set_aspect("equal")
        ax1.set_title("Robot Positions and Network")

        # Draw connections
        for robot in self.robots:
            for neighbor_id in robot.neighbors:
                neighbor = next(r for r in self.robots if r.id == neighbor_id)

                # Color code connections
                if (
                    neighbor_id in robot.critical_neighbors
                    and neighbor_id not in robot.redundant_critical
                ):
                    color = "red"  # Critical connection
                    linewidth = 2
                elif neighbor_id in robot.critical_neighbors:
                    color = "orange"  # Redundant critical connection
                    linewidth = 1
                else:
                    color = "blue"  # Non-critical connection
                    linewidth = 0.5

                ax1.plot(
                    [robot.position[0], neighbor.position[0]],
                    [robot.position[1], neighbor.position[1]],
                    color=color,
                    linewidth=linewidth,
                    alpha=0.7,
                )

        # Draw robots and targets
        for i, robot in enumerate(self.robots):
            # Robot position
            ax1.plot(robot.position[0], robot.position[1], "ko", markersize=8)
            ax1.text(
                robot.position[0],
                robot.position[1],
                f"{i}",
                ha="center",
                va="center",
                color="white",
                fontsize=8,
            )

            # Target position
            ax1.plot(robot.target[0], robot.target[1], "r*", markersize=12)

            # Sensing range (only for first few robots to avoid clutter)
            if i < 3:
                circle = Circle(
                    robot.position, robot.r_c, fill=False, linestyle="--", alpha=0.3
                )
                ax1.add_patch(circle)

        # Plot 2: Connectivity over time
        if self.connectivity_history:
            ax2.plot(self.connectivity_history)
            ax2.set_title("Network Connectivity Over Time")
            ax2.set_xlabel("Step")
            ax2.set_ylabel("Algebraic Connectivity")
            ax2.grid(True)

        # Plot 3: Success rate over time
        if self.success_rate_history:
            ax3.plot(self.success_rate_history)
            ax3.set_title("Success Rate Over Time")
            ax3.set_xlabel("Step")
            ax3.set_ylabel("Success Rate")
            ax3.grid(True)

        # Plot 4: Distance to targets
        distances = [robot.distance_to_target() for robot in self.robots]
        ax4.bar(range(len(distances)), distances)
        ax4.set_title("Distance to Targets")
        ax4.set_xlabel("Robot ID")
        ax4.set_ylabel("Distance")
        ax4.grid(True)

        plt.tight_layout()
        plt.show()


def run_hdc_experiment():
    """Run a complete HDC experiment"""

    # Initialize with random scenario
    num_robots = 8
    # Create multi-robot system
    mrs = MultiRobotSystem(num_robots=num_robots, area_size=(10, 10))
    mrs.initialize_random_scenario(num_robots, seed=42)

    print(f"Initialized {num_robots} robots")
    print("Initial connectivity:", mrs.compute_network_connectivity())

    # Run simulation
    results = mrs.run_simulation(max_steps=3000)

    # Visualize results
    mrs.visualize_system()

    return mrs, results


def compare_control_levels():
    """Compare performance of different HDC control levels"""
    num_robots = 6
    scenarios = 5
    results_comparison = {}

    for scenario in range(scenarios):
        print(f"\n=== Scenario {scenario + 1} ===")

        # Initialize same scenario for all control levels
        mrs = MultiRobotSystem(area_size=(8, 8))
        mrs.initialize_random_scenario(num_robots, seed=scenario)

        # Test different control configurations
        # This is a simplified comparison - in full implementation,
        # you would modify the control levels in the Robot class

        results = mrs.run_simulation(max_steps=2000)
        results_comparison[f"scenario_{scenario}"] = results

    return results_comparison


if __name__ == "__main__":
    # Run the experiment
    mrs, results = run_hdc_experiment()

    print("\nExperiment completed successfully!")
    print(f"Network remained connected: {min(results['connectivity_history']) > 0}")

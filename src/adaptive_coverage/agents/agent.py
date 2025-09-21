import numpy as np


class Agent:
    def __init__(
        self,
        index,
        init_pos,
        size,
        path_planner,
        sensing_range,
        timestep,
        v_max=0.05,
        avoidance_range=0.05,
        tolerance=0.05,
    ):
        # Basic parameters
        self.index = index
        self.pos = init_pos.copy()
        self.theta = 0
        self.vel = np.zeros(2)
        self.size = size
        self.v_max = v_max
        self.sensing_range = sensing_range
        self.timestep = timestep

        # Path planning parameters
        self.path_planner = path_planner
        self.avoidance_range = avoidance_range + size * 2
        self.tolerance = tolerance

        # Visualization parameters
        self.goal = None

    def get_travel_distance(self, state):
        """Get total travel distance."""
        traj = state[self.index, :, :-1]
        if len(traj) < 2:
            return 0.0
        displacements = traj[1:] - traj[:-1]
        distances = np.linalg.norm(displacements, axis=1)
        return np.sum(distances)

    def move_to_goal(self, goal, agents, obstacles, timestep, desired_v=None):
        self.goal = goal
        if self.terminated(goal):
            self.stop()
        else:
            self.vel = self.path_planner.total_force(
                self.pos, self.goal, self.index, agents, obstacles
            )
            self.limit_speed(desired_v=desired_v)
            self.pos += self.vel * timestep

    def terminated(self, goal):
        return np.linalg.norm(self.pos - goal) <= self.tolerance

    def stop(self):
        self.vel = np.zeros(2)

    def limit_speed(self, desired_v=None):
        v = np.linalg.norm(self.vel)
        s = self.v_max if desired_v is None else desired_v
        if v >= self.v_max:
            self.vel = self.vel / v * s

    def step(self, timestep, *args, **kwargs):
        self.theta = np.arctan2(self.vel[1], self.vel[0])

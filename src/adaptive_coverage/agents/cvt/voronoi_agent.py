from adaptive_coverage.agents.agent import Agent
from adaptive_coverage.agents.cvt.lloyd import lloyd


class VoronoiAgent(Agent):
    def __init__(self, index, init_pos, size, path_planner, sensing_range):
        super().__init__(index, init_pos, size, path_planner, sensing_range)

    def step(self, agents, env):
        if self.goal is not None and not self.terminated(self.goal):
            self.move_to_goal(self.goal, agents, env.obstacles)
        else:
            self.goal = lloyd(self, agents, env)

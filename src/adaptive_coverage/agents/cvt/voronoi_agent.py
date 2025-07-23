from adaptive_coverage.agents.agent import Agent
from adaptive_coverage.agents.cvt.lloyd import lloyd


class VoronoiAgent(Agent):
    def __init__(self, *args, valid_ratio=0.9, **kwargs):
        super().__init__(*args, **kwargs)
        self.valid_range = self.sensing_range * valid_ratio

    def step(self, agents, env):
        super().step()
        if self.goal is not None and not self.terminated(self.goal):
            self.move_to_goal(self.goal, agents, env.obstacles)
        else:
            self.goal = lloyd(self, agents, env)

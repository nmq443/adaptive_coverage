import numpy as np
from adaptive_coverage.agents.agent import Agent
from adaptive_coverage.agents.cvt.lloyd import lloyd


class VoronoiAgent(Agent):
    def __init__(self, *args, valid_ratio=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.valid_range = self.sensing_range * valid_ratio

    def step(self, agents, env, timestep):
        super().step()
        if self.goal is not None and not self.terminated(self.goal):
            agents_positions = np.array(
                [agent.pos for agent in agents if agent.index != self.index]
            )
            distances = np.linalg.norm(agents_positions - self.pos, axis=1)
            epsi = np.min(self.sensing_range - distances)
            desired_v = self.v_max * (epsi / (2 * timestep))
            # self.move_to_goal(
            #     self.goal, agents, env.obstacles, timestep, desired_v=desired_v
            # )
            self.move_to_goal(self.goal, agents, env.obstacles, timestep)
        else:
            self.goal = lloyd(self, agents, env)

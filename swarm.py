import os
import numpy as np
import matplotlib.pyplot as plt
from configs import *
from voronoi import *
from collections import deque

if CONTROLLER == 'voronoi':
    from voronoi_agent import Agent

    class Swarm:
        def __init__(self):
            self.num_agents = NUM_AGENTS
            self.agents = []
            self.generators = []

        def init_agents(self, ref_pos=None):
            if RANDOM_INIT and ref_pos is not None:
                random_positions = np.random.rand(self.num_agents, 2) * AGENT_SPREAD + ref_pos
                for index in range(self.num_agents):
                    self.agents.append(Agent(
                        index=index,
                        init_pos=random_positions[index],
                    ))
            else:
                for i in range(NUM_ROWS):
                    for j in range(NUM_COLS):
                        self.agents.append(Agent(i * NUM_COLS + j, INIT_POS[i][j]))
            self.generators = [agent.pos for agent in self.agents]
            self.generators = np.array(self.generators)

        def step(self, env):
            if len(self.agents) > 0:
                for agent in self.agents:
                    agent.step(self.agents, env)

        def render(self, surface, env, font):
            if len(self.agents) > 0:
                for agent in self.agents:
                    agent.render(surface, font)
                    self.generators[agent.index] = agent.pos
                vor = compute_voronoi_diagrams(self.generators, env)
                draw_voronoi(vor, surface)

        def create_plot(self):
            fig, ax = plt.subplots()
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Error")
            for agent in self.agents:
                ax.plot(np.arange(len(agent.errs)), agent.errs, label=f"Agent {agent.index}")
            ax.legend()
            img_path = os.path.join(RES_DIR, "errors.png")
            plt.savefig(img_path)

else:
    from hexagon_agent import Agent

    class Swarm:
        def __init__(self):
            self.num_agents = NUM_AGENTS
            self.agents = []
            self.landmarks = deque([])

        def init_agents(self, ref_pos=None):
            if RANDOM_INIT and ref_pos is not None:
                random_positions = np.random.uniform(-1, 1, (self.num_agents, 2)) * AGENT_SPREAD + ref_pos
                for index in range(self.num_agents):
                    self.agents.append(Agent(
                        index=index,
                        init_pos=random_positions[index],
                    ))
                self.determine_root(0, self.agents[0].pos)
            else:
                for i in range(NUM_ROWS):
                    for j in range(NUM_COLS):
                        self.agents.append(Agent(i * NUM_ROWS + j, INIT_POS[i][j]))
                self.determine_root(0, self.agents[0].pos)
        
        def determine_root(self, agent_id, agent_goal):
            self.agents[agent_id].set_state("occupied")
            # self.agents[agent_id].set_goal(agent_goal)

        def step(self, env):
            if len(self.agents) > 0:
                order = np.random.permutation(len(self.agents))
                for i in order:
                    self.agents[i].step(self.landmarks, self.agents, env)

        def render(self, surface, font):
            if len(self.agents) > 0:
                for agent in self.agents:
                    agent.render(surface, font, self.agents)

        def create_plot(self):
            fig, ax = plt.subplots()
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Error")
            for agent in self.agents:
                ax.plot(np.arange(len(agent.errs)), agent.errs, label=f"Agent {agent.index}")
            ax.legend()
            img_path = os.path.join(RES_DIR, METHOD_DIR, "errors.png")
            plt.savefig(img_path)

import os
import numpy as np
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
            self.graph = None

        def init_agents(self, ref_pos=None):
            if RANDOM_INIT and ref_pos is not None:
                random_positions = np.random.rand(
                    self.num_agents, 2) * AGENT_SPREAD + ref_pos
                for index in range(self.num_agents):
                    self.agents.append(Agent(
                        index=index,
                        init_pos=random_positions[index],
                    ))
            else:
                for i in range(NUM_ROWS):
                    for j in range(NUM_COLS):
                        self.agents.append(
                            Agent(i * NUM_COLS + j, INIT_POS[i][j]))
            self.generators = [agent.pos for agent in self.agents]
            self.generators = np.array(self.generators)

        def step(self, env):
            if len(self.agents) > 0:
                for agent in self.agents:
                    agent.step(self.agents, env)

        def render(self, surface, env, font, timestep):
            if len(self.agents) > 0:
                for agent in self.agents:
                    agent.render(surface, font, self.agents, timestep)
                    self.generators[agent.index] = agent.pos
                vor = compute_voronoi_diagrams(self.generators, env)
                draw_voronoi(vor, surface)

        def save_data(self, res_dir):
            datas = []

            # save poses
            for agent in self.agents:
                data = []
                data.append(agent.trajectory[0])  # first pose
                data.append(agent.pos)  # last pose
                datas.append(data)

            # save graph
            graph = np.zeros((NUM_AGENTS, NUM_AGENTS))
            for i in range(NUM_AGENTS):
                for j in range(NUM_AGENTS):
                    if i == j:
                        graph[i][j] = 0
                    else:
                        distance = np.linalg.norm(
                            self.agents[i].pos - self.agents[j].pos)
                        if distance <= SENSING_RANGE:
                            graph[i][j] = 1
                            graph[j][i] = 1

            self.graph = graph

            datas = np.array(datas)

            save_file = os.path.join(
                res_dir, "swarm_data.npy")

            with open(save_file, 'wb') as f:
                np.save(f, datas)

else:
    from hexagon_agent import Agent

    class Swarm:
        def __init__(self):
            self.num_agents = NUM_AGENTS
            self.agents = []
            self.landmarks = deque([])
            self.graph = None

        def init_agents(self, ref_pos=None):
            if RANDOM_INIT and ref_pos is not None:
                random_positions = np.random.uniform(
                    -1, 1, (self.num_agents, 2)) * AGENT_SPREAD + ref_pos
                for index in range(self.num_agents):
                    self.agents.append(Agent(
                        index=index,
                        init_pos=random_positions[index],
                    ))
                self.determine_root(0, self.agents[0].pos)
            else:
                for i in range(NUM_ROWS):
                    for j in range(NUM_COLS):
                        self.agents.append(
                            Agent(i * NUM_COLS + j, INIT_POS[i][j]))
                # for agent in self.agents:
                #     print(agent.index)
                self.determine_root(0, self.agents[0].pos)

        def determine_root(self, agent_id, agent_goal):
            self.agents[agent_id].set_state("occupied")
            # self.agents[agent_id].set_goal(agent_goal)

        def step(self, env):
            if len(self.agents) > 0:
                order = np.random.permutation(len(self.agents))
                for i in order:
                    self.agents[i].step(self.landmarks, self.agents, env)

        def render(self, surface, font, timestep):
            if len(self.agents) > 0:
                for agent in self.agents:
                    agent.render(surface, font, self.agents, timestep)

        def save_data(self, res_dir):
            datas = []

            # save poses
            for agent in self.agents:
                data = []
                data.append(agent.trajectory[0])  # first pose
                data.append(agent.pos)  # last pose
                datas.append(data)

            # save graph
            graph = np.zeros((NUM_AGENTS, NUM_AGENTS))
            for i in range(NUM_AGENTS):
                for j in range(NUM_AGENTS):
                    if i == j:
                        graph[i][j] = 0
                    else:
                        distance = np.linalg.norm(
                            self.agents[i].pos - self.agents[j].pos)
                        if distance <= SENSING_RANGE:
                            graph[i][j] = 1
                            graph[j][i] = 1

            self.graph = graph

            datas = np.array(datas)

            save_file = os.path.join(
                res_dir, "swarm_data.npy")

            with open(save_file, 'wb') as f:
                np.save(f, datas)

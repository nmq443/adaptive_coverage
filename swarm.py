import os
import numpy as np
from evaluate import lamda2
from configs import *
from collections import deque
from lloyd import compute_voronoi_diagrams
from utils import draw_voronoi

if CONTROLLER == "voronoi":
    from voronoi_agent import Agent

    class Swarm:
        def __init__(self):
            self.num_agents: int = NUM_AGENTS
            self.agents: list = []
            self.generators: list = []
            self.ld2s: list = []

        def init_agents(self, ref_pos=None):
            if RANDOM_INIT and ref_pos is not None:
                random_positions = (
                    np.random.rand(self.num_agents, 2) * AGENT_SPREAD + ref_pos
                )
                for index in range(self.num_agents):
                    self.agents.append(
                        Agent(
                            index=index,
                            init_pos=random_positions[index],
                        )
                    )
            else:
                for i in range(NUM_ROWS):
                    for j in range(NUM_COLS):
                        self.agents.append(Agent(i * NUM_COLS + j, INIT_POS[i][j]))
            self.generators = [agent.pos for agent in self.agents]
            self.generators = np.array(self.generators)

        def step(self, env):
            if len(self.agents) > 0:
                order = np.random.permutation(len(self.agents))
                for i in order:
                    self.agents[i].step(self.agents, env)
                ld2 = lamda2(self.agents)
                self.ld2s.append(ld2)

        def render(self, surface, env, font, timestep):
            if len(self.agents) > 0:
                for agent in self.agents:
                    agent.render(surface, font, self.agents, timestep)
                    self.generators[agent.index] = agent.pos
                vor = compute_voronoi_diagrams(self.generators, env)
                draw_voronoi(vor, surface)

        def get_travel_distance(self):
            """Save travel distance of all agents."""
            distances = []
            for agent in self.agents:
                distance = agent.get_travel_distance()
                distances.append(distance)
            return np.array(distances)

        def save_data(self, res_dir: str):
            # save poses
            datas = []
            for agent in self.agents:
                data = []
                data.append(agent.traj[0])  # first pose
                data.append(agent.pos)  # last pose
                datas.append(data)
            datas = np.array(datas)
            save_file = os.path.join(res_dir, "swarm_data.npy")
            with open(save_file, "wb") as f:
                np.save(f, datas)

            # save travel distances
            distances = self.get_travel_distance()
            save_file = os.path.join(res_dir, "travel_distances.npy")
            with open(save_file, "wb") as f:
                np.save(f, distances)

            # save ld2s
            ld2s = np.array(self.ld2s)
            save_file = os.path.join(res_dir, "ld2s.npy")
            with open(save_file, "wb") as f:
                np.save(f, ld2s)

else:
    from hexagon_agent import Agent

    class Swarm:
        def __init__(self):
            self.num_agents: int = NUM_AGENTS
            self.agents: list = []
            self.landmarks: deque = deque([])
            self.ld2s: list = []

        def init_agents(self, ref_pos=None):
            if RANDOM_INIT and ref_pos is not None:
                random_positions = (
                    np.random.uniform(-1, 1, (self.num_agents, 2)) * AGENT_SPREAD
                    + ref_pos
                )
                for index in range(self.num_agents):
                    self.agents.append(
                        Agent(
                            index=index,
                            init_pos=random_positions[index],
                        )
                    )
                self.determine_root(0, self.agents[0].pos)
            else:
                for i in range(NUM_ROWS):
                    for j in range(NUM_COLS):
                        self.agents.append(Agent(i * NUM_COLS + j, INIT_POS[i][j]))
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
                ld2 = lamda2(self.agents)
                self.ld2s.append(ld2)

        def render(self, surface, font, timestep):
            if len(self.agents) > 0:
                for agent in self.agents:
                    agent.render(surface, font, self.agents, timestep)

        def get_travel_distance(self):
            """Save travel distance of all agents."""
            distances = []
            for agent in self.agents:
                distance = agent.get_travel_distance()
                distances.append(distance)
            return np.array(distances)

        def save_data(self, res_dir: str):
            # save poses
            datas = []
            for agent in self.agents:
                data = []
                data.append(agent.traj[0])  # first pose
                data.append(agent.pos)  # last pose
                datas.append(data)
            datas = np.array(datas)
            save_file = os.path.join(res_dir, "swarm_data.npy")
            with open(save_file, "wb") as f:
                np.save(f, datas)

            # save travel distances
            distances = self.get_travel_distance()
            save_file = os.path.join(res_dir, "travel_distances.npy")
            with open(save_file, "wb") as f:
                np.save(f, distances)

            # save ld2s
            ld2s = np.array(self.ld2s)
            save_file = os.path.join(res_dir, "ld2s.npy")
            with open(save_file, "wb") as f:
                np.save(f, ld2s)

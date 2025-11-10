import numpy as np
from adaptive_coverage.swarms.swarm import Swarm
from adaptive_coverage.agents.cvt.voronoi_agent import VoronoiAgent
from adaptive_coverage.utils.utils import lambda2
from typing import Union, Optional


class VoronoiSwarm(Swarm):
    def __init__(self, *args, critical_ratio, **kwargs):
        super().__init__(*args, **kwargs)
        self.critical_ratio = critical_ratio
        self.generators: Union[list, np.ndarray] = []
        # at each step, we save current agent's critical agents list
        # encode like this: if agent j is critical agent of agent i
        # then agent i's critical_agents[j] = 1
        # else 0
        self.critical_agents: np.ndarray = np.zeros(
            (self.state.shape[0], self.state.shape[1], self.num_agents))
        self.critical_agents_before_removing_redundant: np.ndarray = np.zeros(
            (self.state.shape[0], self.state.shape[1], self.num_agents))

    def init_agents(self):
        """Initialize all agents in a grid-like formation or in random positions."""
        if self.random_init:
            pos = np.random.rand(self.num_agents, 2) + np.array([1, 1])
            for i, p in enumerate(pos):
                self.agents.append(
                    VoronoiAgent(
                        index=i * self.num_cols + j,
                        init_pos=init_pos,
                        size=self.agent_size,
                        critical_ratio=self.critical_ratio,
                        sensing_range=self.sensing_range,
                        timestep=self.timestep,
                        path_planner=self.path_planner,
                    )
                )
        else:
            for i in range(self.num_rows):
                for j in range(self.num_cols):
                    x = self.first_agent_pos[0] + j * self.dist_btw_agents
                    y = (
                        self.first_agent_pos[1]
                        + (self.num_rows - i - 1) * self.dist_btw_agents
                    )
                    init_pos = np.array([x, y])
                    self.agents.append(
                        VoronoiAgent(
                            index=i * self.num_cols + j,
                            init_pos=init_pos,
                            size=self.agent_size,
                            critical_ratio=self.critical_ratio,
                            sensing_range=self.sensing_range,
                            timestep=self.timestep,
                            path_planner=self.path_planner,
                        )
                    )
        self.generators = np.array([agent.pos for agent in self.agents])

    def update_critical_agents_state(self, agent_index, current_step, state, before=False):
        if before:
            self.critical_agents_before_removing_redundant[agent_index,
                                                           current_step] = state
        else:
            self.critical_agents[agent_index, current_step] = state

    def step(self, env, current_step):
        if len(self.agents) > 0:
            for agent in self.agents:
                # record agent's critical agents before removing redundant
                critical_agents = agent.get_critical_agents(self.agents, env)
                state = np.zeros(self.num_agents)
                if len(critical_agents) >= 1:
                    state[critical_agents] = 1
                self.update_critical_agents_state(
                    agent.index, current_step, state, False)

                # perform a step
                agent.step(self.agents, env)
                pos = agent.get_pos()
                vel = agent.get_vel()
                next_pos = pos + vel * self.timestep
                speed = agent.get_speed()
                theta = agent.get_theta()
                goal = agent.get_goal()
                penalty_flag = 0
                state = np.array(
                    [pos[0], pos[1], theta, goal[0], goal[1],
                        vel[0], vel[1], speed, penalty_flag, next_pos[0], next_pos[1]]
                )

                # record agent's state
                self.update_state(
                    agent_index=agent.index, current_step=current_step, state=state)

                # record critical agents after removing redundant
                non_redundant_agents = agent.get_non_redundant_agents(
                )
                state = np.zeros(self.num_agents)
                state[non_redundant_agents] = 1
                self.update_critical_agents_state(
                    agent.index, current_step, state, False)

            self.update_adj_mat(env)
            ld2 = lambda2(self.adjacency_matrix)
            self.ld2s.append(ld2)

        if len(self.agents) > 0:
            for agent in self.agents:
                non_redundant_agents = agent.get_non_redundant_agents(
                )
                state = np.zeros(self.num_agents)
                state[non_redundant_agents] = 1
                self.update_critical_agents_state(
                    agent.index, current_step, state)

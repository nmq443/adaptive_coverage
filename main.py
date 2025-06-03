from configs import *
from swarm import Swarm
from environment import Environment
from simulator import Simulator

if __name__ == '__main__':
    env = Environment()
    swarm = Swarm()
    sim = Simulator(swarm, env)
    sim.execute()
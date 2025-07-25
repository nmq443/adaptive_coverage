import yaml
from adaptive_coverage.environment.environment import Environment
from adaptive_coverage.simulator.renderer import Renderer
from adaptive_coverage.utils.arg_parse import get_args
from adaptive_coverage.simulator.data_manager import LogManager, ResultManager


def render(configs_file):
    with open(configs_file, "r") as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)
    env = Environment(
        area_width=configs["environment"]["env0"]["area_width"],
        area_height=configs["environment"]["env0"]["area_height"],
        obstacles=configs["environment"]["env0"]["obstacles"],
        offset=1,
    )
    renderer = Renderer(
        screen_size=configs["simulation"]["screen_size"],
        trajectories_filepath="results/voronoi/env0/20_agents/run0/swarm_data.npy",
        controller="voronoi",
        agent_size=0.2,
        sensing_range=3.0,
        env=env,
        scale=40,
        linewidth=1,
        show_connections=False,
        show_goal=False,
        show_sensing_range=True,
        show_trajectories=False,
    )
    renderer.run()


def playback(configs_file):
    render(configs_file)


if __name__ == "__main__":
    playback("results/voronoi/env0/20_agents/run0/configs.yaml")

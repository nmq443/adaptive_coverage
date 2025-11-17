import os
import yaml
from adaptive_coverage.simulator.renderer import Renderer
from adaptive_coverage.environment.environment import Environment
from adaptive_coverage.simulator.data_manager import LogManager, ResultManager
from adaptive_coverage.utils.utils import get_args
import argparse


def render_from_results(results_dir: str):
    """Load configs.yaml from the given results directory and render the simulation."""
    configs_path = os.path.join(results_dir, "configs.yaml")
    if not os.path.exists(results_dir):
        raise FileNotFoundError(
            f"[ERROR] Result directory '{results_dir}' does not exist.")
    if not os.path.exists(configs_path):
        raise FileNotFoundError(
            f"[ERROR] No configs.yaml found in '{results_dir}'.")

    # Use your existing get_args() to load configs
    # args = get_args(configs_path)
    with open(configs_path, "r") as f:
        configs: dict = yaml.load(f, Loader=yaml.FullLoader)

    # Make sure res_dir points to the current folder (not the one in the YAML)
    configs['res_dir'] = results_dir
    env_name = next(iter(configs["environment"]))

    # Setup managers
    # result_manager = ResultManager(
    #     num_agents=configs['agents']['num_agents'],
    #     res_dir=results_dir,
    #     env_dir=env_name,
    #     controller=configs['agents']['controller'],
    #     original_method=configs['agents']['original_method'],
    #     create_new_dir=False
    # )
    # log_manager = LogManager(result_manager)

    # Environment
    env = Environment(
        configs['environment'][env_name]['area_width'],
        configs['environment'][env_name]['area_height'],
        configs['environment'][env_name]['vertices'],
        configs['environment'][env_name]['obstacles'],
        offset=1
    )

    # Renderer
    renderer = Renderer(
        screen_size=configs['simulation']['screen_size'],
        trajectories_filepath=os.path.join(results_dir, "swarm_data.npy"),
        controller=configs['agents']['controller'],
        agent_size=configs['agents']['agent_size'],
        sensing_range=configs['agents']['sensing_range'],
        critical_ratio=configs['agents']['critical_ratio'],
        result_manager=None,
        log_manager=None,
        save_video=True,
        res_dir=results_dir,
        env=env,
        linewidth=configs['simulation']['linewidth'],
        show_connections=configs['simulation']['show_connections'],
        show_sensing_range=configs['simulation']['show_sensing_range'],
        show_trajectories=configs['simulation']['show_trajectories'],
    )

    print(f"[INFO] Rendering results from: {results_dir}")
    renderer.run()


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(
    #     description="Render simulation from existing results.")
    # parser.add_argument(
    #     "results_dir",
    #     type=str,
    #     help="Path to the directory containing results (must include configs.yaml)"
    # )
    # args = parser.parse_args()

    results_dir = "results/voronoi/env1/40_agents/run0"

    render_from_results(results_dir)

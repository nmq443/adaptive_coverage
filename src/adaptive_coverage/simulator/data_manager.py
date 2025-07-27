import os
import shutil
import imageio
import logging


class ResultManager:
    def __init__(
        self, num_agents, res_dir, env_dir, controller, original_method, fps=144
    ):
        """
        This class helps with managing results.

        Args:
            num_agents: number of agents to simulate.
            res_dir: result directory.
            env_dir: name of environment.
            controller: controller to use (voronoi or hexagon).
            original_method: if using hexagon controller's original method or PSO method.
            fps: frames per second for video playback.
        """
        self.num_agents = num_agents
        self.res_dir = res_dir
        self.env_dir = env_dir
        self.controller = controller
        self.original_method = original_method
        self.frames = []
        self.fps = fps
        self.init_directories()

        self.video_path = os.path.join(self.res_dir, f"running_video.mp4")
        self.video_writer = imageio.get_writer(self.video_path, fps=self.fps)

        self.start_img_path = os.path.join(self.res_dir, "start_pose.png")
        self.end_img_path = os.path.join(self.res_dir, "final_pose.png")

        self.swarm_data_filepath = os.path.join(self.res_dir, "swarm_data.npy")
        self.ld2s_filepath = os.path.join(self.res_dir, "ld2s_data.npy")
        self.travel_distances_filepath = os.path.join(
            self.res_dir, "travel_distances.npy"
        )

    def init_directories(self):
        """
        Initialize the result tree.
        """

        # Determine the base directory
        if self.controller == "hexagon":
            method_dir = "original" if self.original_method else "pso"
            base_dir = os.path.join(
                self.res_dir,
                self.controller,
                self.env_dir,
                f"{self.num_agents}_agents",
                method_dir,
            )
        else:
            base_dir = os.path.join(
                self.res_dir, self.controller, self.env_dir, f"{self.num_agents}_agents"
            )

        # Create the base directory if it doesn't exist
        os.makedirs(base_dir, exist_ok=True)

        # Find the next available run number
        run_number = 0
        while os.path.exists(os.path.join(base_dir, f"run{run_number}")):
            run_number += 1

        # Set the new results directory
        self.res_dir = os.path.join(base_dir, f"run{run_number}")
        os.makedirs(self.res_dir)

    def update_video(self, frame):
        """
        Update current frame to playback video.

        Args:
            frame: current frame.
        """
        self.video_writer.append_data(frame)

    def update_frames(self, frame):
        """
        Add current frame to frames list.

        Args:
            frame: current frame.
        """
        self.frames.append(frame)

    def save_video(self):
        """Save playback video."""
        self.video_writer.close()

    def save_images(self):
        """Save first and final image."""
        imageio.imwrite(self.start_img_path, self.frames[1])
        imageio.imwrite(self.end_img_path, self.frames[-1])


class LogManager:
    def __init__(self, num_agents, log_dir, env_dir, controller, original_method):
        """
        This class helps with managing log.

        Args:
            num_agents: number of agents to simulate.
            log_dir: logging directory.
            env_dir: name of environment.
            controller: controller to use (voronoi or hexagon).
            original_method: if using hexagon controller's original method or PSO method.
        """
        self.num_agents = num_agents
        self.log_dir = log_dir
        self.env_dir = env_dir
        self.controller = controller
        self.original_method = original_method
        self.log_file = None

        self.init_directories()

        self.logger = logging.getLogger(__name__)
        logging.basicConfig(
            format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s",
            datefmt="%m-%d %H:%M",
            filemode="w",
            filename=self.log_file,
            level=logging.DEBUG,
        )
        # define a Handler which writes INFO messages or higher to the sys.stderr
        self.console = logging.StreamHandler()
        self.console.setLevel(logging.INFO)
        # set a format which is simpler for console use
        formatter = logging.Formatter("%(name)-12s: %(levelname)-8s %(message)s")
        # tell the handler to use this format
        self.console.setFormatter(formatter)
        # add the handler to the root logger
        logging.getLogger().addHandler(self.console)
        self.logger.info("Started")
        self.logger.info(f"Using {self.controller} method")

    def log(self, msg):
        self.logger.info(msg)

    def init_directories(self):
        """
        Initialize the log tree.
        """

        # Determine the base directory
        if self.controller == "hexagon":
            method_dir = "original" if self.original_method else "pso"
            base_dir = os.path.join(
                self.log_dir,
                self.controller,
                self.env_dir,
                f"{self.num_agents}_agents",
                method_dir,
            )
        else:
            base_dir = os.path.join(
                self.log_dir, self.controller, self.env_dir, f"{self.num_agents}_agents"
            )

        # Create the base directory if it doesn't exist
        os.makedirs(base_dir, exist_ok=True)

        # Find the next available run number
        run_number = 0
        while os.path.exists(os.path.join(base_dir, f"run{run_number}")):
            run_number += 1

        # Set the new log directory and file
        self.log_dir = os.path.join(base_dir, f"run{run_number}")
        os.makedirs(self.log_dir)
        self.log_file = os.path.join(self.log_dir, f"{self.controller}_log.log")

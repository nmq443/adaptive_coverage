import os
import shutil
import imageio
import logging


class ResultManager:
    def __init__(self, num_agents, res_dir, env_dir, controller, original_method, fps=144):
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
        self.travel_distances_filepath = os.path.join(self.res_dir, "travel_distances.npy")

    def init_directories(self):
        if self.controller == "hexagon":
            if self.original_method:
                dir = "original"
            else:
                dir = "pso"
            self.res_dir = os.path.join(
                self.res_dir, self.controller, self.env_dir, f"{self.num_agents}_agents", dir
            )
        else:
            self.res_dir = os.path.join(
                self.res_dir, self.controller, self.env_dir, f"{self.num_agents}_agents"
            )
        if os.path.exists(self.res_dir):
            shutil.rmtree(self.res_dir)
        os.makedirs(self.res_dir, exist_ok=True)

    def update_video(self, frame):
        self.video_writer.append_data(frame)

    def update_frames(self, frame):
        self.frames.append(frame)

    def update_images(self, start_img, final_img):
        imageio.imwrite(self.start_img_path, start_img)
        imageio.imwrite(self.end_img_path, final_img)

    def save_video(self):
        self.video_writer.close()

    def save_images(self):
        imageio.imwrite(self.start_img_path, self.frames[0])
        imageio.imwrite(self.end_img_path, self.frames[-1])


class LogManager:
    def __init__(self, num_agents, log_dir, env_dir, controller, original_method):
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
        if self.controller == "hexagon":
            if self.original_method:
                dir = "original"
            else:
                dir = "pso"
            self.log_dir = os.path.join(
                self.log_dir, self.controller, self.env_dir, f"{self.num_agents}_agents", dir
            )
        else:
            self.log_dir = os.path.join(
                self.log_dir, self.controller, self.env_dir, f"{self.num_agents}_agents"
            )
        if os.path.exists(self.log_dir):
            shutil.rmtree(self.log_dir)
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_file = os.path.join(self.log_dir, f"{self.controller}_log.log")

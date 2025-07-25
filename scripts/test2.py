import pygame
import numpy as np
import os  # For checking if the file exists


class TrajectoryPlayer:
    def __init__(
        self,
        trajectories_filepath,
        screen_size=(1600, 900),
        fps=60,
        scale=20,
        background_color=(255, 255, 255),
        agent_color=(0, 0, 255),
        trail_color=(150, 150, 255),
        agent_radius=5,
        trail_length=50,
    ):
        """
        Initializes the TrajectoryPlayer.

        Args:
            trajectories_filepath (str): Path to the .npy file containing the trajectories data.
                                         Expected shape: (num_agents, num_timesteps, 2).
            screen_size (tuple): Width and height of the Pygame window.
            fps (int): Frames per second for playback.
            scale (float): Pixels per unit in your simulation coordinates (e.g., if 20 pixels = 1 meter).
            background_color (tuple): RGB color for the screen background.
            agent_color (tuple): RGB color for drawing the agents.
            trail_color (tuple): RGB color for drawing the agents' trails.
            agent_radius (int): Radius of the circle representing the agent.
            trail_length (int): Number of previous positions to draw for the trail.
        """
        self.trajectories_filepath = trajectories_filepath
        self.screen_size = screen_size
        self.fps = fps
        self.scale = scale
        self.background_color = background_color
        self.agent_color = agent_color
        self.trail_color = trail_color
        self.agent_radius = agent_radius
        self.trail_length = trail_length

        self.trajectories_data = None
        self.num_agents = 0
        self.num_timesteps = 0
        self.current_timestep = 0

        self.screen = None
        self.clock = None
        self.running = False

    def load_data(self):
        """Loads the trajectory data from the specified .npy file."""
        if not os.path.exists(self.trajectories_filepath):
            print(f"Error: Trajectory file not found at {self.trajectories_filepath}")
            return False
        try:
            self.trajectories_data = np.load(self.trajectories_filepath)
            self.num_agents, self.num_timesteps, _ = self.trajectories_data.shape
            print(
                f"Loaded data: {self.num_agents} agents, {self.num_timesteps} timesteps."
            )
            return True
        except Exception as e:
            print(f"Error loading trajectory data: {e}")
            return False

    def init_pygame(self):
        """Initializes Pygame window and assets."""
        pygame.init()
        pygame.display.set_caption("Agent Trajectory Playback")
        self.screen = pygame.display.set_mode(self.screen_size)
        self.clock = pygame.time.Clock()
        self.running = True

    def convert_coords(self, x_sim, y_sim):
        """
        Converts simulation coordinates to Pygame screen coordinates.
        Assumes (0,0) in simulation is center of screen.
        """
        screen_x = int(self.screen_size[0] / 2 + x_sim * self.scale)
        screen_y = int(
            self.screen_size[1] / 2 - y_sim * self.scale
        )  # Pygame y-axis is inverted
        return screen_x, screen_y

    def draw(self):
        """Draws the current frame of the trajectories."""
        self.screen.fill(self.background_color)

        for i in range(self.num_agents):
            # Get current position
            current_pos_sim = self.trajectories_data[i, self.current_timestep, :]
            current_pos_screen = self.convert_coords(
                current_pos_sim[0], current_pos_sim[1]
            )

            # Draw agent
            pygame.draw.circle(
                self.screen, self.agent_color, current_pos_screen, self.agent_radius
            )

            # Draw trail
            start_trail_idx = max(0, self.current_timestep - self.trail_length)
            trail_points_sim = self.trajectories_data[
                i, start_trail_idx : self.current_timestep + 1, :
            ]

            # Convert all trail points to screen coordinates
            trail_points_screen = []
            for j in range(trail_points_sim.shape[0]):
                trail_points_screen.append(
                    self.convert_coords(trail_points_sim[j, 0], trail_points_sim[j, 1])
                )

            if len(trail_points_screen) > 1:
                pygame.draw.lines(
                    self.screen, self.trail_color, False, trail_points_screen, 1
                )  # Width 1 pixel

        pygame.display.flip()

    def run(self):
        """Runs the trajectory playback loop."""
        if not self.load_data():
            return

        self.init_pygame()

        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        self.running = False
                    elif event.key == pygame.K_SPACE:  # Pause/unpause with spacebar
                        self.running = False  # Temporary, just to break the loop. A better pause would be implemented.
                        # For a true pause, you'd toggle a 'paused' flag and skip current_timestep increment.
                        # Example: `if not self.paused: self.current_timestep += 1`

            if self.current_timestep < self.num_timesteps:
                self.draw()
                self.current_timestep += 1
            else:
                # Loop playback or stop when finished
                # self.current_timestep = 0 # Uncomment to loop
                # print("Playback finished.")
                self.running = False  # Stop when finished

            self.clock.tick(self.fps)

        pygame.quit()
        print("Playback stopped.")


# --- How to use it ---
if __name__ == "__main__":
    filepath = "results/voronoi/env0/20_agents/swarm_data.npy"
    # 2. Instantiate and run the TrajectoryPlayer
    player = TrajectoryPlayer(
        trajectories_filepath=filepath,
        screen_size=(1600, 900),  # Smaller window for quick test
        fps=30,  # Slower playback for observation
        scale=50,  # Adjust scale if your simulation units are very small/large
        agent_radius=3,
        trail_length=75,  # Longer trails
    )
    player.run()

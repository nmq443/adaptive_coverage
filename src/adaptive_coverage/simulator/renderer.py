from adaptive_coverage.simulator.data_manager import ResultManager, LogManager
from adaptive_coverage.environment.environment import Environment
from adaptive_coverage.agents.cvt.lloyd import compute_voronoi_diagrams
from adaptive_coverage.utils.utils import ray_intersects_aabb
import os
import numpy as np
from matplotlib.path import Path
from matplotlib.patches import Polygon, Rectangle, Circle, PathPatch
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.animation as animation
matplotlib.use("Agg")  # headless backend for server / no GUI


class Renderer:
    def __init__(
        self,
        env: Environment,
        agent_size: float,
        critical_ratio: float,
        sensing_range: float,
        screen_size,
        trajectories_filepath,
        result_manager: ResultManager,
        log_manager: LogManager,
        controller="voronoi",
        agent_color="red",
        agent_sensing_color="blue",
        goal_color="green",
        heading_color="green",
        index_color="black",
        obs_color="black",
        occupied_color="red",
        assigned_color="blue",
        unassigned_color="black",
        penalty_color="green",
        linewidth=1,
        trail_length=100,
        font_size=11,
        show_sensing_range=False,
        show_goal=False,
        show_connections=False,
        show_trajectories=False,
    ):
        self.env = env
        self.agent_size = agent_size
        self.critical_ratio = critical_ratio
        self.sensing_range = sensing_range
        self.screen_size = screen_size
        self.controller = controller
        self.linewidth = linewidth
        self.show_sensing_range = show_sensing_range
        self.show_connections = show_connections
        self.show_trajectories = show_trajectories
        self.show_goal = show_goal

        self.trajectories_filepath = trajectories_filepath
        self.result_manager = result_manager
        self.log_manager = log_manager

        self.agent_color = agent_color
        self.agent_sensing_color = agent_sensing_color
        self.heading_color = heading_color
        self.index_color = index_color
        if controller == "hexagon":
            self.occupied_color = occupied_color
            self.assigned_color = assigned_color
            self.unassigned_color = unassigned_color
            self.penalty_color = penalty_color

        self.current_timestep = 0
        self.num_timesteps = 0
        self.num_agents = 0

    def load_data(self):
        if not os.path.exists(self.trajectories_filepath):
            print(f"Trajectory file not found: {self.trajectories_filepath}")
            return False
        self.trajectories_data = np.load(self.trajectories_filepath)
        self.num_agents, self.num_timesteps, _ = self.trajectories_data.shape
        return True

    def draw_environment(self, ax):
        # draw boundary
        poly = Polygon(self.env.vertices, closed=True,
                       edgecolor="black", facecolor="white")
        ax.add_patch(poly)

        # draw obstacles
        for obs in self.env.obstacles:
            x, y, w, h = obs
            r = Rectangle((x, y), w, h, edgecolor="black", facecolor="black")
            ax.add_patch(r)

    def draw_agent(self, ax, idx, pos, penalty):
        color = self.penalty_color if penalty == 1 else self.agent_color
        circ = Circle(pos, self.agent_size, color=color)
        ax.add_patch(circ)
        ax.text(pos[0] + self.agent_size, pos[1] - self.agent_size, str(idx),
                color=self.index_color, fontsize=8)

    def draw_heading(self, ax, pos, yaw):
        a = pos
        b = pos + 2 * self.agent_size * np.array([np.cos(yaw), np.sin(yaw)])
        ax.plot([a[0], b[0]], [a[1], b[1]],
                color=self.heading_color, linewidth=1)

    def draw_sensing(self, ax, pos):
        # Create the sensing circle
        circ = Circle(pos, self.sensing_range, fill=False,
                      edgecolor=self.agent_sensing_color, facecolor=self.agent_sensing_color, linewidth=1, alpha=0.5)

        # Create clipping path from environment polygon
        env_path = Path(self.env.vertices)
        clip_patch = PathPatch(env_path, transform=ax.transData)

        # Apply clipping so circle is only drawn inside the environment
        circ.set_clip_path(clip_patch)

        ax.add_patch(circ)

    def draw_voronoi(self, ax, vor):
        for region in vor.filtered_regions:
            pts = vor.vertices[region + [region[0]]]
            for i in range(len(pts)-1):
                a = pts[i]
                b = pts[i+1]
                ax.plot([a[0], b[0]], [a[1], b[1]], color="black", linewidth=1)

    def render_frame(self, show_sensing_range=False, show_connections=False):
        fig, ax = plt.subplots(figsize=(self.screen_size[0]/100,
                                        self.screen_size[1]/100), dpi=100)
        ax.set_aspect('equal')
        ax.set_facecolor("white")

        # -----------------------------
        # compute bounds & add offset
        # -----------------------------
        xs = [v[0] for v in self.env.vertices]
        ys = [v[1] for v in self.env.vertices]

        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        offset = max(self.agent_size, 1.0)  # extra padding around map

        ax.set_xlim(min_x - offset, max_x + offset)
        ax.set_ylim(min_y - offset, max_y + offset)

        # -----------------------------
        # enable map-like visual style
        # -----------------------------
        # ax.set_xlabel("X [meters]")
        # ax.set_ylabel("Y [meters]")
        # ax.set_title("Environment Map View")

        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
        ax.set_xticks(np.arange(0, max_x, 1))
        ax.set_yticks(np.arange(0, max_y + 1, 1))

        ax.axis('on')

        self.draw_environment(ax)

        # -----------------------------
        # Draw agents
        # -----------------------------
        for i in range(self.num_agents):
            pos = self.trajectories_data[i, self.current_timestep, :2]
            yaw = self.trajectories_data[i, self.current_timestep, 2]
            penalty = self.trajectories_data[i, self.current_timestep, 8]

            self.draw_agent(ax, i, pos, penalty)
            self.draw_heading(ax, pos, yaw)
            if show_sensing_range:
                self.draw_sensing(ax, pos)
        if show_connections:
            for i in range(self.num_agents):
                for j in range(self.num_agents):
                    if i == j:
                        continue
                    pos_i = self.trajectories_data[i,
                                                   self.current_timestep, :2]
                    pos_j = self.trajectories_data[j,
                                                   self.current_timestep, :2]
                    rij = np.linalg.norm(pos_i - pos_j)
                    if rij <= self.sensing_range and not ray_intersects_aabb(pos_i, pos_j, self.env.obstacles):
                        ax.plot(
                            [pos_i[0], pos_j[0]],
                            [pos_i[1], pos_j[1]],
                            color=self.agent_color,
                            linestyle="--",
                            linewidth=0.8)
                        # alpha=0.6,
                        # )

        # -----------------------------
        # Draw Voronoi edges
        # -----------------------------
        if self.controller == "voronoi":
            gen = self.trajectories_data[:, self.current_timestep, :2]
            vor = compute_voronoi_diagrams(gen, self.env)
            self.draw_voronoi(ax, vor)

        fig.canvas.draw()
        buf = np.asarray(fig.canvas.renderer.buffer_rgba())
        frame = buf[:, :, :3]  # RGB only

        plt.close(fig)
        return frame

    def save_snapshot(self, timestep, tag):
        """Save 4 images at specific timestep."""
        self.current_timestep = timestep
        snapshots = {
            "pos": (False, False),
            "pos_sensing": (True, False),
            "pos_conn": (False, True),
            "pos_conn_sensing": (True, True)
        }

        for name, (sensing, conn) in snapshots.items():
            frame = self.render_frame(
                show_sensing_range=sensing, show_connections=conn)
            save_path = os.path.join(
                self.result_manager.res_dir, f"{tag}_{name}.png")
            plt.imsave(save_path, frame)
            self.log_manager.log(f"Saved {tag}_{name}.png")

    def run(self):
        """Render animation using matplotlib."""
        if not self.load_data():
            return

        # Save start and finish snapshot
        self.save_snapshot(0, "start")
        self.save_snapshot(self.num_timesteps - 1, "final")

        # Prepare figure once
        fig, ax = plt.subplots(
            figsize=(self.screen_size[0]/100, self.screen_size[1]/100), dpi=100)
        xs = [v[0] for v in self.env.vertices]
        ys = [v[1] for v in self.env.vertices]

        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        ax.set_aspect("equal")
        ax.set_facecolor("white")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
        ax.set_xticks(np.arange(0, max_x, 1))
        ax.set_yticks(np.arange(0, max_y + 1, 1))

        xs = [v[0] for v in self.env.vertices]
        ys = [v[1] for v in self.env.vertices]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        offset = max(self.agent_size, 1.0)
        ax.set_xlim(min_x - offset, max_x + offset)
        ax.set_ylim(min_y - offset, max_y + offset)

        # Draw static parts once
        self.draw_environment(ax)

        # Initialize artists for dynamic elements (agents, headings, sensing)
        agent_patches = []
        heading_lines = []
        sensing_patches = []

        for i in range(self.num_agents):
            pos = self.trajectories_data[i, 0, :2]
            yaw = self.trajectories_data[i, 0, 2]
            penalty = self.trajectories_data[i, 0, 8]
            color = self.penalty_color if penalty == 1 else self.agent_color
            circ = plt.Circle(pos, self.agent_size, color=color)
            ax.add_patch(circ)
            agent_patches.append(circ)

            # heading line
            a = pos
            b = pos + 2 * self.agent_size * \
                np.array([np.cos(yaw), np.sin(yaw)])
            line, = ax.plot([a[0], b[0]], [a[1], b[1]],
                            color=self.heading_color, linewidth=1)
            heading_lines.append(line)

            # sensing range (optional)
            circ2 = plt.Circle(pos, self.sensing_range, fill=False,
                               edgecolor=self.agent_sensing_color, linewidth=0.5, alpha=0.3)
            circ2.set_visible(self.show_sensing_range)
            ax.add_patch(circ2)
            sensing_patches.append(circ2)

        # Update function for each frame
        def update(frame_idx):
            for i in range(self.num_agents):
                pos = self.trajectories_data[i, frame_idx, :2]
                yaw = self.trajectories_data[i, frame_idx, 2]
                penalty = self.trajectories_data[i, frame_idx, 8]
                color = self.penalty_color if penalty == 1 else self.agent_color

                # Update agent circle
                agent_patches[i].center = pos
                agent_patches[i].set_color(color)

                # Update heading
                a = pos
                b = pos + 2 * self.agent_size * \
                    np.array([np.cos(yaw), np.sin(yaw)])
                heading_lines[i].set_data([a[0], b[0]], [a[1], b[1]])

                # Update sensing circle
                sensing_patches[i].center = pos

            return agent_patches + heading_lines + sensing_patches

        # Create animation
        ani = animation.FuncAnimation(fig, update, frames=self.num_timesteps,
                                      interval=1000/self.result_manager.fps,
                                      blit=True)

        # Save animation via ResultManager
        self.result_manager.save_video(ani)

        plt.close(fig)
        self.log_manager.log(
            f"Saved video to {self.result_manager.video_path}")

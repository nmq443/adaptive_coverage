import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
from adaptive_coverage.utils.utils import ray_intersects_aabb

# ================================
# Load data
# ================================
res_dir = "results/voronoi/env4/20_agents/run0/"
lambda2 = np.load(os.path.join(res_dir, "ld2s_data.npy"))
swarm_data = np.load(os.path.join(res_dir, "swarm_data.npy"))
critical_agents_data = np.load(os.path.join(res_dir, "critical_agents.npy"))

tol = 1e-6
zero_indices = np.where(np.abs(lambda2) < tol)[0]
if len(zero_indices) == 0:
    raise ValueError("No λ₂ ≈ 0 found in data.")

failure_idx = zero_indices[0]
start_idx = max(0, failure_idx - 2)
end_idx = min(len(lambda2), failure_idx + 3)
plot_indices = np.arange(start_idx, end_idx)

swarm_selected = swarm_data[:, plot_indices, :]
pos_x = swarm_selected[:, :, 0]
pos_y = swarm_selected[:, :, 1]
theta = swarm_selected[:, :, 2]
goal_x = swarm_selected[:, :, 3]
goal_y = swarm_selected[:, :, 4]
next_pos_x = swarm_selected[:, :, 9]
next_pos_y = swarm_selected[:, :, 10]

# ================================
# Load environment config
# ================================
with open(os.path.join(res_dir, "configs.yaml"), "r") as f:
    config = yaml.safe_load(f)

env_key = "env4"  # hardcoded
env_cfg = config["environment"][env_key]
agents_cfg = config["agents"]
area_width = env_cfg["area_width"]
area_height = env_cfg["area_height"]
obstacles = np.array(env_cfg["obstacles"])
sensing_range = agents_cfg["sensing_range"]

# ================================
# Visualization setup
# ================================
num_agents = pos_x.shape[0]
num_points = pos_x.shape[1]
arrow_scale = 0.3
colors = plt.cm.tab20(np.linspace(0, 1, num_agents))

fig, ax = plt.subplots(figsize=(9, 5))
current_t = 0  # start at first frame


def draw_frame(t):
    ax.clear()
    timestep = plot_indices[t]
    ax.set_xlim(0, area_width)
    ax.set_ylim(0, area_height)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x position")
    ax.set_ylabel("y position")
    title_color = "red" if timestep == failure_idx else "black"
    ax.set_title(f"Swarm at timestep = {timestep}", color=title_color)
    ax.grid(True)

    # Obstacles
    for obs in obstacles:
        ox, oy, w, h = obs
        rect = plt.Rectangle((ox, oy), w, h, color="gray", alpha=0.5)
        ax.add_patch(rect)

    # Agents + goals + orientation + next position
    for i in range(num_agents):
        x, y = pos_x[i, t], pos_y[i, t]
        gx, gy = goal_x[i, t], goal_y[i, t]
        nx, ny = next_pos_x[i, t], next_pos_y[i, t]
        th = theta[i, t]
        c = colors[i]

        ax.scatter(x, y, color=c, s=50, zorder=3)
        ax.arrow(
            x, y,
            arrow_scale * np.cos(th),
            arrow_scale * np.sin(th),
            head_width=0.08,
            head_length=0.12,
            fc=c,
            ec=c,
            zorder=3,
        )
        ax.scatter(gx, gy, color=c, marker="x", s=80, zorder=2)
        ax.scatter(nx, ny, color=c, marker='s', s=10, zorder=2)
        ax.text(x + 0.15, y + 0.15, str(i), fontsize=9, color=c)

    # Connection links (line-of-sight)
    for i in range(num_agents):
        for j in range(i + 1, num_agents):
            if ray_intersects_aabb(np.array((pos_x[i, t], pos_y[i, t])),
                                   np.array((pos_x[j, t], pos_y[j, t])),
                                   obstacles):
                continue
            dist = np.hypot(pos_x[i, t] - pos_x[j, t],
                            pos_y[i, t] - pos_y[j, t])
            if dist <= sensing_range:
                ax.plot(
                    [pos_x[i, t], pos_x[j, t]],
                    [pos_y[i, t], pos_y[j, t]],
                    color="lightblue",
                    linewidth=0.8,
                    alpha=0.6,
                    zorder=1,
                )
    fig.canvas.draw()


def on_key(event):
    global current_t
    if event.key in ["right", "d"]:
        current_t = (current_t + 1) % num_points
    elif event.key in ["left", "a"]:
        current_t = (current_t - 1) % num_points
    elif event.key in ["escape", "q"]:
        plt.close(fig)
        return
    draw_frame(current_t)


disconnect_step = failure_idx
t = disconnect_step - 1

print(f"\n=== State at timestep {t} ===")

for i in range(num_agents):
    x_, y_, theta_, gx_, gy_, dx_, dy_, speed_, penalty_, next_pos_x_, next_pos_y_ = swarm_data[
        i, t]
    critical_agents = critical_agents_data[i, t]
    print(
        f"Agent {i:02d}: "
        f"pos=({x_:.2f}, {y_:.2f})  "
        f"goal=({gx_:.2f}, {gy_:.2f})  "
        f"θ={theta_:.2f}  "
        f"v=({dx_:.2f}, {dy_:.2f})  "
        f"speed={speed_:.3f}  "
        f"penalty={penalty_}  "
        f"next pos=({next_pos_x_:.2f}, {next_pos_y_:.2f})  "
        # f"critical agents={np.where(np.array(critical_agents) != 0)}  "
        f"critical agents={critical_agents}  "
    )

'''
t = t + 1
print(f"State at step {t}")
for i in range(num_agents):
    x_, y_, theta_, gx_, gy_, dx_, dy_, speed_, penalty_, next_pos_x_, next_pos_y_ = swarm_data[
        i, t]
    print(
        f"Agent {i:02d}: "
        f"pos=({x_:.2f}, {y_:.2f})  "
        f"goal=({gx_:.2f}, {gy_:.2f})  "
        f"θ={theta_:.2f}  "
        f"v=({dx_:.2f}, {dy_:.2f})  "
        f"speed={speed_:.3f}  "
        f"penalty={penalty_}"
        f"next pos=({next_pos_x_:.2f}, {next_pos_y_:.2f})  "
    )
'''
fig.canvas.mpl_connect("key_press_event", on_key)
draw_frame(current_t)
plt.show()

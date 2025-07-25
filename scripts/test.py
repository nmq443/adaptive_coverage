import numpy as np
import matplotlib.pyplot as plt

filepath = "results/voronoi/env0/20_agents/swarm_data.npy"
# filepath = "results/hexagon/env0/20_agents/original/swarm_data.npy"
data = np.load(filepath)
print(data.shape)

num_agents, num_timesteps, _ = data.shape

plt.figure(figsize=(10, 8))  # Adjust figure size as needed
plt.title("Agent Trajectories Over Time")
plt.xlim(0, 40)
plt.ylim(0, 40)
plt.xlabel("X-coordinate")
plt.ylabel("Y-coordinate")
plt.grid(True)

for i in range(num_agents):
    # Extract x and y coordinates for the current agent
    x_coords = data[i, :, 0]
    y_coords = data[i, :, 1]

    # Plot the trajectory
    # plt.plot(x_coords, y_coords, label=f"Agent {i+1}")
    plt.plot(x_coords, y_coords)

    # Optionally, mark start and end points
    plt.plot(
        x_coords[0],
        y_coords[0],
        "go",
        markersize=2,
    )  # Green circle for start
    plt.plot(
        x_coords[-1],
        y_coords[-1],
        "ro",
        markersize=3,
    )  # Red 'o' for end

plt.legend()
plt.gca().set_aspect(
    "equal", adjustable="box"
)  # Keep aspect ratio equal for accurate spatial representation
plt.show()

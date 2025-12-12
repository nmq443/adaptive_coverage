import matplotlib.pyplot as plt
import numpy as np
# Assuming 'timesteps' is your X-data and 'lambda2_values' is your Y-data

lambda2_values = np.load("results/voronoi/env6/20_agents/run2/ld2s_data.npy")
timesteps = np.arange(len(lambda2_values))

plt.figure(figsize=(12, 8))
plt.plot(timesteps, lambda2_values, marker='.', linestyle='-',
         color='cornflowerblue', label='Lambda2 Value')

# 🌟 KEY CHANGE: Set the Y-axis scale to 'log'
plt.yscale('log')

plt.xlabel('Timestep')
plt.ylabel('Lambda2 Value (Log Scale)')
plt.title('Lambda2 Value Over Time (Logarithmic Y-axis)')
# 'both' ensures minor ticks are also gridded
plt.grid(True, which="both", ls="--", alpha=0.7)
plt.legend()
plt.show()

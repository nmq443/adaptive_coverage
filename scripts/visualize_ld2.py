import numpy as np
from adaptive_coverage.utils.evaluate import plot_ld2

path = "results/hexagon/env0/20_agents/original/ld2s_data.npy"
data = np.load(path)
plot_ld2(data)

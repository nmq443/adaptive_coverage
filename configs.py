import os
import shutil
import numpy as np
from utils import meters2pixels

# Simulation settings
SCREEN_SIZE = (1920, 1080)
RANDOM_INIT = True
SCALE = 50
LINE_WIDTH = 5
EPS = meters2pixels(0.1, SCALE)
CENTER = np.array([SCREEN_SIZE[0] / 2, SCREEN_SIZE[1] / 2]
                  )  # density function center
CENTER_COLOR = 'purple'
CENTER_SIZE = meters2pixels(0.5, SCALE)
LIMIT_RUNNING = True
FPS = 30
ITERATIONS = 500
RIDGE_COLOR = 'blue'
FONT_SIZE = 9
SHOW_SENSING_RANGE = True

# Swarm settings
# CONTROLLER = 'voronoi'  # 'hexagon' or 'voronoi'
CONTROLLER = 'hexagon'  # 'hexagon' or 'voronoi'
NUM_AGENTS = 31
if RANDOM_INIT:
    AGENT_SPREAD = meters2pixels(2., SCALE)

# Agent's settings
COLOR = 'red'
GOAL_COLOR = 'green'
SENSING_COLOR = 'blue' 
SIZE = meters2pixels(0.2, SCALE)
SENSING_RANGE = meters2pixels(6., SCALE) # rc and rs
VMAX = meters2pixels(1, SCALE)
DIST_BTW_AGENTS = meters2pixels(0.7, SCALE)
AGENT_ANCHOR_POS = np.array([SCREEN_SIZE[0] / 3 + 200, SCREEN_SIZE[1] / 2])
KG = 0.1
NUM_ROWS = 5
NUM_COLS = 4
INIT_POS = []
for i in range(NUM_ROWS):
    pos = []
    for j in range(NUM_COLS):
        x = AGENT_ANCHOR_POS[0] + j * DIST_BTW_AGENTS
        y = AGENT_ANCHOR_POS[1] + i * DIST_BTW_AGENTS
        pos.append([x, y])
    INIT_POS.append(pos)
INIT_POS = np.array(INIT_POS)
if CONTROLLER == 'hexagon':
    # HEXAGON_RANGE = meters2pixels(4.75, SCALE)  # rh
    HEXAGON_RANGE = 0.75 * SENSING_RANGE  # rh
    AVOIDANCE_RANGE = meters2pixels(0.5, SCALE)  # ra
    ASSIGNED_AGENT_COLOR = 'blue'
    OCCUPIED_AGENT_COLOR = 'black'
    UNASSIGNED_AGENT_COLOR = COLOR
    PENALTY_AGENT_COLOR = 'green'
    USE_PENALTY_NODE = False
    # PSO
    PSO_ITERATIONS = 100
    PSO_PARTICLES = 50
    PSO_VMAX = meters2pixels(0.5, SCALE)
    PSO_SPREAD = meters2pixels(0.05, SCALE)
else:
    VALID_RANGE = 0.75 * SENSING_RANGE

# Area
# Area 1 is a simple rectangle without obstacles
ENV = 1
ENV_ANCHOR_POS = np.array([SCREEN_SIZE[0] / 2, SCREEN_SIZE[1] / 3])
# obstacles are saved in (x, y, width, height) format
if ENV == 1:
    # OBSTACLES = np.array([
    #     [ENV_ANCHOR_POS[0], ENV_ANCHOR_POS[1], SCREEN_SIZE[0] / 2, SCREEN_SIZE[1] / 3]
    # ])  
    OBSTACLES = np.array([])
elif ENV == 2:
    OBSTACLES = np.array([
        [ENV_ANCHOR_POS[0], ENV_ANCHOR_POS[1], SCREEN_SIZE[0] / 2, SCREEN_SIZE[1] / 3],
        [ENV_ANCHOR_POS[0] / 2, ENV_ANCHOR_POS[1] / 2, SCREEN_SIZE[0] / 5, SCREEN_SIZE[1] / 5],
    ])
elif ENV == 3:
    OBSTACLES = np.array([
        [ENV_ANCHOR_POS[0], ENV_ANCHOR_POS[1], SCREEN_SIZE[0] / 2, SCREEN_SIZE[1] / 3],
        [ENV_ANCHOR_POS[0] / 2, ENV_ANCHOR_POS[1] / 2, SCREEN_SIZE[0] / 5, SCREEN_SIZE[1] / 5],
        [ENV_ANCHOR_POS[0] / 2, ENV_ANCHOR_POS[1] / 2 + 500, SCREEN_SIZE[0] / 5, SCREEN_SIZE[1] / 5],
    ])
VERTICES = np.array([
    [0, 0],
    [SCREEN_SIZE[0], 0],
    [SCREEN_SIZE[0], SCREEN_SIZE[1]],
    [0, SCREEN_SIZE[1]]
], dtype=float)


# Results
RES_DIR = "results"
METHOD_DIR = CONTROLLER
ENV_DIR = f"env_{ENV}"
os.makedirs(os.path.join(RES_DIR, METHOD_DIR, ENV_DIR), exist_ok=True)
if CONTROLLER == 'voronoi':
    VIDEO_NAME = "uniform_density.mp4"
    START_FIG = "start_pose.png"
    FINAL_FIG = "final_pose.png"
else:
    VIDEO_NAME = "hexagonal_lattices.mp4"
    START_FIG = "start_pose.png"
    FINAL_FIG = "final_pose.png"

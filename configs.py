import os
import shutil
import numpy as np
from utils import meters2pixels

# Simulation settings
SCREEN_SIZE = (1920, 1088)
RANDOM_INIT = False
SCALE = 50
LINE_WIDTH = 5
EPS = meters2pixels(0.01, SCALE)
CENTER = np.array([SCREEN_SIZE[0] / 2, SCREEN_SIZE[1] / 2])  # density function center
CENTER_COLOR = "purple"
CENTER_SIZE = meters2pixels(0.5, SCALE)
LIMIT_RUNNING = True
SAVE_VIDEO = True
FPS = 30
ITERATIONS = 300
RIDGE_COLOR = "black"
FONT_SIZE = 13
SHOW_SENSING_RANGE = False
SHOW_CONNECTIONS = True
SHOW_TRAJECTORY = False

# Swarm settings
# CONTROLLER = "voronoi"  # 'hexagon' or 'voronoi'
CONTROLLER = "hexagon"  # 'hexagon' or 'voronoi'
NUM_AGENTS = 30
if RANDOM_INIT:
    AGENT_SPREAD = meters2pixels(0.5, SCALE)

# Agent's settings
COLOR = "red"
GOAL_COLOR = "green"
SENSING_COLOR = "blue"
SIZE = meters2pixels(0.2, SCALE)
SENSING_RANGE = meters2pixels(6.0, SCALE)  # rc and rs
AVOIDANCE_RANGE = SIZE + meters2pixels(0.2, SCALE)  # ra
VMAX = meters2pixels(0.1, SCALE)
DIST_BTW_AGENTS = meters2pixels(0.7, SCALE)
AGENT_ANCHOR_POS = np.array(
    [SCREEN_SIZE[0] / 8, SCREEN_SIZE[1] / 3 + SCREEN_SIZE[1] / 10]
)
KG = 0.5
KA = 0.5
BETA_C = 1.0
KO = 0.1
NUM_ROWS = 5
NUM_COLS = 6
INIT_POS = []
for i in range(NUM_ROWS):
    pos = []
    for j in range(NUM_COLS):
        x = AGENT_ANCHOR_POS[0] + j * DIST_BTW_AGENTS
        y = AGENT_ANCHOR_POS[1] + i * DIST_BTW_AGENTS
        pos.append([x, y])
    INIT_POS.append(pos)
INIT_POS = np.array(INIT_POS)
if CONTROLLER == "hexagon":
    HEXAGON_RANGE = 0.8 * SENSING_RANGE  # rh
    ASSIGNED_AGENT_COLOR = "blue"
    OCCUPIED_AGENT_COLOR = "black"
    UNASSIGNED_AGENT_COLOR = COLOR
    PENALTY_AGENT_COLOR = "green"
    USE_PENALTY_NODE = True
    ORIGINAL_METHOD = True
    RHO = 1.0
    NV = 6
    # PSO
    PSO_ITERATIONS = 50
    PSO_PARTICLES = 20
    PSO_VMAX = meters2pixels(0.25, SCALE)
    PSO_SPREAD = meters2pixels(0.05, SCALE)
else:
    VALID_RANGE = 0.75 * SENSING_RANGE

# Area
# Area 1 is a simple rectangle without obstacles
# Area 2 is a hexagon
# Area 3 is an octagon
# Area 4 is an office-like environment
# Area 5 is a simple non-convex environment
ENV = 4
VERTICES = np.array(
    [
        [0, 0],
        [SCREEN_SIZE[0], 0],
        [SCREEN_SIZE[0], SCREEN_SIZE[1]],
        [0, SCREEN_SIZE[1]],
    ],
    dtype=float,
)
ENV_ANCHOR_POS = np.array([SCREEN_SIZE[0] / 2, SCREEN_SIZE[1] / 3])
# obstacles are saved in (x, y, width, height) format
if ENV == 1:
    OBSTACLES = np.array([])
elif ENV == 2:
    VERTICES = np.array(
        [
            [SCREEN_SIZE[0] / 3, 0],
            [SCREEN_SIZE[0] / 3 * 2, 0],
            [SCREEN_SIZE[0], SCREEN_SIZE[1] / 2],
            [SCREEN_SIZE[0] / 3 * 2, SCREEN_SIZE[1]],
            [SCREEN_SIZE[0] / 3, SCREEN_SIZE[1]],
            [0, SCREEN_SIZE[1] / 2],
        ],
        dtype=float,
    )
    OBSTACLES = np.array([])
    ITERATIONS = 500
elif ENV == 3:
    VERTICES = np.array(
        [
            [0, SCREEN_SIZE[1] / 3],
            [SCREEN_SIZE[0] / 4, 0],
            [SCREEN_SIZE[0] / 4 * 3, 0],
            [SCREEN_SIZE[0], SCREEN_SIZE[1] / 3],
            [SCREEN_SIZE[0], SCREEN_SIZE[1] / 3 * 2],
            [SCREEN_SIZE[0] / 4 * 3, SCREEN_SIZE[1]],
            [SCREEN_SIZE[0] / 4, SCREEN_SIZE[1]],
            [0, SCREEN_SIZE[1] / 3 * 2],
        ],
        dtype=float,
    )
    OBSTACLES = np.array([])
    ITERATIONS = 500
elif ENV == 4:
    OBSTACLES = np.array(
        [
            # upper half
            [0, SCREEN_SIZE[1] / 3, SCREEN_SIZE[0] / 6, SCREEN_SIZE[1] / 20],
            [
                SCREEN_SIZE[0] / 5,
                SCREEN_SIZE[1] / 3,
                SCREEN_SIZE[0] / 8 * 2,
                SCREEN_SIZE[1] / 20,
            ],
            [
                SCREEN_SIZE[0] / 5 + SCREEN_SIZE[0] / 12,
                0,
                SCREEN_SIZE[0] / 25,
                SCREEN_SIZE[1] / 3,
            ],
            [
                SCREEN_SIZE[0] / 2,
                SCREEN_SIZE[1] / 3,
                SCREEN_SIZE[0] / 8 * 2,
                SCREEN_SIZE[1] / 20,
            ],
            [
                SCREEN_SIZE[0] / 2 + SCREEN_SIZE[0] / 12,
                0,
                SCREEN_SIZE[0] / 25,
                SCREEN_SIZE[1] / 3,
            ],
            [
                SCREEN_SIZE[0] / 2 + SCREEN_SIZE[0] / 5 * 1.5,
                SCREEN_SIZE[1] / 3,
                SCREEN_SIZE[0] / 5,
                SCREEN_SIZE[1] / 20,
            ],
            # lower half
            # [0, SCREEN_SIZE[1] / 2, SCREEN_SIZE[0] / 5, SCREEN_SIZE[1] / 20],
            [
                SCREEN_SIZE[0] / 7 * 2,
                SCREEN_SIZE[1] / 2,
                SCREEN_SIZE[0] / 5,
                SCREEN_SIZE[1] / 20,
            ],
            [
                SCREEN_SIZE[0] / 5 + SCREEN_SIZE[0] / 6,
                SCREEN_SIZE[1] / 2 + SCREEN_SIZE[1] / 20,
                SCREEN_SIZE[0] / 25,
                SCREEN_SIZE[1] / 3 * 2,
            ],
            [
                SCREEN_SIZE[0] / 7 * 2 + SCREEN_SIZE[0] / 5 * 1.5,
                SCREEN_SIZE[1] / 2,
                SCREEN_SIZE[0] / 5,
                SCREEN_SIZE[1] / 20,
            ],
            [
                SCREEN_SIZE[0] / 5 + SCREEN_SIZE[0] / 2,
                SCREEN_SIZE[1] / 2 + SCREEN_SIZE[1] / 20,
                SCREEN_SIZE[0] / 25,
                SCREEN_SIZE[1] / 3 * 2,
            ],
            [
                SCREEN_SIZE[0] / 7 * 2
                + SCREEN_SIZE[0] / 5 * 1.5
                + SCREEN_SIZE[0] / 5 * 1.5,
                SCREEN_SIZE[1] / 2,
                SCREEN_SIZE[0] / 5,
                SCREEN_SIZE[1] / 20,
            ],
        ]
    )
    ITERATIONS = 1000
elif ENV == 5:
    OBSTACLES = np.array(
        [[ENV_ANCHOR_POS[0], ENV_ANCHOR_POS[1], SCREEN_SIZE[0] / 2, SCREEN_SIZE[1] / 3]]
    )
    ITERATIONS = 1000

# Logging
LOG_DIR = "log"
LOG_FILE = "hexagon_log.log" if CONTROLLER == "hexagon" else "voronoi_log.log"

# Results
RES_DIR = "results"
METHOD_DIR = CONTROLLER
ENV_DIR = f"env_{ENV}"
os.makedirs(os.path.join(RES_DIR, METHOD_DIR, ENV_DIR), exist_ok=True)
if CONTROLLER == "voronoi":
    VIDEO_NAME = "uniform_density.mp4"
    START_FIG = "start_pose.png"
    FINAL_FIG = "final_pose.png"
else:
    if ORIGINAL_METHOD:
        VIDEO_NAME = "hexagonal_lattices_original.mp4"
        START_FIG = "start_pose.png"
        FINAL_FIG = "final_pose.png"
    else:
        VIDEO_NAME = "hexagonal_lattices_pso.mp4"
        START_FIG = "start_pose.png"
        FINAL_FIG = "final_pose.png"

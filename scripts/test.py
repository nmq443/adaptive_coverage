import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point, box
from shapely.ops import unary_union
import numpy as np
from matplotlib.patches import Polygon as MplPolygon

# Agent class


class Agent:
    def __init__(self, id, pos, sensing_range):
        self.id = id
        self.pos = pos
        self.sensing_range = sensing_range


def clipped_region_centroid(voronoi_cell_polygon, agent, obstacles=[], circle_resolution=64):
    """
    Compute centroid of free region = Voronoi cell ∩ sensing circle \ obstacles
    """
    cell = Polygon(voronoi_cell_polygon)

    sensing_circle = Point(agent.pos[0], agent.pos[1]).buffer(
        agent.sensing_range, resolution=circle_resolution
    )

    if len(obstacles) > 0:
        obstacles_union = unary_union(obstacles)
        free_space = cell.intersection(
            sensing_circle).difference(obstacles_union)
    else:
        free_space = cell.intersection(sensing_circle)

    if free_space.is_empty:
        print(f"Agent {agent.id} has no free space.")
        return agent.pos, free_space

    # Better centroid calculation
    if free_space.geom_type == "MultiPolygon":
        total_area = sum(p.area for p in free_space.geoms)
        if total_area == 0:
            return agent.pos, free_space
        cx = sum(p.centroid.x * p.area for p in free_space.geoms) / total_area
        cy = sum(p.centroid.y * p.area for p in free_space.geoms) / total_area
        centroid = (cx, cy)
    else:
        centroid = (free_space.centroid.x, free_space.centroid.y)

    return centroid, free_space


# ========== DEMO ==========
if __name__ == "__main__":
    # Workspace rectangle
    workspace = Polygon([(0, 0), (10, 0), (10, 6), (0, 6)])

    # Example obstacle (black rectangle)
    obstacles = [box(3, 2, 7, 5)]

    # Example agent
    agent = Agent(1, pos=(2, 3), sensing_range=3.0)

    # Fake "Voronoi cell" (here just the whole workspace for demo)
    voronoi_cell = workspace.exterior.coords

    # Compute centroid & free space
    centroid, free_space = clipped_region_centroid(
        voronoi_cell, agent, obstacles)

    # ---------- Plot ----------
    fig, ax = plt.subplots()
    ax.set_aspect('equal')

    # Draw workspace
    ax.add_patch(MplPolygon(list(workspace.exterior.coords),
                 closed=True, fc='none', ec='gray', linestyle='--'))

    # Draw obstacle
    # for obs in obstacles:
    # ax.add_patch(MplPolygon(list(obs.exterior.coords),
    #              closed=True, fc='black', ec='black'))
    # ax.add_patch(MplPolygon(list(obs.exterior.coords),
    #              closed=True, fill=False, ec='black'))

    # Draw sensing circle
    sensing_circle = Point(agent.pos).buffer(
        agent.sensing_range, resolution=128)
    ax.add_patch(MplPolygon(list(sensing_circle.exterior.coords),
                 closed=True, fc='none', ec='blue', linestyle=':'))

    # Draw free space
    if not free_space.is_empty:
        if free_space.geom_type == "MultiPolygon":
            for p in free_space.geoms:
                ax.add_patch(MplPolygon(list(p.exterior.coords),
                             closed=True, fc='lightgreen', alpha=0.5))
        else:
            ax.add_patch(MplPolygon(list(free_space.exterior.coords),
                         closed=True, fc='lightgreen', alpha=0.5))

    # Draw agent
    ax.plot(agent.pos[0], agent.pos[1], 'ro', label='Agent')

    # Draw centroid
    ax.plot(centroid[0], centroid[1], 'bx', markersize=10, label='Centroid')

    ax.legend()
    plt.title("Free Area and Centroid")
    plt.show()

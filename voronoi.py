import pygame
import numpy as np
from configs import *
from scipy.spatial import Voronoi
from shapely.geometry import Point
from utils import perpendicular


def compute_voronoi_diagrams(generators, env):
    """
    Compute bounded voronoi diagrams inside a polygon. 

    Parameters
    ----------
    generators : np.ndarray
        The generators array for computing polygon.
    polygon : shapely.Polygon
        The bounding polygon.

    Returns
    -------
    vor : scipy.spatial.Voronoi
        The resulting voronoi
    """
    mirroreds = []
    # mirror over edges
    for generator in generators:
        for edge in env.edges:
            projected_on_edge = perpendicular(
                generator, edge[0], edge[1])
            mirrored = 2 * projected_on_edge - generator
            mirroreds.append(mirrored)
    mirroreds = np.array(mirroreds)
    new_generators = np.vstack((generators, mirroreds))
    vor = Voronoi(new_generators)

    # Filter regions
    regions = []
    for region in vor.regions:
        flag = True
        for vertex_idx in region:
            if vertex_idx == -1:
                flag = False
                break
            vertex = vor.vertices[vertex_idx]
            if not (env.contains(vertex)):
                flag = False
                break
        if region and flag:
            regions.append(region)
    vor.filtered_points = generators
    vor.filtered_regions = regions
    return vor


def draw_voronoi(vor, surface):
    # Plot ridges points
    # for region in vor.filtered_regions:
    #     vertices = vor.vertices[region, :]
    #     for vertex in vertices:
    #         pygame.draw.circle(surface, 'green', vertex, 2)

    # Plot ridges
    for region in vor.filtered_regions:
        vertices = vor.vertices[region + [region[0]], :]
        for i in range(len(vertices) - 1):
            pygame.draw.line(surface, RIDGE_COLOR, vertices[i], vertices[i + 1], LINE_WIDTH)
        pygame.draw.line(surface, RIDGE_COLOR, vertices[0], vertices[-1], LINE_WIDTH)

    # Plot vertices
    # for region in vor.filtered_regions:
    #     vertices = vor.vertices[region, :]
    #     for vertex in vertices:
    #         pygame.draw.circle(surface, 'black', vertex, 5)

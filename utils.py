import numpy as np


def meters2pixels(x, scale):
    return x * scale


def perpendicular(x: np.array, a: np.array, b: np.array):
    """
    Finds projection of x on ab.

    Args:
        a: NumPy array [ax, ay] representing point A.
        b: NumPy array [bx, by] representing point B.
        c: NumPy array [cx, cy] representing point C.

    Returns:
        NumPy array [hx, hy] representing point H.
    """
    ab = b - a  # Vector AB
    ac = x - a  # Vector AC
    t = (ac @ ab) / (ab @ ab)
    return a + t * ab


def ray_intersects_aabb(p1, p2, obstacles):
    """Test if segment [p1, p2] intersects any AABB in 'obstacles'."""
    if len(obstacles) <= 0:
        return False
    p1 = np.asarray(p1)
    p2 = np.asarray(p2)

    # Directions
    d = p2 - p1  # shape: (2,)

    # Avoid division by zero
    d = np.where(d == 0, 1e-12, d)

    # Unpack rects: shape (N,)
    x_min = obstacles[:, 0]
    y_min = obstacles[:, 1]
    x_max = x_min + obstacles[:, 2]
    y_max = y_min + obstacles[:, 3]

    # Parametric t values
    tx1 = (x_min - p1[0]) / d[0]
    tx2 = (x_max - p1[0]) / d[0]
    ty1 = (y_min - p1[1]) / d[1]
    ty2 = (y_max - p1[1]) / d[1]

    tmin = np.maximum(np.minimum(tx1, tx2), np.minimum(ty1, ty2))
    tmax = np.minimum(np.maximum(tx1, tx2), np.maximum(ty1, ty2))

    # Segment overlaps if tmin ≤ tmax and tmax ≥ 0 and tmin ≤ 1
    overlaps = (tmax >= 0) & (tmin <= tmax) & (tmin <= 1)

    return np.any(overlaps)

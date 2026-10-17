# ========================================================================
# The copyrights of the software are owned by Duke University.
# Developed by Ming-Yang Ho and Alberto Bartesaghi at the Bartesaghi Lab,
# Department of Computer Science, Trinity College of Arts & Sciences, Duke
# University.
# ========================================================================
import typing as t

import numpy as np
import numpy.typing as npt


def rotate_coordinate(
    xyz_indices: npt.NDArray[t.Union[int, float]],
    theta: float,
    W: int,
    D: int,
) -> npt.NDArray[int]:
    """
    Rotate 3D coordinates around the y-axis by a given tilt angle.

    This function performs a rotation in the x-z plane (i.e., around the y-axis),
    which is common in tomographic projection or 3D-to-2D tilt simulations.
    The rotation is centered around the midpoint of the given image width (W)
    and depth (D) to maintain consistent spatial alignment.

    Parameters
    ----------
    xyz_indices : np.ndarray of shape (N, 3)
        Input 3D coordinates or voxel indices, where each row corresponds to (x, y, z).
    theta : float
        Rotation angle in degrees. Positive values correspond to counterclockwise
        rotation when viewing from the top of the y-axis.
    W : int
        Width (x-dimension) of the volume or image; used to define the rotation center.
    D : int
        Depth (z-dimension) of the volume or image; used to define the rotation center.

    Returns
    -------
    np.ndarray of shape (N, 3), dtype=int
        Rotated coordinates with integer rounding applied to match voxel grid indices.

    Notes
    -----
    - The rotation is defined by:
        x' = (x - W/2) * cos(θ) + (z - D/2) * sin(θ) + W/2
        z' = -(x - W/2) * sin(θ) + (z - D/2) * cos(θ) + D/2
    - The y-coordinate remains unchanged.
    - The output coordinates are rounded and cast to `int`.

    Examples
    --------
    >>> coords = np.array([[10, 5, 20], [30, 5, 40]])
    >>> rotate_coordinate(coords, theta=30, W=64, D=64)
    array([[7, 5, 33],
           [34, 5, 40]])
    """
    xyz_indices = xyz_indices.copy().astype(float)
    x = xyz_indices[:, 0]
    z = xyz_indices[:, 2]

    theta_rad = np.deg2rad(theta)
    cos_t = np.cos(theta_rad)
    sin_t = np.sin(theta_rad)

    x_rot = (x - W / 2) * cos_t + (z - D / 2) * sin_t + W / 2
    z_rot = -(x - W / 2) * sin_t + (z - D / 2) * cos_t + D / 2

    rot_xyz_indices = np.zeros_like(xyz_indices)
    rot_xyz_indices[:, 0] = x_rot
    rot_xyz_indices[:, 1] = xyz_indices[:, 1]
    rot_xyz_indices[:, 2] = z_rot

    return np.around(rot_xyz_indices).astype(int)

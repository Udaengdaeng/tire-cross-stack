"""3D tire mesh generation utilities.

The local tire coordinate system is:
- x: tire width direction
- y-z: tire front circular plane
"""

from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np
import plotly.graph_objects as go


def rotation_matrix_axis_angle(axis, angle_deg: float) -> np.ndarray:
    """Create a 3x3 rotation matrix from an axis and angle in degrees."""

    axis_arr = np.asarray(axis, dtype=float)
    norm = np.linalg.norm(axis_arr)
    if norm == 0:
        raise ValueError("rotation axis must be non-zero")
    x, y, z = axis_arr / norm
    angle = math.radians(angle_deg)
    c = math.cos(angle)
    s = math.sin(angle)
    C = 1.0 - c
    return np.array(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ]
    )


def rotation_matrix_from_euler(rx_deg=0.0, ry_deg=0.0, rz_deg=0.0) -> np.ndarray:
    """Create a 3x3 rotation matrix from XYZ Euler angles in degrees."""

    rx, ry, rz = map(math.radians, (rx_deg, ry_deg, rz_deg))
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def create_tire_mesh(
    D: float,
    d: float,
    SW: float,
    center: tuple[float, float, float],
    rotation_matrix: np.ndarray,
    radial_segments: int = 48,
    width_segments: int = 8,
    add_tread: bool = True,
    tire_color: str = "rgb(45,45,45)",
    opacity: float = 1.0,
    hover_text: str = "",
) -> go.Mesh3d:
    """Create a hollow cylindrical tire mesh with tread and sidewalls."""

    if D <= 0 or d <= 0 or SW <= 0:
        raise ValueError("D, d, and SW must be positive")
    if d >= D:
        raise ValueError("inner diameter d must be smaller than outer diameter D")
    if rotation_matrix.shape != (3, 3):
        raise ValueError("rotation_matrix must be 3x3")

    R_outer = D / 2.0
    R_inner = d / 2.0
    radial_segments = max(12, int(radial_segments))
    width_segments = max(2, int(width_segments))

    vertices: List[Tuple[float, float, float]] = []
    intensities: List[float] = []
    faces: List[Tuple[int, int, int]] = []

    outer_idx = _add_cylinder_surface(
        vertices,
        intensities,
        radius=R_outer,
        SW=SW,
        radial_segments=radial_segments,
        width_segments=width_segments,
        outer=True,
        add_tread=add_tread,
    )
    inner_idx = _add_cylinder_surface(
        vertices,
        intensities,
        radius=R_inner,
        SW=SW,
        radial_segments=radial_segments,
        width_segments=width_segments,
        outer=False,
        add_tread=False,
    )

    _grid_faces(faces, outer_idx, radial_segments, width_segments, reverse=False)
    _grid_faces(faces, inner_idx, radial_segments, width_segments, reverse=True)
    _sidewall_faces(faces, outer_idx, inner_idx, radial_segments, width_i=0, reverse=True)
    _sidewall_faces(faces, outer_idx, inner_idx, radial_segments, width_i=width_segments, reverse=False)

    local = np.asarray(vertices)
    world = (rotation_matrix @ local.T).T + np.asarray(center)

    return go.Mesh3d(
        x=world[:, 0],
        y=world[:, 1],
        z=world[:, 2],
        i=[f[0] for f in faces],
        j=[f[1] for f in faces],
        k=[f[2] for f in faces],
        intensity=intensities,
        colorscale=[
            [0.0, "rgb(42,42,42)"],
            [0.45, tire_color],
            [0.78, tire_color],
            [1.0, "rgb(180,180,180)"],
        ],
        cmin=0,
        cmax=1,
        flatshading=False,
        opacity=opacity,
        showscale=False,
        lighting={
            "ambient": 0.58,
            "diffuse": 0.9,
            "specular": 0.28,
            "roughness": 0.72,
            "fresnel": 0.18,
        },
        lightposition={"x": 220, "y": -260, "z": 420},
        hovertemplate=hover_text + "<extra></extra>",
    )


def quality_to_segments(mesh_quality: str) -> tuple[int, int]:
    """Map UI quality labels to radial and width segment counts."""

    if mesh_quality == "low":
        return 24, 4
    if mesh_quality == "medium":
        return 36, 6
    if mesh_quality == "high":
        return 56, 10
    raise ValueError("mesh_quality must be low, medium, or high")


def _add_cylinder_surface(
    vertices: List[Tuple[float, float, float]],
    intensities: List[float],
    radius: float,
    SW: float,
    radial_segments: int,
    width_segments: int,
    outer: bool,
    add_tread: bool,
) -> list[list[int]]:
    idx_grid: list[list[int]] = []
    for wi in range(width_segments + 1):
        x = -SW / 2.0 + SW * wi / width_segments
        row = []
        width_t = wi / max(width_segments, 1)
        for ri in range(radial_segments):
            phi = 2.0 * math.pi * ri / radial_segments
            tread = 0.0
            if outer and add_tread:
                circum = max(0.0, math.sin(24.0 * phi))
                lateral = max(0.0, math.sin(5.0 * math.pi * width_t))
                tread = 0.018 * SW * circum + 0.012 * SW * lateral
            r = radius - tread if outer else radius
            y = r * math.cos(phi)
            z = r * math.sin(phi)
            row.append(len(vertices))
            vertices.append((x, y, z))
            base = 0.48 + 0.22 * math.cos(phi - math.pi / 4.0)
            if outer:
                base += 0.18 * (0.5 + 0.5 * math.sin(18.0 * phi))
            else:
                base -= 0.18
            intensities.append(float(np.clip(base, 0.0, 1.0)))
        idx_grid.append(row)
    return idx_grid


def _grid_faces(faces, idx_grid, radial_segments, width_segments, reverse: bool) -> None:
    for wi in range(width_segments):
        for ri in range(radial_segments):
            rn = (ri + 1) % radial_segments
            a = idx_grid[wi][ri]
            b = idx_grid[wi + 1][ri]
            c = idx_grid[wi + 1][rn]
            d = idx_grid[wi][rn]
            _add_quad(faces, a, b, c, d, reverse)


def _sidewall_faces(faces, outer_idx, inner_idx, radial_segments, width_i: int, reverse: bool) -> None:
    for ri in range(radial_segments):
        rn = (ri + 1) % radial_segments
        a = outer_idx[width_i][ri]
        b = outer_idx[width_i][rn]
        c = inner_idx[width_i][rn]
        d = inner_idx[width_i][ri]
        _add_quad(faces, a, b, c, d, reverse)


def _add_quad(faces, a: int, b: int, c: int, d: int, reverse: bool = False) -> None:
    if reverse:
        faces.append((a, c, b))
        faces.append((a, d, c))
    else:
        faces.append((a, b, c))
        faces.append((a, c, d))

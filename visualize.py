"""Plotly 3D visualization for formula-based tire cross stacking."""

from __future__ import annotations

import math
from typing import Dict, List

import numpy as np
import plotly.graph_objects as go

from mesh_tire import create_tire_mesh, quality_to_segments, rotation_matrix_axis_angle


def create_stack_figure(
    result: Dict[str, float | int | str | bool],
    render_mode: str = "front_only",
    simplified: bool = False,
    realistic_tire_mesh: bool = True,
    tread_detail: bool = True,
    mesh_quality: str = "low",
    tire_opacity: float = 1.0,
    show_container: bool = True,
    show_edge_tires: bool = True,
) -> go.Figure:
    """Create a Plotly 3D figure from calculation output.

    The layout is a formula-based approximation, not a physical collision
    simulation. Coordinates are in centimeters.
    """

    if realistic_tire_mesh:
        return _create_realistic_stack_figure(
            result,
            render_mode=render_mode,
            tread_detail=tread_detail,
            mesh_quality=mesh_quality,
            tire_opacity=tire_opacity,
            show_container=show_container,
            show_edge_tires=show_edge_tires,
        )

    W_c = float(result["W_c"])
    H_c = float(result["H_c"])
    L_c = float(result["L_c"])
    D = float(result["D"])
    SW = float(result["SW"])
    p = float(result["p"])
    H_pair = float(result["H_pair"])
    first_layer_height = float(result.get("first_layer_height", D))
    nesting_depth = float(result.get("a", 0.0))
    q_L = float(result["q_L"])
    n_width = int(result["n_width"])
    n_width_tilted = int(result.get("n_width_tilted", n_width))
    n_pair = int(result["n_pair"])
    top_single_layer = int(result.get("top_single_layer", 0))
    half_pair_height = float(result.get("half_pair_height", H_pair / 2.0))
    n_L = int(result["n_L"])
    n_edge_H = int(result["n_edge_H"])
    edge_columns = int(result.get("edge_columns", 0))
    edge_left_fit = bool(result.get("edge_left_fit", edge_columns >= 1))
    edge_right_fit = bool(result.get("edge_right_fit", edge_columns >= 2))
    edge_start_z = float(result.get("edge_start_z", 0.0))
    alpha_deg = float(result["alpha_deg"])
    stack_model = str(result.get("stack_model", "report_pairs"))

    if render_mode == "front_only":
        return _create_front_face_figure(
            W_c=W_c,
            H_c=H_c,
            D=D,
            SW=SW,
            p=p,
            H_pair=H_pair,
            n_width=n_width,
            n_pair=n_pair,
            n_edge_H=n_edge_H,
            edge_columns=edge_columns,
            alpha_deg=alpha_deg,
        )

    layer_count = _resolve_layer_count(render_mode, n_L)
    if render_mode == "front_only":
        shown_length = min(L_c, max(1.25 * q_L, D))
    elif render_mode == "sample_layers":
        shown_length = min(L_c, max(3.25 * q_L, D))
    else:
        shown_length = L_c
    fig = go.Figure()
    _add_container_wireframe(fig, shown_length, W_c, H_c)

    tire_index = 1
    for layer in range(layer_count):
        x = layer * q_L + D / 2.0
        for pair_idx in range(n_pair):
            z_base = pair_idx * H_pair
            for row, z, angle, color in (
                ("Row 1", z_base + D / 2.0, alpha_deg, "#1f77b4"),
                ("Row 2", z_base + H_pair / 2.0 + D / 2.0, -alpha_deg, "#ff7f0e"),
            ):
                if z > H_c + D / 2.0:
                    continue
                for width_idx in range(n_width):
                    y = _row_y_position(row, width_idx, D, p)
                    if y > W_c:
                        continue
                    tire_angle = 0.0 if row == "Row 1" and width_idx == 0 else angle
                    hover = _hover_text(tire_index, row, pair_idx + 1, layer + 1, x, y, z)
                    _add_tire(fig, x, y, z, D, SW, tire_angle, hover, simplified, color=color)
                    tire_index += 1

        if n_edge_H > 0:
            for side, y in (("left edge", SW / 2.0), ("right edge", W_c - SW / 2.0)):
                for edge_idx in range(n_edge_H):
                    z = edge_idx * D + D / 2.0
                    hover = _hover_text(tire_index, side, edge_idx + 1, layer + 1, x, y, z)
                    _add_tire(
                        fig,
                        x,
                        y,
                        z,
                        D,
                        SW,
                        0.0,
                        hover,
                        simplified,
                        color="#2ca02c",
                        plane="xz",
                    )
                    tire_index += 1

    x_aspect = max(0.8, min(3.5, shown_length / W_c))
    z_aspect = max(0.8, min(1.4, H_c / W_c))

    if render_mode == "front_only":
        aspect_ratio = {"x": 0.22, "y": 1.0, "z": max(0.9, H_c / W_c)}
        camera = {
            "eye": {"x": 2.65, "y": 0.0, "z": 0.02},
            "up": {"x": 0, "y": 0, "z": 1},
            "projection": {"type": "orthographic"},
        }
    else:
        aspect_ratio = {
            "x": max(0.8, min(4.6, shown_length / W_c)),
            "y": 1.0,
            "z": max(0.8, H_c / W_c),
        }
        camera = {
            "eye": {"x": 1.55, "y": -1.65, "z": 0.9},
            "projection": {"type": "orthographic"},
        }

    if render_mode == "front_only":
        aspect_ratio = {"x": 0.22, "y": 1.0, "z": max(0.9, H_c / W_c)}
        camera = {
            "eye": {"x": 2.65, "y": 0.0, "z": 0.02},
            "up": {"x": 0, "y": 0, "z": 1},
            "projection": {"type": "orthographic"},
        }
    else:
        aspect_ratio = {
            "x": max(0.8, min(4.6, shown_length / W_c)),
            "y": 1.0,
            "z": max(0.8, H_c / W_c),
        }
        camera = {
            "eye": {"x": 1.55, "y": -1.65, "z": 0.9},
            "projection": {"type": "orthographic"},
        }

    fig.update_layout(
        scene={
            "xaxis_title": "Length x (cm)",
            "yaxis_title": "Width y (cm)",
            "zaxis_title": "Height z (cm)",
            "xaxis": {"range": [0, shown_length]},
            "yaxis": {"range": [0, W_c]},
            "zaxis": {"range": [0, H_c]},
            "aspectmode": "manual",
            "aspectratio": {"x": x_aspect, "y": 1.0, "z": z_aspect},
            "camera": {"eye": {"x": 1.45, "y": -1.8, "z": 1.15}},
        },
        margin={"l": 0, "r": 0, "t": 24, "b": 0},
        showlegend=False,
        height=720,
    )
    return fig


def create_full_stack_figure(result: Dict[str, float | int | str | bool]) -> go.Figure:
    """Render all length layers as merged low-poly tire meshes."""

    W_c = float(result["W_c"])
    H_c = float(result["H_c"])
    L_c = float(result["L_c"])
    D = float(result["D"])
    SW = float(result["SW"])
    p = float(result["p"])
    W2 = float(result["W2"])
    H_pair = float(result["H_pair"])
    q_L = float(result["q_L"])
    n_width = int(result["n_width"])
    n_width_tilted = int(result.get("n_width_tilted", n_width))
    n_pair = int(result["n_pair"])
    top_single_layer = int(result.get("top_single_layer", 0))
    half_pair_height = float(result.get("half_pair_height", H_pair / 2.0))
    n_L = int(result["n_L"])
    n_edge_H = int(result["n_edge_H"])
    edge_columns = int(result.get("edge_columns", 0))
    edge_left_fit = bool(result.get("edge_left_fit", edge_columns >= 1))
    edge_right_fit = bool(result.get("edge_right_fit", edge_columns >= 2))
    edge_start_z = float(result.get("edge_start_z", 0.0))
    alpha_deg = float(result["alpha_deg"])
    stack_model = str(result.get("stack_model", "base_layer_pairs"))

    fig = go.Figure()
    _add_container_wireframe(fig, L_c, W_c, H_c)

    line_buckets = {
        "odd_row1": _empty_line_bucket("#1f77b4"),
        "odd_row2": _empty_line_bucket("#1f77b4"),
        "even_row1": _empty_line_bucket("#d97808"),
        "even_row2": _empty_line_bucket("#d97808"),
        "edge": _empty_line_bucket("#159447"),
    }

    visual_half_height = _ellipse_extent_z(D, SW, alpha_deg)
    row2_offset = max(H_pair / 2.0, visual_half_height * 0.78)

    for layer_idx in range(n_L):
        x = layer_idx * q_L + D / 2.0
        for pair_idx in range(n_pair):
            base_z = pair_idx * H_pair + visual_half_height
            level_key = "odd" if pair_idx % 2 == 0 else "even"
            for row, z, angle, key in (
                ("Row 1", base_z, alpha_deg, f"{level_key}_row1"),
                ("Row 2", base_z + row2_offset, -alpha_deg, f"{level_key}_row2"),
            ):
                if z - visual_half_height > H_c:
                    continue
                for width_idx in range(n_width):
                    y = _front_face_y_position(row, width_idx, D, p)
                    tire_angle = 0.0 if row == "Row 1" and width_idx == 0 else angle
                    _extend_cross_outline_bucket(line_buckets[key], x, y, z, D, SW, tire_angle)

        for edge_col in range(edge_columns):
            y = SW / 2.0 if edge_col == 0 else W_c - SW / 2.0
            for edge_idx in range(n_edge_H):
                z = edge_idx * D + D / 2.0
                _extend_edge_outline_bucket(line_buckets["edge"], x, y, z, D, SW)

    for key in ("odd_row1", "odd_row2", "even_row1", "even_row2", "edge"):
        _add_line_bucket(fig, line_buckets[key], _full_hover_text(key))

    fig.update_layout(
        scene={
            "xaxis_title": "Length x (cm)",
            "yaxis_title": "Width y (cm)",
            "zaxis_title": "Height z (cm)",
            "xaxis": {"range": [0, L_c]},
            "yaxis": {"range": [0, W_c]},
            "zaxis": {"range": [0, H_c]},
            "aspectmode": "manual",
            "aspectratio": {"x": 4.6, "y": 1.0, "z": H_c / W_c},
            "camera": {
                "eye": {"x": 1.75, "y": -1.45, "z": 0.72},
                "projection": {"type": "orthographic"},
            },
        },
        margin={"l": 0, "r": 0, "t": 24, "b": 0},
        showlegend=False,
        height=720,
    )
    return fig


def _create_realistic_stack_figure(
    result: Dict[str, float | int | str | bool],
    render_mode: str,
    tread_detail: bool,
    mesh_quality: str,
    tire_opacity: float,
    show_container: bool,
    show_edge_tires: bool,
) -> go.Figure:
    """Render formula-based tire placement using true hollow tire meshes."""

    W_c = float(result["W_c"])
    H_c = float(result["H_c"])
    L_c = float(result["L_c"])
    D = float(result["D"])
    d = float(result["d"])
    SW = float(result["SW"])
    p = float(result["p"])
    H_pair = float(result["H_pair"])
    first_layer_height = float(result.get("first_layer_height", D))
    nesting_depth = float(result.get("a", 0.0))
    q_L = float(result["q_L"])
    n_width = int(result["n_width"])
    n_width_tilted = int(result.get("n_width_tilted", n_width))
    n_pair = int(result["n_pair"])
    top_single_layer = int(result.get("top_single_layer", 0))
    half_pair_height = float(result.get("half_pair_height", H_pair / 2.0))
    n_L = int(result["n_L"])
    n_edge_H = int(result["n_edge_H"])
    edge_columns = int(result.get("edge_columns", 0))
    edge_left_fit = bool(result.get("edge_left_fit", edge_columns >= 1))
    edge_right_fit = bool(result.get("edge_right_fit", edge_columns >= 2))
    edge_start_z = float(result.get("edge_start_z", 0.0))
    alpha_deg = float(result["alpha_deg"])
    stack_model = str(result.get("stack_model", "base_layer_pairs"))

    layer_count = _resolve_layer_count(render_mode, n_L)
    shown_length = L_c if render_mode == "full" else min(L_c, max(layer_count * q_L + D, D * 1.5))
    radial_segments, width_segments = quality_to_segments(mesh_quality)
    if render_mode == "full" and mesh_quality == "high":
        radial_segments, width_segments = quality_to_segments("medium")

    fig = go.Figure()
    if show_container:
        _add_container_wireframe(fig, shown_length, W_c, H_c)

    flat_tire_rotation = np.array(
        [
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )
    left_lean_rotation = rotation_matrix_axis_angle((1, 0, 0), -alpha_deg) @ flat_tire_rotation
    right_lean_rotation = rotation_matrix_axis_angle((1, 0, 0), alpha_deg) @ flat_tire_rotation
    edge_rotation = np.array(
        [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    tire_index = 1
    for layer_idx in range(layer_count):
        x = layer_idx * q_L + D / 2.0
        if stack_model == "base_layer_pairs":
            alpha_rad = math.radians(alpha_deg)
            tilted_floor_center_z = D / 2.0 * abs(math.sin(alpha_rad)) + SW / 2.0 * abs(math.cos(alpha_rad))
            first_row_tilted_z = tilted_floor_center_z - 2.0 * nesting_depth
            tire_index = _add_realistic_row(
                fig,
                tire_index,
                D,
                d,
                SW,
                p,
                n_width,
                x,
                first_row_tilted_z,
                "1층 기준",
                0,
                layer_idx + 1,
                _stack_layer_rotation(1, left_lean_rotation, right_lean_rotation),
                _stack_layer_color(1),
                radial_segments,
                width_segments,
                tread_detail,
                tire_opacity,
                first_tire_flat=True,
                W_c=W_c,
                flat_rotation=flat_tire_rotation,
                flat_z=SW / 2.0 - 2.0 * nesting_depth,
                y_shift=0.0,
            )
            pair_z_start = first_layer_height
            pair_width = n_width_tilted
        else:
            pair_z_start = D / 2.0
            pair_width = n_width

        for pair_idx in range(n_pair):
            z_base = pair_z_start + pair_idx * H_pair
            if stack_model == "base_layer_pairs":
                lower_z = z_base + H_pair / 4.0
                upper_z = z_base + 3.0 * H_pair / 4.0
            else:
                lower_z = z_base
                upper_z = z_base + H_pair / 2.0
            lower_layer_number = 2 * pair_idx + 2 if stack_model == "base_layer_pairs" else 2 * pair_idx + 1
            upper_layer_number = lower_layer_number + 1
            tire_index = _add_realistic_row(
                fig,
                tire_index,
                D,
                d,
                SW,
                p,
                pair_width,
                x,
                lower_z,
                f"{lower_layer_number}층 lower",
                pair_idx + 1,
                layer_idx + 1,
                _stack_layer_rotation(lower_layer_number, left_lean_rotation, right_lean_rotation),
                _stack_layer_color(lower_layer_number),
                radial_segments,
                width_segments,
                tread_detail,
                tire_opacity,
                first_tire_flat=(stack_model == "report_pairs"),
                W_c=W_c,
                flat_rotation=flat_tire_rotation,
                flat_z=SW / 2.0,
                y_shift=_stack_layer_y_shift(lower_layer_number, p),
            )
            tire_index = _add_realistic_row(
                fig,
                tire_index,
                D,
                d,
                SW,
                p,
                pair_width,
                x,
                upper_z,
                f"{upper_layer_number}층 upper",
                pair_idx + 1,
                layer_idx + 1,
                _stack_layer_rotation(upper_layer_number, left_lean_rotation, right_lean_rotation),
                _stack_layer_color(upper_layer_number),
                radial_segments,
                width_segments,
                tread_detail,
                tire_opacity,
                first_tire_flat=False,
                W_c=W_c,
                y_shift=_stack_layer_y_shift(upper_layer_number, p),
            )

        if stack_model == "base_layer_pairs" and top_single_layer:
            top_layer_number = 2 * n_pair + 2
            top_z = first_layer_height + n_pair * H_pair + half_pair_height / 2.0
            tire_index = _add_realistic_row(
                fig,
                tire_index,
                D,
                d,
                SW,
                p,
                n_width,
                x,
                top_z,
                f"{top_layer_number}층 top",
                n_pair + 1,
                layer_idx + 1,
                _stack_layer_rotation(top_layer_number, left_lean_rotation, right_lean_rotation),
                _stack_layer_color(top_layer_number),
                radial_segments,
                width_segments,
                tread_detail,
                tire_opacity,
                first_tire_flat=False,
                W_c=W_c,
                y_shift=_stack_layer_y_shift(top_layer_number, p),
            )

        if show_edge_tires and n_edge_H > 0 and edge_columns > 0:
            edge_specs = []
            if edge_left_fit:
                edge_specs.append((SW / 2.0, "left edge"))
            if edge_right_fit:
                edge_specs.append((W_c - SW / 2.0, "right edge"))
            for y, edge_label in edge_specs:
                edge_color = "rgb(64,210,115)"
                for edge_idx in range(n_edge_H):
                    z = edge_start_z + edge_idx * D + D / 2.0 - 3.0 * nesting_depth
                    hover = _hover_text(tire_index, edge_label, edge_idx + 1, layer_idx + 1, x, y, z)
                    fig.add_trace(
                        create_tire_mesh(
                            D=D,
                            d=d,
                            SW=SW,
                            center=(x, y, z),
                            rotation_matrix=edge_rotation,
                            radial_segments=radial_segments,
                            width_segments=width_segments,
                            add_tread=tread_detail,
                            tire_color=edge_color,
                            opacity=tire_opacity,
                            hover_text=hover,
                        )
                    )
                    tire_index += 1

    if render_mode == "front_only":
        aspect_ratio = {"x": 0.22, "y": 1.0, "z": max(0.9, H_c / W_c)}
        camera = {
            "eye": {"x": 2.65, "y": 0.0, "z": 0.02},
            "up": {"x": 0, "y": 0, "z": 1},
            "projection": {"type": "orthographic"},
        }
    else:
        aspect_ratio = {
            "x": max(0.8, min(4.6, shown_length / W_c)),
            "y": 1.0,
            "z": max(0.8, H_c / W_c),
        }
        camera = {
            "eye": {"x": 1.55, "y": -1.65, "z": 0.9},
            "projection": {"type": "orthographic"},
        }

    fig.update_layout(
        scene={
            "xaxis_title": "Length x (cm)",
            "yaxis_title": "Width y (cm)",
            "zaxis_title": "Height z (cm)",
            "xaxis": {"range": [0, shown_length]},
            "yaxis": {"range": [0, W_c]},
            "zaxis": {"range": [0, H_c]},
            "aspectmode": "manual",
            "aspectratio": aspect_ratio,
            "camera": camera,
        },
        margin={"l": 0, "r": 0, "t": 24, "b": 0},
        showlegend=False,
        height=720,
    )
    return fig


def _add_realistic_row(
    fig: go.Figure,
    tire_index: int,
    D: float,
    d: float,
    SW: float,
    p: float,
    n_width: int,
    x: float,
    z: float,
    row: str,
    pair_index: int,
    layer_index: int,
    row_rotation: np.ndarray,
    tire_color: str,
    radial_segments: int,
    width_segments: int,
    tread_detail: bool,
    tire_opacity: float,
    first_tire_flat: bool,
    W_c: float,
    flat_rotation: np.ndarray | None = None,
    flat_z: float | None = None,
    y_shift: float = 0.0,
) -> int:
    for width_idx in range(n_width):
        row_position = "Row 2" if row == "Row 2" or "upper" in row else "Row 1"
        y = _row_y_position(row_position, width_idx, D, p) + y_shift
        if y < -D / 2.0 or y > W_c + D / 2.0:
            continue
        is_flat_first = first_tire_flat and width_idx == 0
        rotation = flat_rotation if is_flat_first and flat_rotation is not None else row_rotation
        tire_z = flat_z if is_flat_first and flat_z is not None else z
        hover = _hover_text(tire_index, row, pair_index, layer_index, x, y, tire_z)
        hover += f"<br>width index: {width_idx + 1}"
        fig.add_trace(
            create_tire_mesh(
                D=D,
                d=d,
                SW=SW,
                center=(x, y, tire_z),
                rotation_matrix=rotation,
                radial_segments=radial_segments,
                width_segments=width_segments,
                add_tread=tread_detail,
                tire_color=tire_color,
                opacity=tire_opacity,
                hover_text=hover,
            )
        )
        tire_index += 1
    return tire_index


def _stack_layer_color(layer_number: int) -> str:
    """Return the requested layer color: first gray, even orange, odd blue."""

    if layer_number == 1:
        return "rgb(190,196,204)"
    if layer_number % 2 == 0:
        return "rgb(255,166,42)"
    return "rgb(66,165,245)"


def _stack_layer_rotation(
    layer_number: int, left_lean_rotation: np.ndarray, right_lean_rotation: np.ndarray
) -> np.ndarray:
    """Odd layers lean left and even layers lean right in the current camera view."""

    if layer_number % 2 == 0:
        return right_lean_rotation
    return left_lean_rotation


def _stack_layer_y_shift(layer_number: int, p: float) -> float:
    """Move even layers by a half target pitch in the container width direction."""

    if layer_number % 2 == 0:
        return p / 2.0
    return 0.0


def _create_front_face_figure(
    W_c: float,
    H_c: float,
    D: float,
    SW: float,
    p: float,
    H_pair: float,
    n_width: int,
    n_pair: int,
    n_edge_H: int,
    edge_columns: int,
    alpha_deg: float,
) -> go.Figure:
    """Render one width-height face so the cross pattern is legible."""

    fig = go.Figure()
    face_depth = max(D + SW, 80.0)
    x_face = face_depth / 2.0
    _add_container_wireframe(fig, face_depth, W_c, H_c)

    tire_index = 1
    visual_half_height = _ellipse_extent_z(D, SW, alpha_deg)
    row2_offset = max(H_pair / 2.0, visual_half_height * 0.78)

    for pair_idx in range(n_pair):
        base_z = pair_idx * H_pair + visual_half_height
        pair_color = "#1f77b4" if pair_idx % 2 == 0 else "#ff7f0e"
        row_specs = (
            ("Row 1", base_z, alpha_deg, pair_color),
            ("Row 2", base_z + row2_offset, -alpha_deg, pair_color),
        )
        for row, z, angle, color in row_specs:
            if z - visual_half_height > H_c:
                continue
            for width_idx in range(n_width):
                y = _front_face_y_position(row, width_idx, D, p)
                tire_angle = 0.0 if row == "Row 1" and width_idx == 0 else angle
                hover = _hover_text(tire_index, row, pair_idx + 1, 1, x_face, y, z)
                _add_cross_tire_outline(fig, x_face, y, z, D, SW, tire_angle, hover, color)
                tire_index += 1

    if n_edge_H > 0 and edge_columns > 0:
        for edge_col in range(edge_columns):
            y = SW / 2.0 if edge_col == 0 else W_c - SW / 2.0
            for edge_idx in range(n_edge_H):
                z = edge_idx * D + D / 2.0
                edge_label = "left edge" if edge_col == 0 else "right edge"
                hover = _hover_text(tire_index, edge_label, edge_idx + 1, 1, x_face, y, z)
                edge_color = "#159447" if edge_col == 0 else "#8a4bd9"
                _add_edge_tire_outline(fig, x_face, y, z, D, SW, hover, edge_color)
                tire_index += 1

    fig.update_layout(
        scene={
            "xaxis_title": "Slice depth x (cm)",
            "yaxis_title": "Width y (cm)",
            "zaxis_title": "Height z (cm)",
            "xaxis": {"range": [0, face_depth], "showticklabels": False},
            "yaxis": {"range": [0, W_c]},
            "zaxis": {"range": [0, H_c]},
            "aspectmode": "manual",
            "aspectratio": {"x": 0.5, "y": 1.0, "z": H_c / W_c},
            "camera": {
                "eye": {"x": 1.15, "y": -0.85, "z": 0.38},
                "projection": {"type": "orthographic"},
            },
        },
        margin={"l": 0, "r": 0, "t": 24, "b": 0},
        showlegend=False,
        height=720,
    )
    return fig


def _resolve_layer_count(render_mode: str, n_L: int) -> int:
    if render_mode == "front_only":
        return min(1, n_L)
    if render_mode == "sample_layers":
        return min(3, n_L)
    if render_mode == "full":
        return n_L
    raise ValueError("render mode는 front_only, sample_layers, full 중 하나여야 합니다.")


def _row_y_position(row: str, width_idx: int, D: float, p: float) -> float:
    if row == "Row 1":
        return D / 2.0 + width_idx * p
    return p / 2.0 + width_idx * p


def _front_face_y_position(row: str, width_idx: int, D: float, p: float) -> float:
    if row == "Row 1":
        return D / 2.0 + width_idx * p
    return p / 2.0 + width_idx * p


def _ellipse_extent_z(D: float, SW: float, angle_deg: float) -> float:
    major = D / 2.0
    minor = max(SW * 0.45, D * 0.14)
    angle = math.radians(abs(angle_deg))
    return major * math.sin(angle) + minor * math.cos(angle)


def _add_faceted_cross_tire(
    fig: go.Figure,
    x0: float,
    y0: float,
    z0: float,
    D: float,
    SW: float,
    angle_deg: float,
    hover: str,
    color: str,
) -> None:
    vertices, faces = _cross_tire_mesh((x0, y0, z0), D, SW, angle_deg)
    _add_mesh_tire(fig, vertices, faces, hover, color)


def _add_cross_tire_outline(
    fig: go.Figure,
    x0: float,
    y0: float,
    z0: float,
    D: float,
    SW: float,
    angle_deg: float,
    hover: str,
    accent: str,
) -> None:
    outer_major = D * 0.39
    outer_minor = D * 0.26
    inner_major = outer_major * 0.58
    inner_minor = outer_minor * 0.56
    depth = max(SW * 0.60, D * 0.12)
    _add_ellipse_outline(fig, x0, y0, z0, outer_major, outer_minor, 0.0, angle_deg, accent, 4, hover)
    _add_ellipse_outline(fig, x0, y0, z0, inner_major, inner_minor, 0.0, angle_deg, accent, 3, hover)
    _add_ellipse_outline(fig, x0, y0, z0, outer_major * 0.82, outer_minor * 0.82, 0.0, angle_deg, "#141414", 1, hover, opacity=0.3)


def _add_edge_tire_outline(
    fig: go.Figure,
    x0: float,
    y0: float,
    z0: float,
    D: float,
    SW: float,
    hover: str,
    color: str = "#159447",
) -> None:
    radius = D * 0.40
    inner_radius = radius * 0.58
    depth = min(max(SW * 0.90, D * 0.14), SW)
    _add_xz_circle_outline(fig, x0, y0, z0, radius, 0.0, color, 4, hover)
    _add_xz_circle_outline(fig, x0, y0, z0, inner_radius, 0.0, color, 3, hover)
    _add_xz_circle_outline(fig, x0, y0, z0, radius * 0.82, 0.0, "#141414", 1, hover, opacity=0.35)


def _add_face_centerline(
    fig: go.Figure,
    x0: float,
    y0: float,
    z0: float,
    major: float,
    minor: float,
    angle_deg: float,
    color: str,
    hover: str,
) -> None:
    t = np.linspace(0, 2 * np.pi, 96)
    x, y, z = _tilted_ellipse_points(t, x0, y0, z0, major, minor, angle_deg)
    fig.add_trace(
        go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode="lines",
            line={"color": color, "width": 2},
            opacity=0.35,
            hovertemplate=hover + "<extra></extra>",
        )
    )


def _add_faceted_edge_tire(
    fig: go.Figure,
    x0: float,
    y0: float,
    z0: float,
    D: float,
    SW: float,
    hover: str,
) -> None:
    vertices, faces = _edge_tire_mesh((x0, y0, z0), D, SW)
    _add_mesh_tire(fig, vertices, faces, hover, "#20b96b")


def _add_edge_side_tire(
    fig: go.Figure,
    x0: float,
    y0: float,
    z0: float,
    D: float,
    SW: float,
    hover: str,
) -> None:
    vertices, faces = _standing_tire_side_mesh((x0, y0, z0), D, SW)
    _add_mesh_tire(fig, vertices, faces, hover, "#20b96b")


def _cross_tire_mesh(
    center: tuple[float, float, float],
    D: float,
    SW: float,
    angle_deg: float,
) -> tuple[np.ndarray, List[tuple[int, int, int]]]:
    outer_major = D * 0.36
    outer_minor = D * 0.24
    inner_major = outer_major * 0.66
    inner_minor = outer_minor * 0.62
    depth = max(SW * 0.68, D * 0.14)
    return _annular_prism_vertices(
        center=center,
        outer=(outer_major, outer_minor),
        inner=(inner_major, inner_minor),
        depth=depth,
        angle_deg=angle_deg,
        plane="yz",
        segments=20,
    )


def _edge_tire_mesh(
    center: tuple[float, float, float],
    D: float,
    SW: float,
) -> tuple[np.ndarray, List[tuple[int, int, int]]]:
    outer_major = D * 0.39
    outer_minor = D * 0.39
    inner_major = outer_major * 0.60
    inner_minor = outer_minor * 0.60
    depth = min(max(SW * 0.92, D * 0.18), SW)
    return _annular_prism_vertices(
        center=center,
        outer=(outer_major, outer_minor),
        inner=(inner_major, inner_minor),
        depth=depth,
        angle_deg=0.0,
        plane="xz",
        segments=20,
    )


def _add_mesh_tire(
    fig: go.Figure,
    vertices: np.ndarray,
    faces: List[tuple[int, int, int]],
    hover: str,
    color: str,
) -> None:
    fig.add_trace(
        go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            i=[face[0] for face in faces],
            j=[face[1] for face in faces],
            k=[face[2] for face in faces],
            color=color,
            opacity=0.96,
            flatshading=True,
            lighting={
                "ambient": 0.45,
                "diffuse": 0.65,
                "specular": 0.22,
                "roughness": 0.8,
                "fresnel": 0.15,
            },
            lightposition={"x": 90, "y": -160, "z": 260},
            hovertemplate=hover + "<extra></extra>",
        )
    )


def _empty_mesh_bucket(color: str) -> dict:
    return {"vertices": [], "faces": [], "color": color}


def _extend_mesh_bucket(
    bucket: dict,
    vertices: np.ndarray,
    faces: List[tuple[int, int, int]],
) -> None:
    offset = len(bucket["vertices"])
    bucket["vertices"].extend(tuple(row) for row in vertices)
    bucket["faces"].extend((a + offset, b + offset, c + offset) for a, b, c in faces)


def _add_bucket_mesh(fig: go.Figure, bucket: dict, hover: str) -> None:
    if not bucket["vertices"]:
        return
    vertices = np.array(bucket["vertices"])
    faces = bucket["faces"]
    _add_mesh_tire(fig, vertices, faces, hover, bucket["color"])


def _full_hover_text(key: str) -> str:
    labels = {
        "odd_row1": "odd pair Row 1",
        "odd_row2": "odd pair Row 2",
        "even_row1": "even pair Row 1",
        "even_row2": "even pair Row 2",
        "edge": "standing edge stack",
    }
    return labels.get(key, key)


def _empty_line_bucket(color: str) -> dict:
    return {"x": [], "y": [], "z": [], "color": color}


def _extend_cross_outline_bucket(
    bucket: dict,
    x0: float,
    y0: float,
    z0: float,
    D: float,
    SW: float,
    angle_deg: float,
) -> None:
    outer_major = D * 0.35
    outer_minor = D * 0.23
    inner_major = outer_major * 0.58
    inner_minor = outer_minor * 0.56
    depth = max(SW * 0.55, D * 0.10)
    _append_ellipse_outline(bucket, x0, y0, z0, outer_major, outer_minor, depth, angle_deg)
    _append_ellipse_outline(bucket, x0, y0, z0, inner_major, inner_minor, depth, angle_deg)


def _extend_edge_outline_bucket(bucket: dict, x0: float, y0: float, z0: float, D: float, SW: float) -> None:
    radius = D * 0.38
    inner_radius = radius * 0.58
    depth = min(max(SW * 0.88, D * 0.12), SW)
    _append_xz_circle_outline(bucket, x0, y0, z0, radius, depth)
    _append_xz_circle_outline(bucket, x0, y0, z0, inner_radius, depth)


def _append_ellipse_outline(
    bucket: dict,
    x0: float,
    y0: float,
    z0: float,
    major: float,
    minor: float,
    depth: float,
    angle_deg: float,
) -> None:
    t = np.linspace(0, 2 * np.pi, 48)
    offsets = [-depth / 2.0, depth / 2.0]
    for offset in offsets:
        x, y, z = _tilted_ellipse_points(t, x0 + offset, y0, z0, major, minor, angle_deg)
        _append_segment(bucket["x"], bucket["y"], bucket["z"], x, y, z)


def _append_xz_circle_outline(bucket: dict, x0: float, y0: float, z0: float, radius: float, depth: float) -> None:
    t = np.linspace(0, 2 * np.pi, 48)
    offsets = [-depth / 2.0, depth / 2.0]
    for offset in offsets:
        x = x0 + radius * np.cos(t)
        y = np.full_like(t, y0 + offset)
        z = z0 + radius * np.sin(t)
        _append_segment(bucket["x"], bucket["y"], bucket["z"], x, y, z)


def _add_line_bucket(fig: go.Figure, bucket: dict, hover: str) -> None:
    if not bucket["x"]:
        return
    fig.add_trace(
        go.Scatter3d(
            x=bucket["x"],
            y=bucket["y"],
            z=bucket["z"],
            mode="lines",
            line={"color": bucket["color"], "width": 2},
            opacity=0.55,
            hovertemplate=hover + "<extra></extra>",
        )
    )


def _elliptical_torus_mesh(
    center: tuple[float, float, float],
    radius_y: float,
    radius_z: float,
    tube_x: float,
    tube_radial: float,
    angle_deg: float,
    plane: str,
    u_segments: int,
    v_segments: int,
) -> tuple[np.ndarray, List[tuple[int, int, int]]]:
    """Build a rounded low-poly tire surface around an elliptical centerline."""

    cx, cy, cz = center
    angle = math.radians(angle_deg)
    vertices: List[tuple[float, float, float]] = []
    for i in range(u_segments):
        u = 2.0 * math.pi * i / u_segments
        for j in range(v_segments):
            v = 2.0 * math.pi * j / v_segments
            radial = tube_radial * math.cos(v)
            depth = tube_x * math.sin(v)

            if plane == "yz":
                local_y = (radius_y + radial) * math.cos(u)
                local_z = (radius_z + radial) * math.sin(u)
                y = cy + local_y * math.cos(angle) - local_z * math.sin(angle)
                z = cz + local_y * math.sin(angle) + local_z * math.cos(angle)
                x = cx + depth
            elif plane == "xz":
                x = cx + (radius_y + radial) * math.cos(u)
                y = cy + depth
                z = cz + (radius_z + radial) * math.sin(u)
            else:
                raise ValueError("plane must be yz or xz")
            vertices.append((x, y, z))

    faces: List[tuple[int, int, int]] = []
    for i in range(u_segments):
        ni = (i + 1) % u_segments
        for j in range(v_segments):
            nj = (j + 1) % v_segments
            a = i * v_segments + j
            b = ni * v_segments + j
            c = ni * v_segments + nj
            d = i * v_segments + nj
            _add_quad(faces, a, b, c, d)

    return np.array(vertices), faces


def _standing_tire_side_mesh(
    center: tuple[float, float, float],
    D: float,
    SW: float,
) -> tuple[np.ndarray, List[tuple[int, int, int]]]:
    """Build the side view of an upright tire constrained to the gap width."""

    cx, cy, cz = center
    half_w = SW / 2.0
    half_h = D * 0.43
    half_d = max(SW * 0.62, D * 0.12)
    bevel = min(SW * 0.18, D * 0.055)
    y_levels = [-half_w, -half_w + bevel, half_w - bevel, half_w]
    z_levels = [-half_h, -half_h + bevel, half_h - bevel, half_h]
    x_levels = [-half_d, half_d]

    vertices: List[tuple[float, float, float]] = []
    for x in x_levels:
        for y in y_levels:
            for z in z_levels:
                vertices.append((cx + x, cy + y, cz + z))

    def idx(x_i: int, y_i: int, z_i: int) -> int:
        return x_i * len(y_levels) * len(z_levels) + y_i * len(z_levels) + z_i

    faces: List[tuple[int, int, int]] = []
    for x_i in (0, 1):
        _add_grid_face(faces, idx, x_i=x_i, y_count=len(y_levels), z_count=len(z_levels))
    for y_i in (0, len(y_levels) - 1):
        _add_side_grid_face(faces, idx, fixed_axis="y", fixed_i=y_i, x_count=2, other_count=len(z_levels))
    for z_i in (0, len(z_levels) - 1):
        _add_side_grid_face(faces, idx, fixed_axis="z", fixed_i=z_i, x_count=2, other_count=len(y_levels))
    return np.array(vertices), faces


def _add_grid_face(faces, idx, x_i: int, y_count: int, z_count: int) -> None:
    for y_i in range(y_count - 1):
        for z_i in range(z_count - 1):
            _add_quad(
                faces,
                idx(x_i, y_i, z_i),
                idx(x_i, y_i + 1, z_i),
                idx(x_i, y_i + 1, z_i + 1),
                idx(x_i, y_i, z_i + 1),
            )


def _add_side_grid_face(
    faces,
    idx,
    fixed_axis: str,
    fixed_i: int,
    x_count: int,
    other_count: int,
) -> None:
    for x_i in range(x_count - 1):
        for other_i in range(other_count - 1):
            if fixed_axis == "y":
                _add_quad(
                    faces,
                    idx(x_i, fixed_i, other_i),
                    idx(x_i + 1, fixed_i, other_i),
                    idx(x_i + 1, fixed_i, other_i + 1),
                    idx(x_i, fixed_i, other_i + 1),
                )
            else:
                _add_quad(
                    faces,
                    idx(x_i, other_i, fixed_i),
                    idx(x_i + 1, other_i, fixed_i),
                    idx(x_i + 1, other_i + 1, fixed_i),
                    idx(x_i, other_i + 1, fixed_i),
                )


def _annular_prism_vertices(
    center: tuple[float, float, float],
    outer: tuple[float, float],
    inner: tuple[float, float],
    depth: float,
    angle_deg: float,
    plane: str,
    segments: int,
) -> tuple[np.ndarray, List[tuple[int, int, int]]]:
    """Build a low-poly extruded ring that reads like a stacked tire."""

    cx, cy, cz = center
    angle = math.radians(angle_deg)
    points: List[tuple[float, float, float]] = []
    half_depth = depth / 2.0
    offsets = (-half_depth, half_depth)
    for offset in offsets:
        for radius_y, radius_z in (outer, inner):
            for idx in range(segments):
                t = 2 * np.pi * idx / segments
                local_a = radius_y * math.cos(t)
                local_b = radius_z * math.sin(t)
                if plane == "yz":
                    y = cy + local_a * math.cos(angle) - local_b * math.sin(angle)
                    z = cz + local_a * math.sin(angle) + local_b * math.cos(angle)
                    x = cx + offset
                elif plane == "xz":
                    x = cx + local_a
                    z = cz + local_b
                    y = cy + offset
                else:
                    raise ValueError("plane must be yz or xz")
                points.append((x, y, z))

    front_outer = 0
    front_inner = segments
    back_outer = segments * 2
    back_inner = segments * 3
    faces: List[tuple[int, int, int]] = []
    for idx in range(segments):
        nxt = (idx + 1) % segments

        fo0, fo1 = front_outer + idx, front_outer + nxt
        fi0, fi1 = front_inner + idx, front_inner + nxt
        bo0, bo1 = back_outer + idx, back_outer + nxt
        bi0, bi1 = back_inner + idx, back_inner + nxt

        _add_quad(faces, fo0, fo1, bo1, bo0)
        _add_quad(faces, fi1, fi0, bi0, bi1)
        _add_quad(faces, fo1, fo0, fi0, fi1)
        _add_quad(faces, bo0, bo1, bi1, bi0)

    return np.array(points), faces


def _add_quad(faces: List[tuple[int, int, int]], a: int, b: int, c: int, d: int) -> None:
    faces.append((a, b, c))
    faces.append((a, c, d))


def _tilted_ellipse_points(
    t: np.ndarray,
    x0: float,
    y0: float,
    z0: float,
    major: float,
    minor: float,
    angle_deg: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    angle = math.radians(angle_deg)
    local_major = major * np.cos(t)
    local_minor = minor * np.sin(t)
    y = y0 + local_major * math.cos(angle) - local_minor * math.sin(angle)
    z = z0 + local_major * math.sin(angle) + local_minor * math.cos(angle)
    x = np.full_like(t, x0)
    return x, y, z


def _add_ellipse_outline(
    fig: go.Figure,
    x0: float,
    y0: float,
    z0: float,
    major: float,
    minor: float,
    depth: float,
    angle_deg: float,
    color: str,
    width: int,
    hover: str,
    opacity: float = 0.9,
) -> None:
    t = np.linspace(0, 2 * np.pi, 96)
    xs: List[float | None] = []
    ys: List[float | None] = []
    zs: List[float | None] = []
    offsets = [-depth / 2.0, depth / 2.0] if depth > 0 else [0.0]
    for offset in offsets:
        x, y, z = _tilted_ellipse_points(t, x0 + offset, y0, z0, major, minor, angle_deg)
        _append_segment(xs, ys, zs, x, y, z)
    if depth > 0:
        for point in np.linspace(0, 2 * np.pi, 8, endpoint=False):
            coords = []
            for offset in offsets:
                x, y, z = _tilted_ellipse_points(np.array([point]), x0 + offset, y0, z0, major, minor, angle_deg)
                coords.append((float(x[0]), float(y[0]), float(z[0])))
            _append_segment(
                xs,
                ys,
                zs,
                np.array([coords[0][0], coords[1][0]]),
                np.array([coords[0][1], coords[1][1]]),
                np.array([coords[0][2], coords[1][2]]),
            )
    fig.add_trace(
        go.Scatter3d(
            x=xs,
            y=ys,
            z=zs,
            mode="lines",
            line={"color": color, "width": width},
            opacity=opacity,
            hovertemplate=hover + "<extra></extra>",
        )
    )


def _add_xz_circle_outline(
    fig: go.Figure,
    x0: float,
    y0: float,
    z0: float,
    radius: float,
    depth: float,
    color: str,
    width: int,
    hover: str,
    opacity: float = 0.9,
) -> None:
    t = np.linspace(0, 2 * np.pi, 96)
    xs: List[float | None] = []
    ys: List[float | None] = []
    zs: List[float | None] = []
    offsets = [-depth / 2.0, depth / 2.0] if depth > 0 else [0.0]
    for offset in offsets:
        x = x0 + radius * np.cos(t)
        y = np.full_like(t, y0 + offset)
        z = z0 + radius * np.sin(t)
        _append_segment(xs, ys, zs, x, y, z)
    if depth > 0:
        for point in np.linspace(0, 2 * np.pi, 8, endpoint=False):
            coords = []
            for offset in offsets:
                coords.append((x0 + radius * math.cos(point), y0 + offset, z0 + radius * math.sin(point)))
            _append_segment(
                xs,
                ys,
                zs,
                np.array([coords[0][0], coords[1][0]]),
                np.array([coords[0][1], coords[1][1]]),
                np.array([coords[0][2], coords[1][2]]),
            )
    fig.add_trace(
        go.Scatter3d(
            x=xs,
            y=ys,
            z=zs,
            mode="lines",
            line={"color": color, "width": width},
            opacity=opacity,
            hovertemplate=hover + "<extra></extra>",
        )
    )


def _add_container_wireframe(fig: go.Figure, L_c: float, W_c: float, H_c: float) -> None:
    vertices = {
        "000": (0, 0, 0),
        "100": (L_c, 0, 0),
        "010": (0, W_c, 0),
        "110": (L_c, W_c, 0),
        "001": (0, 0, H_c),
        "101": (L_c, 0, H_c),
        "011": (0, W_c, H_c),
        "111": (L_c, W_c, H_c),
    }
    edges = [
        ("000", "100"),
        ("010", "110"),
        ("001", "101"),
        ("011", "111"),
        ("000", "010"),
        ("100", "110"),
        ("001", "011"),
        ("101", "111"),
        ("000", "001"),
        ("100", "101"),
        ("010", "011"),
        ("110", "111"),
    ]

    xs: List[float | None] = []
    ys: List[float | None] = []
    zs: List[float | None] = []
    for start, end in edges:
        for point in (vertices[start], vertices[end]):
            xs.append(point[0])
            ys.append(point[1])
            zs.append(point[2])
        xs.append(None)
        ys.append(None)
        zs.append(None)

    fig.add_trace(
        go.Scatter3d(
            x=xs,
            y=ys,
            z=zs,
            mode="lines",
            line={"color": "rgba(60,60,60,0.55)", "width": 3},
            hoverinfo="skip",
        )
    )


def _add_tire(
    fig: go.Figure,
    x0: float,
    y0: float,
    z0: float,
    D: float,
    SW: float,
    angle_deg: float,
    hover: str,
    simplified: bool,
    color: str = "#1f77b4",
    plane: str = "yz",
) -> None:
    if simplified:
        _add_wire_tire(fig, x0, y0, z0, D, angle_deg, hover, color, plane=plane)
        return

    _add_wire_tire(fig, x0, y0, z0, D, angle_deg, hover, color, plane=plane, thick=True, SW=SW)


def _add_wire_tire(
    fig: go.Figure,
    x0: float,
    y0: float,
    z0: float,
    D: float,
    angle_deg: float,
    hover: str,
    color: str,
    plane: str = "yz",
    thick: bool = False,
    SW: float = 0.0,
) -> None:
    t = np.linspace(0, 2 * np.pi, 72)
    radius = D / 2.0
    inner_radius = radius * 0.55
    offsets = [-SW / 2.0, SW / 2.0] if thick else [0.0]
    xs: List[float | None] = []
    ys: List[float | None] = []
    zs: List[float | None] = []

    for offset in offsets:
        x, y, z = _ring_points(t, radius, x0, y0, z0, angle_deg, plane, offset)
        _append_segment(xs, ys, zs, x, y, z)
        x, y, z = _ring_points(t, inner_radius, x0, y0, z0, angle_deg, plane, offset)
        _append_segment(xs, ys, zs, x, y, z)

    if thick and len(offsets) == 2:
        connector_angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        for a in connector_angles:
            coords = []
            for offset in offsets:
                x, y, z = _ring_points(np.array([a]), radius, x0, y0, z0, angle_deg, plane, offset)
                coords.append((float(x[0]), float(y[0]), float(z[0])))
            _append_segment(
                xs,
                ys,
                zs,
                np.array([coords[0][0], coords[1][0]]),
                np.array([coords[0][1], coords[1][1]]),
                np.array([coords[0][2], coords[1][2]]),
            )

    fig.add_trace(
        go.Scatter3d(
            x=xs,
            y=ys,
            z=zs,
            mode="lines",
            line={"color": color, "width": 3},
            opacity=0.85,
            hovertemplate=hover + "<extra></extra>",
        )
    )


def _append_segment(
    xs: List[float | None],
    ys: List[float | None],
    zs: List[float | None],
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
) -> None:
    xs.extend(float(v) for v in x)
    ys.extend(float(v) for v in y)
    zs.extend(float(v) for v in z)
    xs.append(None)
    ys.append(None)
    zs.append(None)


def _ring_points(
    t: np.ndarray,
    radius: float,
    x0: float,
    y0: float,
    z0: float,
    angle_deg: float,
    plane: str,
    offset: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    angle = math.radians(angle_deg)
    if plane == "yz":
        x = np.full_like(t, offset)
        y = radius * np.cos(t)
        z = radius * np.sin(t)
        x_rot = x * math.cos(angle) - y * math.sin(angle)
        y_rot = x * math.sin(angle) + y * math.cos(angle)
        return x_rot + x0, y_rot + y0, z + z0
    if plane == "xz":
        x = radius * np.cos(t) + x0
        z = radius * np.sin(t) + z0
        y = np.full_like(t, y0 + offset)
        return x, y, z
    raise ValueError("plane must be yz or xz")


def _hover_text(
    tire_index: int,
    row: str,
    pair_index: int,
    length_layer: int,
    x: float,
    y: float,
    z: float,
) -> str:
    return (
        f"tire index: {tire_index}<br>"
        f"row: {row}<br>"
        f"pair index: {pair_index}<br>"
        f"length layer: {length_layer}<br>"
        f"x: {x:.1f} cm<br>y: {y:.1f} cm<br>z: {z:.1f} cm"
    )

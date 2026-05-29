"""Calculation helpers for 40FT tire cross-stack loading.

All dimensions returned by this module are in centimeters unless a key name
explicitly says otherwise.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Dict, Literal


DEFAULT_W_C = 235.0
DEFAULT_H_C = 270.0
DEFAULT_L_C = 1203.0

DEFAULT_BASE_TIRE = "215/55R17"
DEFAULT_D_BASE = 66.8
DEFAULT_P_BASE = 45.0
DEFAULT_C_BASE = 30.0

LMode = Literal["base_45", "scaled_p", "manual"]
PairMode = Literal["strict", "tolerance"]
StackModel = Literal["base_layer_pairs", "report_pairs"]


@dataclass(frozen=True)
class TireSpec:
    """Parsed tire specification and derived dimensions."""

    spec: str
    SW_mm: int
    aspect: int
    rim_inch: int
    SW: float
    d: float
    D: float


def parse_tire_spec(spec: str) -> TireSpec:
    """Parse a tire spec such as ``205/55R16`` and derive SW, d, and D.

    SW is converted from millimeters to centimeters. Rim diameter ``d`` and
    outer diameter ``D`` are returned in centimeters.
    """

    match = re.fullmatch(r"\s*(\d{3})\s*/\s*(\d{2})\s*[Rr]\s*(\d{2})\s*", spec)
    if not match:
        raise ValueError("타이어 규격은 205/55R16 형식으로 입력해야 합니다.")

    sw_mm = int(match.group(1))
    aspect = int(match.group(2))
    rim_inch = int(match.group(3))

    if sw_mm <= 0 or aspect <= 0 or rim_inch <= 0:
        raise ValueError("타이어 폭, 편평비, 림 인치는 모두 양수여야 합니다.")

    sw = sw_mm / 10.0
    d = rim_inch * 2.54
    D = d + 2.0 * sw * (aspect / 100.0)

    return TireSpec(
        spec=f"{sw_mm}/{aspect}R{rim_inch}",
        SW_mm=sw_mm,
        aspect=aspect,
        rim_inch=rim_inch,
        SW=sw,
        d=d,
        D=D,
    )


def resolve_l_value(l_mode: LMode, p: float, manual_l: float) -> float:
    """Resolve the height-formula repeat length L from the selected mode."""

    if l_mode == "base_45":
        return 45.0
    if l_mode == "scaled_p":
        return p
    if l_mode == "manual":
        if manual_l <= 0:
            raise ValueError("manual L은 양수여야 합니다.")
        return manual_l
    raise ValueError(f"지원하지 않는 L mode입니다: {l_mode}")


def _layer_y_shift(layer_number: int, p: float) -> float:
    return p / 2.0 if layer_number % 2 == 0 else 0.0


def calculate_side_edge_fit(
    D: float,
    SW: float,
    p: float,
    W_c: float,
    n_width_first: int,
    n_width_tilted: int,
    n_pair: int,
    alpha_deg: float,
    stack_model: StackModel,
) -> Dict[str, float | int | bool]:
    """Estimate true left/right side strips available for standing edge tires.

    This mirrors the layer placement used by the 3D view. A side column is
    allowed only when the visible free strip is at least the tire width SW.
    """

    lean_half_width = (D / 2.0) * math.cos(math.radians(abs(alpha_deg))) + (SW / 2.0) * math.sin(
        math.radians(abs(alpha_deg))
    )
    flat_half_width = D / 2.0
    y_min = math.inf
    y_max = -math.inf

    side_start_z = SW if stack_model == "base_layer_pairs" else 0.0

    def include(center_y: float, half_width: float, center_z: float, radius_z: float) -> None:
        nonlocal y_min, y_max
        if center_z + radius_z <= side_start_z:
            return
        y_min = min(y_min, center_y - half_width)
        y_max = max(y_max, center_y + half_width)

    if stack_model == "base_layer_pairs":
        for idx in range(n_width_first):
            if idx == 0:
                include(D / 2.0 + idx * p, flat_half_width, SW / 2.0, SW / 2.0)
            else:
                include(D / 2.0 + idx * p, lean_half_width, D / 2.0, D / 2.0)
        for pair_idx in range(n_pair):
            lower_layer = 2 * pair_idx + 2
            upper_layer = lower_layer + 1
            for idx in range(n_width_tilted):
                include(D / 2.0 + idx * p + _layer_y_shift(lower_layer, p), lean_half_width, D, D / 2.0)
                include(p / 2.0 + idx * p + _layer_y_shift(upper_layer, p), lean_half_width, D, D / 2.0)
    else:
        for pair_idx in range(n_pair):
            lower_layer = 2 * pair_idx + 1
            upper_layer = lower_layer + 1
            for idx in range(n_width_first):
                include(
                    D / 2.0 + idx * p + _layer_y_shift(lower_layer, p),
                    flat_half_width if idx == 0 else lean_half_width,
                    D / 2.0,
                    D / 2.0,
                )
                include(p / 2.0 + idx * p + _layer_y_shift(upper_layer, p), lean_half_width, D / 2.0, D / 2.0)

    left_gap = max(0.0, y_min)
    right_gap = max(0.0, W_c - y_max)
    if stack_model == "base_layer_pairs" and n_width_first > 1:
        first_layer_left = min(D / 2.0 + idx * p - lean_half_width for idx in range(1, n_width_first))
        first_layer_right = max(D / 2.0 + idx * p + lean_half_width for idx in range(1, n_width_first))
        left_gap = max(0.0, first_layer_left)
        right_gap = max(0.0, W_c - first_layer_right)
    edge_left_fit = left_gap >= SW
    edge_right_fit = right_gap >= SW

    return {
        "cross_y_min": y_min,
        "cross_y_max": y_max,
        "edge_left_gap": left_gap,
        "edge_right_gap": right_gap,
        "edge_left_fit": edge_left_fit,
        "edge_right_fit": edge_right_fit,
        "edge_start_z": side_start_z,
        "edge_columns": int(edge_left_fit) + int(edge_right_fit),
    }


def calculate_loading(
    spec: str,
    W_c: float = DEFAULT_W_C,
    H_c: float = DEFAULT_H_C,
    L_c: float = DEFAULT_L_C,
    D_base: float = DEFAULT_D_BASE,
    p_base: float = DEFAULT_P_BASE,
    c_base: float = DEFAULT_C_BASE,
    alpha_deg: float = 30.0,
    theta_deg: float = 30.0,
    l_mode: LMode = "scaled_p",
    manual_l: float = 45.0,
    height_tolerance: float = 1.5,
    n_pair_mode: PairMode = "tolerance",
    stack_model: StackModel = "base_layer_pairs",
) -> Dict[str, float | int | str | bool | TireSpec]:
    """Calculate cross-stack capacity and expose all formula intermediates.

    The implementation follows the report formula:
    ``H_pair = 2 * D * sin(alpha) - (L - a * cos(theta)) * tan(theta)``.
    """

    tire = parse_tire_spec(spec)
    _validate_positive_dimensions(W_c=W_c, H_c=H_c, L_c=L_c)
    _validate_positive_dimensions(D_base=D_base, p_base=p_base, c_base=c_base)

    if height_tolerance < 0:
        raise ValueError("height tolerance는 0 이상이어야 합니다.")

    r = tire.D / D_base
    p = p_base * r
    c = c_base * r
    L = resolve_l_value(l_mode, p, manual_l)

    if c > tire.d:
        raise ValueError("접촉 현 길이 c가 림 직경 d보다 큽니다. 공식의 sqrt 항이 유효하지 않습니다.")
    if p <= 0 or c <= 0 or L <= 0:
        raise ValueError("p, c, L은 모두 양수여야 합니다.")

    n_width_first = _calculate_n_width(tire.D, p, W_c)
    n_width_even = math.floor((W_c - p / 2.0) / p)
    n_width_odd = math.floor(W_c / p)
    n_width_tilted = min(n_width_even, n_width_odd)
    n_width = min(n_width_first, n_width_tilted)
    if n_width <= 0:
        raise ValueError("???? ?? ??? ???? ?? ? ?? ??? ??????.")

    W1 = tire.D + (n_width - 1) * p
    W2 = n_width * p + p / 2.0
    W3 = n_width * p
    gap = W_c - W2

    alpha = math.radians(alpha_deg)
    theta = math.radians(theta_deg)
    R = tire.d / 2.0
    sqrt_term = R**2 - (c / 2.0) ** 2
    if sqrt_term < 0:
        raise ValueError("R^2 - (c/2)^2 값이 음수입니다. c와 d 입력을 확인하세요.")

    a = R - math.sqrt(sqrt_term)
    one_len = a * math.cos(theta)
    two_len = L - one_len
    overlap_height = two_len * math.tan(theta)
    H_pair = 2.0 * tire.D * math.sin(alpha) - overlap_height

    if H_pair <= 0:
        raise ValueError("H_pair가 0 이하입니다. alpha, theta, L 값을 확인하세요.")

    if stack_model not in ("base_layer_pairs", "report_pairs"):
        raise ValueError("stack_model은 base_layer_pairs 또는 report_pairs여야 합니다.")

    first_layer_height = max(tire.SW, tire.D * math.sin(alpha)) if stack_model == "base_layer_pairs" else 0.0
    available_pair_height = H_c - first_layer_height
    if available_pair_height < 0:
        raise ValueError("컨테이너 높이가 1층 기준 타이어 높이보다 작습니다.")

    half_pair_height = H_pair / 2.0
    rows_after_first_strict = max(0, math.floor(available_pair_height / half_pair_height))
    rows_after_first_tolerance = max(
        0, math.floor((available_pair_height + height_tolerance) / half_pair_height)
    )
    n_pair_strict = rows_after_first_strict // 2
    n_pair_tolerance = rows_after_first_tolerance // 2
    top_single_layer_strict = rows_after_first_strict % 2
    top_single_layer_tolerance = rows_after_first_tolerance % 2
    n_pair = n_pair_tolerance if n_pair_mode == "tolerance" else n_pair_strict
    top_single_layer = (
        top_single_layer_tolerance if n_pair_mode == "tolerance" else top_single_layer_strict
    )
    if n_pair_mode not in ("strict", "tolerance"):
        raise ValueError("n_pair mode는 strict 또는 tolerance여야 합니다.")
    if n_pair <= 0:
        raise ValueError("컨테이너 높이가 부족해 한 쌍도 적재할 수 없습니다.")

    if stack_model != "base_layer_pairs":
        top_single_layer = 0
        top_single_layer_strict = 0
        top_single_layer_tolerance = 0
    selected_stack_height = first_layer_height + n_pair * H_pair + top_single_layer * half_pair_height
    strict_stack_height = (
        first_layer_height + n_pair_strict * H_pair + top_single_layer_strict * half_pair_height
    )
    tolerance_stack_height = (
        first_layer_height + n_pair_tolerance * H_pair + top_single_layer_tolerance * half_pair_height
    )
    selected_height_margin = H_c - selected_stack_height
    strict_height_margin = H_c - strict_stack_height
    tolerance_height_margin = H_c - tolerance_stack_height

    if stack_model == "base_layer_pairs":
        n_cross_rows = 1 + n_pair * 2 + top_single_layer
        N_cross = n_width * n_cross_rows
    else:
        N_cross = n_width * n_pair * 2
        n_cross_rows = n_pair * 2

    h3_edge = first_layer_height + H_pair
    h_need_edge = tire.SW + tire.D - 2.0 * a
    edge_left_fit = h_need_edge < h3_edge
    edge_right_fit = False
    edge_columns = 1 if edge_left_fit else 0
    edge_start_z = tire.SW
    n_edge_H = edge_columns
    N_edge = edge_columns

    N_face = N_cross + N_edge
    q_L = tire.D
    n_L = math.floor(L_c / q_L)
    if n_L <= 0:
        raise ValueError("컨테이너 길이가 부족해 길이 방향 반복이 불가능합니다.")

    N_total = N_face * n_L
    container_fit = W1 <= W_c and W2 <= W_c and W3 <= W_c and selected_stack_height <= H_c + (
        height_tolerance if n_pair_mode == "tolerance" else 0.0
    )

    return {
        "spec": tire.spec,
        "tire": tire,
        "SW_mm": tire.SW_mm,
        "aspect": tire.aspect,
        "rim_inch": tire.rim_inch,
        "SW": tire.SW,
        "d": tire.d,
        "D": tire.D,
        "W_c": W_c,
        "H_c": H_c,
        "L_c": L_c,
        "D_base": D_base,
        "p_base": p_base,
        "c_base": c_base,
        "r": r,
        "p": p,
        "c": c,
        "L": L,
        "l_mode": l_mode,
        "alpha_deg": alpha_deg,
        "theta_deg": theta_deg,
        "n_width": n_width,
        "n_width_first": n_width_first,
        "n_width_even": n_width_even,
        "n_width_odd": n_width_odd,
        "n_width_tilted": n_width_tilted,
        "W1": W1,
        "W2": W2,
        "W3": W3,
        "gap": gap,
        "cross_y_min": 0.0,
        "cross_y_max": W2,
        "edge_left_gap": p / 2.0,
        "edge_right_gap": W_c - W2,
        "edge_left_fit": edge_left_fit,
        "edge_right_fit": edge_right_fit,
        "edge_start_z": edge_start_z,
        "h3_edge": h3_edge,
        "h_need_edge": h_need_edge,
        "R": R,
        "a": a,
        "one_len": one_len,
        "two_len": two_len,
        "overlap_height": overlap_height,
        "H_pair": H_pair,
        "height_tolerance": height_tolerance,
        "n_pair_strict": n_pair_strict,
        "n_pair_tolerance": n_pair_tolerance,
        "n_pair": n_pair,
        "top_single_layer": top_single_layer,
        "half_pair_height": half_pair_height,
        "n_pair_mode": n_pair_mode,
        "stack_model": stack_model,
        "first_layer_height": first_layer_height,
        "selected_stack_height": selected_stack_height,
        "n_cross_rows": n_cross_rows,
        "strict_height_margin": strict_height_margin,
        "tolerance_height_margin": tolerance_height_margin,
        "selected_height_margin": selected_height_margin,
        "N_cross": N_cross,
        "n_edge_H": n_edge_H,
        "edge_columns": edge_columns,
        "N_edge": N_edge,
        "N_face": N_face,
        "q_L": q_L,
        "n_L": n_L,
        "N_total": N_total,
        "container_fit": container_fit,
    }


def _calculate_n_width(D: float, p: float, W_c: float) -> int:
    """Return the largest n satisfying D + (n - 1) * p <= W_c."""

    if D <= 0 or p <= 0 or W_c <= 0:
        return 0
    if D > W_c:
        return 0
    return 1 + math.floor((W_c - D) / p)


def _validate_positive_dimensions(**values: float) -> None:
    for name, value in values.items():
        if value <= 0:
            raise ValueError(f"{name} 값은 양수여야 합니다.")

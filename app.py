import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from calc import calculate_loading


st.set_page_config(page_title="40HFT Tire Cross Stack", layout="wide")


# =========================================================
# 1) Table helpers
# =========================================================
def _format_value(value):
    if isinstance(value, float):
        return f"{value:,.2f}"
    return value


def _safe_get(result, key, default=None):
    return result[key] if key in result else default


def _metrics_table(result):
    rows = [
        ("입력 타이어 규격", result["spec"], ""),
        ("SW", result["SW"], "cm"),
        ("림 직경 d", result["d"], "cm"),
        ("외경 D", result["D"], "cm"),
        ("스케일 r", result["r"], ""),
        ("폭 방향 전진량 p", result["p"], "cm"),
        ("접촉 현 길이 c", result["c"], "cm"),
        ("높이 공식 L", result["L"], "cm"),
        ("n_width", result["n_width"], "개"),
        ("n_width_tilted", result["n_width_tilted"], "개"),
        ("Row 1 폭 W1", result["W1"], "cm"),
        ("Row 2 폭 W2", result["W2"], "cm"),
        ("gap", result["gap"], "cm"),
        ("left edge actual gap", result["edge_left_gap"], "cm"),
        ("right edge actual gap", result["edge_right_gap"], "cm"),
        ("left edge fit", "yes" if result["edge_left_fit"] else "no", ""),
        ("right edge fit", "yes" if result["edge_right_fit"] else "no", ""),
        ("edge start z", result["edge_start_z"], "cm"),
        ("edge h_need", result["h_need_edge"], "cm"),
        ("edge h3", result["h3_edge"], "cm"),
        ("a", result["a"], "cm"),
        ("one_len", result["one_len"], "cm"),
        ("two_len", result["two_len"], "cm"),
        ("overlap_height", result["overlap_height"], "cm"),
        ("H_pair", result["H_pair"], "cm"),
        ("top single layer", result["top_single_layer"], "?"),
        ("half pair height", result["half_pair_height"], "cm"),
        ("strict 기준 n_pair", result["n_pair_strict"], "쌍"),
        ("tolerance 적용 n_pair", result["n_pair_tolerance"], "쌍"),
        ("최종 사용 n_pair", result["n_pair"], "쌍"),
        ("stack model", result["stack_model"], ""),
        ("1층 기준 높이", result["first_layer_height"], "cm"),
        ("선택 기준 총 stack 높이", result["selected_stack_height"], "cm"),
        ("크로스 row 수", result["n_cross_rows"], "줄"),
        ("strict 초과/여유 높이", result["strict_height_margin"], "cm"),
        ("tolerance 초과/여유 높이", result["tolerance_height_margin"], "cm"),
        ("선택 기준 초과/여유 높이", result["selected_height_margin"], "cm"),
        ("N_cross", result["N_cross"], "개/단면"),
        ("edge column 수", result["edge_columns"], "열"),
        ("n_edge_H", result["n_edge_H"], "개/측면"),
        ("N_edge", result["N_edge"], "개/단면"),
        ("N_face", result["N_face"], "개/단면"),
        ("n_L", result["n_L"], "층"),
        ("N_total", result["N_total"], "개"),
        ("컨테이너 적합 여부", "적합" if result["container_fit"] else "초과", ""),
    ]
    return pd.DataFrame(rows, columns=["항목", "값", "단위"])


# =========================================================
# 2) Visualization helpers
#    계산 로직은 건드리지 않고, 3D tire mesh 표시만 app.py 안에서 개선함.
# =========================================================
def _rotation_x(angle_rad):
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([
        [1, 0, 0],
        [0, c, -s],
        [0, s, c],
    ])


def _make_tire_segment(center, D, d, SW, tilt_deg=0.0, u0=0.0, u1=2 * math.pi, nu=18, nv=12):
    """
    도넛형 타이어 segment mesh 생성.
    - X축: 타이어 폭 방향
    - Y-Z 평면: 타이어 외경/내경이 보이는 링 평면
    - D: 외경, d: 림/내경, SW: 단면폭
    """
    outer_r = D / 2.0
    inner_r = d / 2.0

    # 링 중심 반지름과 고무 두께. 단면은 완전 원형이 아니라 SW 방향으로 긴 타원형으로 근사.
    R = (outer_r + inner_r) / 2.0
    r_radial = (outer_r - inner_r) / 2.0
    r_width = SW / 2.0

    u = np.linspace(u0, u1, nu)
    v = np.linspace(0, 2 * math.pi, nv)
    uu, vv = np.meshgrid(u, v, indexing="ij")

    x = r_width * np.sin(vv)
    radial = R + r_radial * np.cos(vv)
    y = radial * np.cos(uu)
    z = radial * np.sin(uu)

    pts = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=0)
    pts = _rotation_x(math.radians(tilt_deg)) @ pts
    pts[0, :] += center[0]
    pts[1, :] += center[1]
    pts[2, :] += center[2]

    x = pts[0, :].reshape(nu, nv)
    y = pts[1, :].reshape(nu, nv)
    z = pts[2, :].reshape(nu, nv)

    faces_i, faces_j, faces_k = [], [], []
    for a in range(nu - 1):
        for b in range(nv - 1):
            p0 = a * nv + b
            p1 = (a + 1) * nv + b
            p2 = (a + 1) * nv + (b + 1)
            p3 = a * nv + (b + 1)
            faces_i += [p0, p0]
            faces_j += [p1, p2]
            faces_k += [p2, p3]

    # v 방향 닫기
    for a in range(nu - 1):
        p0 = a * nv + (nv - 1)
        p1 = (a + 1) * nv + (nv - 1)
        p2 = (a + 1) * nv
        p3 = a * nv
        faces_i += [p0, p0]
        faces_j += [p1, p2]
        faces_k += [p2, p3]

    return x.ravel(), y.ravel(), z.ravel(), faces_i, faces_j, faces_k


def _add_tire(fig, center, D, d, SW, tilt_deg, color, opacity=0.94, name="tire", quality="medium", weave_phase=0):
    """
    Plotly Mesh3d의 투명도 정렬 한계를 줄이기 위해 타이어를 여러 angular segment로 쪼개서 그림.
    이렇게 하면 겹침부에서 한 타이어 전체가 무조건 위에 뜨는 현상이 완화됨.
    """
    if quality == "low":
        seg_count, nu, nv = 8, 10, 8
    elif quality == "high":
        seg_count, nu, nv = 20, 18, 16
    else:
        seg_count, nu, nv = 14, 14, 12

    segments = []
    for s in range(seg_count):
        u0 = 2 * math.pi * s / seg_count
        u1 = 2 * math.pi * (s + 1) / seg_count
        order_key = (s + weave_phase) % seg_count
        x, y, z, i, j, k = _make_tire_segment(
            center=center,
            D=D,
            d=d,
            SW=SW,
            tilt_deg=tilt_deg,
            u0=u0,
            u1=u1,
            nu=nu,
            nv=nv,
        )
        segments.append((order_key, x, y, z, i, j, k))

    segments.sort(key=lambda t: t[0])

    for _, x, y, z, i, j, k in segments:
        fig.add_trace(
            go.Mesh3d(
                x=x,
                y=y,
                z=z,
                i=i,
                j=j,
                k=k,
                color=color,
                opacity=opacity,
                flatshading=False,
                lighting=dict(
                    ambient=0.45,
                    diffuse=0.75,
                    specular=0.18,
                    roughness=0.72,
                    fresnel=0.08,
                ),
                lightposition=dict(x=100, y=-200, z=300),
                name=name,
                showscale=False,
                hoverinfo="skip",
            )
        )


def _add_container(fig, W_c, H_c, L_sample):
    x0, x1 = 0, L_sample
    y0, y1 = 0, W_c
    z0, z1 = 0, H_c

    edges = [
        [(x0, y0, z0), (x1, y0, z0)],
        [(x0, y1, z0), (x1, y1, z0)],
        [(x0, y0, z1), (x1, y0, z1)],
        [(x0, y1, z1), (x1, y1, z1)],
        [(x0, y0, z0), (x0, y1, z0)],
        [(x1, y0, z0), (x1, y1, z0)],
        [(x0, y0, z1), (x0, y1, z1)],
        [(x1, y0, z1), (x1, y1, z1)],
        [(x0, y0, z0), (x0, y0, z1)],
        [(x0, y1, z0), (x0, y1, z1)],
        [(x1, y0, z0), (x1, y0, z1)],
        [(x1, y1, z0), (x1, y1, z1)],
    ]

    for a, b in edges:
        fig.add_trace(
            go.Scatter3d(
                x=[a[0], b[0]],
                y=[a[1], b[1]],
                z=[a[2], b[2]],
                mode="lines",
                line=dict(color="rgba(80,80,80,0.45)", width=3),
                hoverinfo="skip",
                showlegend=False,
            )
        )


def _generate_positions(result, render_mode):
    D = float(result["D"])
    SW = float(result["SW"])
    p = float(result["p"])

    W_c = float(_safe_get(result, "W_c", 235.0))
    H_c = float(_safe_get(result, "H_c", 270.0))
    L_c = float(_safe_get(result, "L_c", 1203.0))

    n_width = int(result["n_width"])
    n_width_tilted = int(result["n_width_tilted"])
    n_pair = int(result["n_pair"])

    alpha = float(_safe_get(result, "alpha_deg", 30.0))
    H_pair = float(result["H_pair"])
    first_layer_height = float(_safe_get(result, "first_layer_height", max(SW, D * math.sin(math.radians(alpha)))))

    if render_mode == "front_only":
        x_layers = [SW / 2.0]
    elif render_mode == "sample_layers":
        x_layers = [SW / 2.0, SW * 1.8, SW * 3.1]
    else:
        n_L = int(_safe_get(result, "n_L", 1))
        x_layers = [SW / 2.0 + i * SW * 1.18 for i in range(n_L)]

    positions = []

    for layer_idx, x in enumerate(x_layers):
        # 1층: 기존 계산 기준 유지. 첫 타이어는 평치, 이후 기울임.
        z1 = first_layer_height / 2.0
        for i in range(n_width):
            y = D / 2.0 + i * p
            tilt = 0.0 if i == 0 else alpha
            positions.append({
                "center": (x, y, z1),
                "tilt": tilt,
                "color": "rgba(40,130,80,1)" if layer_idx == 0 else "rgba(120,120,120,1)",
                "phase": i % 4,
            })

        # 2층 이상: +alpha / -alpha가 서로 구멍 쪽으로 맞물리는 것처럼 보이도록 y를 반 칸 이동.
        for k in range(n_pair):
            base = D + k * H_pair
            z_lower = base + H_pair * 0.25
            z_upper = base + H_pair * 0.75

            for i in range(n_width_tilted):
                y_lower = D / 2.0 + i * p
                y_upper = D / 2.0 + i * p + p * 0.45

                positions.append({
                    "center": (x, y_lower, z_lower),
                    "tilt": alpha,
                    "color": "rgba(52,103,155,1)",
                    "phase": (i + k) % 6,
                })
                positions.append({
                    "center": (x, y_upper, z_upper),
                    "tilt": -alpha,
                    "color": "rgba(176,119,35,1)",
                    "phase": (i + k + 3) % 6,
                })

    return positions, W_c, H_c, L_c


def create_stack_figure(
    result,
    render_mode="front_only",
    realistic_tire_mesh=True,
    tread_detail=True,
    mesh_quality="medium",
    tire_opacity=0.94,
    show_container=True,
    show_edge_tires=True,
):
    D = float(result["D"])
    d = float(result["d"])
    SW = float(result["SW"])

    positions, W_c, H_c, L_c = _generate_positions(result, render_mode)

    fig = go.Figure()

    max_x = max(p["center"][0] for p in positions) + SW
    L_sample = max(max_x, SW * 2.2)

    if show_container:
        _add_container(fig, W_c=W_c, H_c=H_c, L_sample=L_sample)

    # 뒤쪽/아래쪽부터 추가하되 segment phase를 섞어 Plotly depth-sort 오류를 완화.
    positions.sort(key=lambda p: (p["center"][0], p["center"][2], p["center"][1], p["phase"]))

    opacity = min(max(float(tire_opacity), 0.88), 0.98)
    for idx, p in enumerate(positions):
        _add_tire(
            fig,
            center=p["center"],
            D=D,
            d=d,
            SW=SW,
            tilt_deg=p["tilt"],
            color=p["color"],
            opacity=opacity,
            quality=mesh_quality,
            weave_phase=p["phase"],
            name=f"tire_{idx}",
        )

    fig.update_layout(
        height=760,
        margin=dict(l=0, r=0, t=20, b=0),
        scene=dict(
            xaxis=dict(title="Length X (cm)", range=[0, L_sample], backgroundcolor="white"),
            yaxis=dict(title="Width Y (cm)", range=[0, W_c], backgroundcolor="white"),
            zaxis=dict(title="Height Z (cm)", range=[0, H_c], backgroundcolor="white"),
            aspectmode="manual",
            aspectratio=dict(x=max(L_sample / W_c, 0.75), y=1, z=H_c / W_c),
            camera=dict(eye=dict(x=1.65, y=-1.65, z=1.15), center=dict(x=0, y=0, z=0)),
        ),
        showlegend=False,
    )
    return fig


# =========================================================
# 3) Streamlit app
# =========================================================
with st.sidebar:
    st.header("Input")
    tire_spec = st.text_input("Tire spec", value="205/55R16")

    st.divider()
    st.caption("시각화 옵션")
    mesh_quality = st.selectbox("Mesh quality", ["low", "medium", "high"], index=1)
    tire_opacity = st.slider("Tire opacity", min_value=0.88, max_value=0.98, value=0.94, step=0.01)
    show_container = st.checkbox("Show container", value=True)
    show_edge_tires = st.checkbox("Show edge tires", value=True)


W_c = 235.0
H_c = 270.0
L_c = 1203.0
alpha_deg = 30.0
theta_deg = 30.0
p_base = 45.0
c_base = 30.0
l_mode = "scaled_p"
manual_l = 45.0
height_tolerance = 1.5
n_pair_mode = "tolerance"
stack_model = "base_layer_pairs"
render_mode = "front_only"
realistic_tire_mesh = True
tread_detail = True
show_calculation_table = True

st.title("40HFT 컨테이너 타이어 꽈배기/크로스 적재 계산기")
st.caption(
    "계산 로직은 기존 수식 기반 모델을 그대로 사용하고, 3D 시각화만 도넛형 segment mesh 방식으로 보완했습니다. "
    "실제 물리 충돌/변형 시뮬레이션은 아닙니다."
)

try:
    result = calculate_loading(
        tire_spec,
        W_c=W_c,
        H_c=H_c,
        L_c=L_c,
        p_base=p_base,
        c_base=c_base,
        alpha_deg=alpha_deg,
        theta_deg=theta_deg,
        l_mode=l_mode,
        manual_l=manual_l,
        height_tolerance=height_tolerance,
        n_pair_mode=n_pair_mode,
        stack_model=stack_model,
    )
except ValueError as exc:
    st.error(str(exc))
    st.stop()

# visualization 쪽에서도 동일한 컨테이너/각도 값을 쓰도록 result에 보강.
result["W_c"] = W_c
result["H_c"] = H_c
result["L_c"] = L_c
result["alpha_deg"] = alpha_deg

summary_cols = st.columns(5)
summary_cols[0].metric("n_width", int(result["n_width"]))
summary_cols[1].metric("n_pair", int(result["n_pair"]))
summary_cols[2].metric("N_face", int(result["N_face"]))
summary_cols[3].metric("n_L", int(result["n_L"]))
summary_cols[4].metric("N_total", f"{int(result['N_total']):,}")

section_tab, total_tab, calc_tab = st.tabs(["단면 적재", "전체 적재량", "계산 결과"])

with section_tab:
    st.subheader("단면 3D tire mesh")
    st.caption(
        "타이어를 하나의 투명 덩어리로 그리지 않고 segment 단위의 도넛형 mesh로 나누어 그립니다. "
        "따라서 겹침부에서 한 타이어가 통째로 위에 떠 보이는 현상이 완화됩니다."
    )
    fig = create_stack_figure(
        result,
        render_mode=render_mode,
        realistic_tire_mesh=realistic_tire_mesh,
        tread_detail=tread_detail,
        mesh_quality=mesh_quality,
        tire_opacity=tire_opacity,
        show_container=show_container,
        show_edge_tires=show_edge_tires,
    )
    st.plotly_chart(fig, use_container_width=True)

with total_tab:
    st.subheader("전체 적재량")
    total_cols = st.columns(4)
    total_cols[0].metric("단면 적재 N_face", f"{int(result['N_face']):,}")
    total_cols[1].metric("길이 방향 n_L", f"{int(result['n_L']):,}")
    total_cols[2].metric("총 적재 N_total", f"{int(result['N_total']):,}")
    total_cols[3].metric("컨테이너", "적합" if result["container_fit"] else "초과")
    st.caption(
        f"N_total = N_face × n_L = {int(result['N_face']):,} × "
        f"{int(result['n_L']):,} = {int(result['N_total']):,}"
    )
    st.caption(
        "아래 그림은 길이 방향 반복 구조를 앞쪽 3줄로 샘플 표시합니다. "
        "전체 적재량은 계산된 n_L 전체를 기준으로 계산됩니다."
    )
    total_fig = create_stack_figure(
        result,
        render_mode="sample_layers",
        realistic_tire_mesh=True,
        tread_detail=False,
        mesh_quality="low",
        tire_opacity=0.94,
        show_container=show_container,
        show_edge_tires=show_edge_tires,
    )
    st.plotly_chart(total_fig, use_container_width=True)

    if st.button("전체 3D 렌더링", help="전체 mesh 렌더링은 타이어 수가 많아 느릴 수 있습니다."):
        full_fig = create_stack_figure(
            result,
            render_mode="full",
            realistic_tire_mesh=True,
            tread_detail=False,
            mesh_quality="low",
            tire_opacity=0.92,
            show_container=show_container,
            show_edge_tires=show_edge_tires,
        )
        st.plotly_chart(full_fig, use_container_width=True)

with calc_tab:
    if show_calculation_table:
        st.subheader("계산 결과")
        table = _metrics_table(result)
        display_table = table.copy()
        display_table["값"] = display_table["값"].map(_format_value)
        st.dataframe(display_table, use_container_width=True, hide_index=True)

with st.expander("205/55R16 검산 포인트", expanded=False):
    st.write(
        "기본 stack model은 `base_layer_pairs`입니다. 1층을 별도 기준층으로 두고, "
        "2-3층/4-5층/...만 꽈배기 pair로 계산합니다. "
        "사이드 타이어는 실제 좌우 빈 폭이 SW 이상일 때만 계산/표시합니다."
    )

with st.expander("층별 꽈배기 수식", expanded=False):
    st.markdown(
        """
        기본 계산은 `1층 단독 + (2,3층), (4,5층), ... 꽈배기 pair` 모델입니다.
        1층만 첫 번째 타이어를 편평하게 두고, 2층 이상은 첫 타이어부터 모두 기울어진 타이어로 계산합니다.

        ```text
        H_pair = 2 * D * sin(alpha) - overlap_height

        overlap_height = (L - a*cos(theta)) * tan(theta)
        a = d/2 - sqrt((d/2)^2 - (c/2)^2)
        ```

        1층:

        ```text
        first_layer_height = max(SW, D * sin(alpha))
        z_1_center = first_layer_height / 2
        y_i = D/2 + i*p
        i = 0 orientation = 0도, i >= 1 orientation = +alpha
        n_width_first = max n such that D + (n - 1)*p <= W_c
        ```

        2,3층 pair (`k=1`):

        ```text
        z_pair_base_1 = D
        z_2_center = z_pair_base_1 + H_pair/4
        z_3_center = z_pair_base_1 + 3*H_pair/4
        ```

        4,5층 pair (`k=2`):

        ```text
        z_pair_base_2 = D + H_pair
        z_4_center = z_pair_base_2 + H_pair/4
        z_5_center = z_pair_base_2 + 3*H_pair/4
        ```

        일반식 (`k=1, 2, 3, ...`):

        ```text
        z_pair_base_k = D + (k-1)*H_pair
        z_lower_center = z_pair_base_k + H_pair/4
        z_upper_center = z_pair_base_k + 3*H_pair/4

        Row lower: orientation = +alpha for every i
        Row upper: orientation = -alpha for every i
        n_width_tilted = floor(W_c / p)
        ```

        이 모델은 1층을 별도로 빼므로 기존 보고서 재현용 `report_pairs` 모델보다 총량이 줄어듭니다.
        `report_pairs`는 기존 보고서의 pair 계산 방식만 비교할 때 사용합니다.
        사이드 타이어는 실제 좌우 빈 폭이 SW 이상일 때만 추가합니다.
        """
    )

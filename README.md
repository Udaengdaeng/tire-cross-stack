# 40HFT Tire Cross Stack Visualizer

Streamlit and Plotly app for calculating and visualizing formula-based tire cross stacking in a 40HFT high-cube container.

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

## Deploy With Streamlit Community Cloud

1. Create a new GitHub repository.
2. Push these project files to the repository root:
   - `app.py`
   - `calc.py`
   - `mesh_tire.py`
   - `visualize.py`
   - `requirements.txt`
   - `README.md`
3. Open Streamlit Community Cloud: https://share.streamlit.io/
4. Click **New app**.
5. Select the GitHub repository and branch.
6. Set **Main file path** to:

```text
app.py
```

7. Click **Deploy**.

No API key or Streamlit secret is required for this app.

## Files

- `app.py`: Streamlit UI
- `calc.py`: report formula calculations
- `mesh_tire.py`: realistic hollow tire mesh generation
- `visualize.py`: container and tire placement figure generation
- `test_calc.py`: calculation regression checks

## Default Units And Container

All calculation units are centimeters. Tire section width from the spec string is converted from mm to cm.

Default 40HFT high-cube container:

```python
W_c = 235.0
H_c = 270.0
L_c = 1203.0
```

The app also includes a `40FT general` preset with `H_c = 239.5`.

## Formula Summary

Tire spec parsing:

```python
SW = SW_mm / 10
d = rim_inch * 2.54
D = rim_inch * 2.54 + 2 * SW * (aspect / 100)
```

Scaling from base tire:

```python
r = D_target / D_base
p = p_base * r
c = c_base * r
```

Height formula:

```python
R = d / 2
a = R - sqrt(R**2 - (c / 2)**2)
one_len = a * cos(theta)
two_len = L - a * cos(theta)
overlap_height = (L - a * cos(theta)) * tan(theta)
H_pair = 2 * D * sin(alpha) - overlap_height
```

By default, `L mode=scaled_p`, so the height-formula repeat length follows the target tire:

```python
L = p
```

Use `L mode=base_45` only when reproducing the older report check with fixed `L = 45.0 cm`.

Capacity in the default `base_layer_pairs` model:

```python
n_width_first = max n such that D + (n - 1) * p <= W_c
n_width_tilted = floor(W_c / p)

first_layer_height = max(SW, D * sin(alpha))
n_pair = floor((H_c - first_layer_height + height_tolerance) / H_pair)
N_cross = n_width_first + n_width_tilted * n_pair * 2

left_edge_gap, right_edge_gap = visible free side strips after tire mesh extents
edge_columns = count of side strips where gap >= SW
N_edge = edge_columns * floor(H_c / D)

N_face = N_cross + N_edge
n_L = floor(L_c / D)
N_total = N_face * n_L
```

The app also provides `report_pairs`, which keeps the original report-style
`N_cross = n_width * n_pair * 2` formula. Edge tires are still added only when
the actual left/right visible side strip is at least SW.

## 205/55R16 Check

With 40FT general, `L mode=base_45`, `height_tolerance=1.5`, tolerance pair mode, and `stack_model=report_pairs`:

| Item | Value |
| --- | ---: |
| D | about 63.2 cm |
| d | about 40.64 cm |
| c | about 28.4 cm |
| H_pair | about 40.11 cm |
| n_width | 5 |
| n_pair | 6 |
| N_cross | 60 |
| N_edge | 0 |
| N_face | 60 |
| n_L | 19 |
| N_total | 1140 |

With the default 40HFT high-cube preset and `stack_model=report_pairs`, the taller height changes only the height-dependent counts:

| Item | Value |
| --- | ---: |
| H_c | 269.5 cm |
| n_pair | 6 |
| N_cross | 60 |
| N_edge | 0 |
| N_face | 60 |
| n_L | 19 |
| N_total | 1140 |

With the default 40HFT high-cube preset, default `L mode=scaled_p`, and default `base_layer_pairs` model, the same tire is counted as a separate first layer plus five upper twisted pairs:

| Item | Value |
| --- | ---: |
| L | about 42.57 cm |
| H_pair | about 41.50 cm |
| n_width_first | 5 |
| n_width_tilted | 5 |
| n_pair | 5 |
| N_cross | 55 |
| N_edge | 1 |
| N_face | 61 |
| n_L | 19 |
| N_total | 1159 |

## Visualization

The realistic mode uses a hollow tire mesh:

- local x axis is tire width
- local y-z plane is the tire face
- outer radius is `D / 2`
- inner radius is `d / 2`
- width is `SW`
- tread surface, inner hole surface, and sidewall annuli are modeled

The tire mesh is rotated and translated into the container coordinate system:

- x axis: container length
- y axis: container width
- z axis: container height

## Caution

This is not a physics-engine collision simulation. It is a 3D approximation based on the report formulas. Before real loading, these measured values should be verified:

- actual tilt angle `alpha`
- actual repeated angle `theta`
- actual contact chord `c`
- actual width advance `p`
- actual length repeat pitch `q_L`

## Latest Report Alignment

The 2026-05-29 report uses `40hFT` with `H_c = 270.0`, even rows shifted by `p/2`, a top single row after five full pairs, and `N_edge = 1 or 0`. For `205/55R16`, this gives `N_cross = 60`, `N_edge = 1`, `N_face = 61`, `n_L = 19`, and `N_total = 1159`.

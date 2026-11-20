# MATLAB layer

This folder adds a MATLAB layer to the Python package `conjce` in the repository root. It

1. ports the core model (collision probability, burn-to-miss rule, cost tables, the coordinator's linear program);
2. checks the linear burn-to-miss rule against a two-body propagation of both satellites;
3. holds a Simulink model of one conjunction event (`simulink/conjunction_sim.slx`);
4. renders four animations.

## Requirements

- MATLAB with the Optimization Toolbox (`linprog`).
- Simulink, to open the model in `simulink/`.

Everything here was run with MATLAB R2024b (24.2) and Simulink 24.2 on Windows.

## Where the numbers come from

`export_matlab.py` in the repository root writes two files from the Python package:

| File | Content |
|---|---|
| `params.json` | Every model parameter: constants, the baseline conjunction, the settings of the two-body check, the twelve gallery conjunctions |
| `expected.json` | Reference numbers computed in Python: collision probability per joint action, cost tables, the coordinator's rules, the mixed equilibrium, a two-body reference, the learning curves |

MATLAB reads its parameters from `params.json`, so no parameter is typed twice.

## Layout

| Path | What it is |
|---|---|
| `core/` | Port of the core model: `pc2d.m` (collision probability), `miss_shift.m` (linear burn-to-miss rule), `cost_matrices.m` (cost tables of the game), `solve_ce.m` (the coordinator's linear program, with `linprog`), `obedience_system.m`, `mixed_nash.m`, and the loaders of the two JSON files |
| `twobody/` | Two-body check: states at the encounter, two propagators (`ode113` and a universal-variable Kepler solver), closest approach, the second-order miss formula, a Monte Carlo estimate of the collision probability |
| `simulink/` | `conjunction_sim.slx`, the Simulink block diagram of one conjunction event |
| `viz/` | One style file (`viz_style.m`), helpers, and one `make_*_animation.m` per animation |
| `run_twobody_check.m`, `run_visuals.m`, `setup_paths.m` | Entry points |

## Run

From the repository root:

```
.venv\Scripts\python export_matlab.py
matlab -batch "run_twobody_check" -sd matlab
matlab -batch "run_visuals" -sd matlab
```

| Step | What it does |
|---|---|
| `export_matlab.py` | Writes `params.json` and `expected.json` from the Python package |
| `run_twobody_check` | Propagates the nine joint actions, recomputes probabilities and incentives, runs the Monte Carlo check, writes `results/twobody_check.json` |
| `run_visuals` | Renders the four animations into `figures/` (MP4, GIF, final-frame PNG) and writes `results/viz_*.json` |

## Animations

| Animation | Files in `figures/` | What it shows |
|---|---|---|
| Encounter plane | `encounter_plane.mp4`, `.gif`, `.png` | The pass point of satellite 2 and its uncertainty ellipse in the encounter plane; each burn moves it away from satellite 1 and the collision probability falls |
| Orbits in 3D | `orbits_3d.mp4`, `.gif`, `.png` | Both orbits through the closest approach, without burns and with the coordinator's most frequent recommendation, and the distance between the satellites |
| Gallery | `gallery.mp4`, `.gif`, `.png` | Twelve conjunction geometries side by side, each with the coordinator's recommendation and its outcome |
| Learning | `learning.mp4`, `.gif`, `.png` | Realised coordinator cost against the number of past events operators have seen, for four kinds of rule |

All four use the style file `viz/viz_style.m`, which holds the colours and type sizes of the Python figures (`conjce/figures.py`).
